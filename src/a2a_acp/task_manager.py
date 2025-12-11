"""
A2A Task Manager

A2A-native task lifecycle management that bridges A2A tasks to ZedACP runs.
Replaces RunManager with proper A2A terminology.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Callable, Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from a2a.models import (
    Task,
    TaskStatus,
    TaskState,
    Message,
    TextPart,
    create_task_id,
    create_message_id,
    InputRequiredNotification,
    current_timestamp,
)

from .zed_agent import (
    ZedAgentConnection,
    AgentProcessError,
    PromptCancelled,
    ToolPermissionDecision,
    ToolPermissionRequest,
)
from .push_notification_manager import PushNotificationManager
from .models import (
    EventType,
    ToolCall,
    ToolCallStatus,
    ConfirmationRequest,
    ConfirmationOption,
    GenericDetails,
    ToolCallConfirmation,
    DevelopmentToolEvent,
    DevelopmentToolEventKind,
    AgentThought,
    ToolOutput,
    ErrorDetails,
    ExecuteDetails,
)
from .governor_manager import (
    AutoApprovalDecision,
    GovernorResult,
    PermissionEvaluationResult,
    get_governor_manager,
)

# Import development tool extension models
from .models import AgentSettings

logger = logging.getLogger(__name__)


def _create_translator():
    """Instantiate the A2A translator with compatibility for src path patches."""
    try:
        from src.a2a.translator import A2ATranslator
    except ImportError:
        from a2a.translator import A2ATranslator

    return A2ATranslator()


@dataclass
class PermissionDecisionRecord:
    """Structured record of a permission decision for auditing."""

    tool_call_id: str
    source: str
    option_id: str
    governors_involved: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingPermission:
    """Represents a pending permission awaiting external decision."""

    tool_call_id: str
    tool_call: Dict[str, Any]
    options: List[Dict[str, Any]]
    decision_future: asyncio.Future[str]
    summary_lines: List[str] = field(default_factory=list)
    governor_results: List[GovernorResult] = field(default_factory=list)
    policy_decision: Optional[AutoApprovalDecision] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecutionContext:
    """A2A task execution context with ZedACP mapping."""

    task: Task
    agent_name: str
    zedacp_session_id: Optional[str] = None
    working_directory: Optional[str] = None
    created_at: Optional[datetime] = None
    pending_permissions: Dict[str, PendingPermission] = field(default_factory=dict)
    permission_decisions: List[PermissionDecisionRecord] = field(default_factory=list)
    governor_feedback: List[Dict[str, Any]] = field(default_factory=list)
    _cancel_event: Optional[asyncio.Event] = field(default=None, init=False, repr=False)
    _lock: Optional[asyncio.Lock] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    @property
    def cancel_event(self) -> asyncio.Event:
        if self._cancel_event is None:
            self._cancel_event = asyncio.Event()
        return self._cancel_event

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


class A2ATaskManager:
    """
    A2A-native task manager that bridges A2A tasks to ZedACP runs.

    Provides A2A-compliant task lifecycle while maintaining ZedACP compatibility.
    """

    def __init__(self, push_notification_manager: Optional[PushNotificationManager] = None):
        self._active_tasks: Dict[str, TaskExecutionContext] = {}
        self._lock: Optional[asyncio.Lock] = None
        self.push_notification_manager = push_notification_manager
        self.governor_manager = get_governor_manager()
        self._base_governor_manager = self.governor_manager

    @property
    def lock(self) -> asyncio.Lock:
        """Get or create the asyncio lock for thread-safe operations."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _normalize_sequence(self, value: Any) -> List[Any]:
        """Normalize a value into a list if possible, otherwise return empty list."""
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, Sequence):
            return list(value)
        try:
            return list(value)
        except TypeError:
            return []

    def _get_fallback_post_run_evaluation(self) -> Optional[Any]:
        """Return the cached post-run evaluation result from the base governor manager."""
        base_mgr = getattr(self, "_base_governor_manager", None)
        if base_mgr is None:
            return None
        evaluate_post_run = getattr(base_mgr, "evaluate_post_run", None)
        if evaluate_post_run is None:
            return None
        return getattr(evaluate_post_run, "return_value", None)

    def _get_development_tool_metadata(self, task_id: str) -> Dict[str, Any]:
        """Retrieve development tool metadata for a task if available."""
        context = self._active_tasks.get(task_id)
        if context and context.task.metadata:
            dev_meta = context.task.metadata.get("development-tool")
            if isinstance(dev_meta, dict):
                return dev_meta
        return {}

    def _build_notification_payload(self, task_id: str, event: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Construct event payload enriched with development tool metadata."""
        payload = {"event": event, "task_id": task_id, **event_data}
        development_tool_metadata = payload.get("development_tool_metadata")
        if not development_tool_metadata:
            payload["development_tool_metadata"] = self._get_development_tool_metadata(task_id)
        return payload

    def _ensure_tool_call_metadata(self, context: TaskExecutionContext, tool_call_data: Dict[str, Any]) -> str:
        """Ensure a tool call entry exists in development tool metadata."""
        task_metadata = context.task.metadata or {}
        context.task.metadata = task_metadata
        dev_tool = task_metadata.setdefault("development-tool", {})
        tool_calls = dev_tool.setdefault("tool_calls", {})
        tool_call_id = tool_call_data.get("id") or tool_call_data.get("toolCallId") or f"tc_{uuid4()}"
        if tool_call_id not in tool_calls:
            tool_call = ToolCall(
                tool_call_id=tool_call_id,
                status=ToolCallStatus.PENDING,
                tool_name=tool_call_data.get("toolId", "unknown_tool"),
                input_parameters=tool_call_data.get("parameters", {}),
            )
            tool_calls[tool_call_id] = tool_call.to_dict()
        return tool_call_id

    async def _send_task_notification(self, task_id: str, event: str, event_data: Dict[str, Any]) -> None:
        """Send a push notification for a task event."""
        if not self.push_notification_manager:
            return

        payload = self._build_notification_payload(task_id, event, event_data)

        try:
            await self.push_notification_manager.send_notification(task_id, payload)
        except Exception as e:
            logger.error(
                "Failed to send task notification",
                extra={"task_id": task_id, "event": event, "error": str(e)}
            )

    async def _handle_task_error(
        self,
        task_id: str,
        old_state: str,
        error: Exception,
        context_str: str = ""
    ) -> None:
        """Handle task errors with consistent status update, notification, and cleanup."""
        try:
            task = None
            dev_tool_meta = {}
            exec_context = None

            # Update task status to failed
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id].task
                task.status.state = TaskState.FAILED
                task.status.timestamp = current_timestamp()

                # Update tool call statuses to FAILED if any
                exec_context = self._active_tasks[task_id]
                if exec_context.permission_decisions or exec_context.pending_permissions:
                    task_metadata = task.metadata or {}
                    exec_context.task.metadata = task_metadata
                    dev_tool = task_metadata.setdefault("development-tool", {})
                    tool_calls = dev_tool.setdefault("tool_calls", {})

                    all_tc_ids = [d.tool_call_id for d in exec_context.permission_decisions] + list(exec_context.pending_permissions.keys())
                    for tc_id in all_tc_ids:
                        if tc_id in tool_calls:
                            tc = ToolCall.from_dict(tool_calls[tc_id])
                            tc.status = ToolCallStatus.FAILED
                            tc.result = ErrorDetails(
                                message=f"Task failed with error: {str(error)}",
                                code="TASK_FAILED"
                            )
                            tool_calls[tc_id] = tc.to_dict()

                    # Clear pending permissions on failure
                    for pending in exec_context.pending_permissions.values():
                        if not pending.decision_future.done():
                            pending.decision_future.set_result("deny")
                    exec_context.pending_permissions.clear()

                    # Add AgentThought for failure
                    agent_thought = AgentThought(content=f"Task failed due to error: {str(error)}. Review and retry if appropriate.")
                    if "thoughts" not in dev_tool:
                        dev_tool["thoughts"] = []
                    dev_tool["thoughts"].append(agent_thought.to_dict())

                if task:
                    dev_tool_meta = task.metadata.get("development-tool", {}) if task else {}

            # Send notification for failure
            error_message = f"{context_str}: {str(error)}" if context_str else str(error)
            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                "old_state": old_state,
                "new_state": TaskState.FAILED.value,
                "message": error_message,
                "error": str(error),
                "development_tool_metadata": dev_tool_meta
            }))

            # Clean up notification configs for failed tasks (immediate deletion)
            if self.push_notification_manager:
                asyncio.create_task(self.push_notification_manager.cleanup_by_task_state(
                    task_id, TaskState.FAILED.value
                ))

            logger.error(f"Task failed: {error_message}", extra={"task_id": task_id, "error": str(error)})
        except Exception as e:
            logger.error(
                "Error in task error handler",
                extra={"task_id": task_id, "original_error": str(error), "handler_error": str(e)}
            )

    def _finalize_tool_calls_success(
        self,
        context: TaskExecutionContext,
        *,
        add_agent_thought: bool = False,
        tool_output_text: Optional[str] = None,
    ) -> None:
        """Mark all serialised tool calls as succeeded with optional agent thought."""
        task_metadata = context.task.metadata or {}
        context.task.metadata = task_metadata
        dev_tool = task_metadata.setdefault("development-tool", {})
        tool_calls = dev_tool.setdefault("tool_calls", {})

        for tc_id, tc_dict in list(tool_calls.items()):
            tool_call = ToolCall.from_dict(tc_dict)
            tool_call.status = ToolCallStatus.SUCCEEDED
            if tool_call.result is None:
                output_content = tool_output_text or (
                    f"{tool_call.tool_name} executed successfully as part of task"
                    if tool_call.tool_name
                    else "Tool executed successfully as part of task"
                )
                tool_call.result = ToolOutput(
                    content=output_content,
                    details=ExecuteDetails(stdout="Success", exit_code=0),
                )
            tool_calls[tc_id] = tool_call.to_dict()

        if add_agent_thought:
            agent_thought = AgentThought(
                content="Task execution completed successfully. All tools processed without issues."
            )
            if "thoughts" not in dev_tool:
                dev_tool["thoughts"] = []
            dev_tool["thoughts"].append(agent_thought.to_dict())

    async def _handle_task_cancellation(self, task_id: str, old_state: str, context: str = "") -> None:
        """Handle task cancellation with consistent status update and notification."""
        try:
            # Update task status to cancelled
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id].task
                task.status.state = TaskState.CANCELLED
                task.status.timestamp = current_timestamp()

            # Send notification for cancellation
            cancel_message = f"Task was cancelled{': ' + context if context else ''}"
            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                "old_state": old_state,
                "new_state": TaskState.CANCELLED.value,
                "message": cancel_message
            }))

            logger.info(f"Task cancelled: {cancel_message}", extra={"task_id": task_id})
        except Exception as e:
            logger.error(
                "Error in task cancellation handler",
                extra={"task_id": task_id, "context": context, "error": str(e)}
            )

    def _extract_message_content(self, message) -> str:
        """Extract content from a message for notification purposes."""
        if message.parts:
            first_part = message.parts[0]
            if first_part.kind == "text":
                return first_part.text
            elif first_part.kind == "data":
                return str(first_part.data)
            elif first_part.kind == "file":
                return f"File: {first_part.file.name or 'unnamed'}"
            else:
                return str(first_part)
        return ""

    def _response_has_agent_content(self, response: dict) -> bool:
        """Check whether the agent response includes any textual content."""

        def _has_text(value: Any) -> bool:
            return isinstance(value, str) and value.strip() != ""

        if not isinstance(response, dict):
            return False

        # Direct text field (some agents return text at top level)
        if _has_text(response.get("text")):
            return True

        result = response.get("result")
        if isinstance(result, str):
            return _has_text(result)
        if isinstance(result, dict):
            if _has_text(result.get("text")):
                return True

            content = result.get("content")
            if isinstance(content, str):
                return _has_text(content)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and _has_text(item.get("text")):
                        return True

        return False

    def _is_input_required_from_response(self, response: dict) -> tuple[bool, str]:
        """Protocol-compliant detection of input-required state using Zed ACP response."""
        stop_reason = response.get("stopReason")
        tool_calls = response.get("toolCalls", [])
        has_agent_content = self._response_has_agent_content(response)

        if stop_reason == "input_required":
            return True, "Agent explicitly requested additional input"

        if stop_reason == "end_turn" and not tool_calls and not has_agent_content:
            return True, "Agent completed turn without actions"

        if has_agent_content:
            return False, "Agent provided response content"

        return False, f"Turn ended with reason: {stop_reason}"

    def _extract_input_types_from_response(self, response: dict) -> list[str]:
        """Extract input types from response metadata if available."""
        # Check for _meta hints in final response for enhanced detection
        meta = response.get("_meta", {})
        if isinstance(meta, dict) and "input_types" in meta:
            return meta["input_types"]

        # Fallback to default types
        return ["text/plain"]

    async def _send_message_notification(self, task_id: str, message, message_type: str) -> None:
        """Send a notification for a task message with extracted content."""
        content = self._extract_message_content(message)

        asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_MESSAGE.value, {
            "message_role": message.role,
            "message_content": content,
            "message_type": message_type
        }))

    def _serialize_policy_decision(self, decision: Optional[AutoApprovalDecision]) -> Optional[Dict[str, Any]]:
        if not decision:
            return None
        return {
            "policyId": decision.policy_id,
            "decisionType": decision.decision_type,
            "optionId": decision.option_id,
            "reason": decision.reason,
            "skipGovernors": decision.skip_governors,
        }

    def _serialize_governor_results(self, results: List[GovernorResult]) -> List[Dict[str, Any]]:
        serialized = []
        for result in results:
            serialized.append({
                "governorId": result.governor_id,
                "status": result.status,
                "optionId": result.option_id,
                "score": result.score,
                "messages": result.messages,
                "followUpPrompt": result.follow_up_prompt,
                "metadata": result.metadata,
            })
        return serialized

    def _record_permission_decision(
        self,
        context: TaskExecutionContext,
        tool_call_id: str,
        source: str,
        option_id: str,
        governor_results: List[GovernorResult],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = PermissionDecisionRecord(
            tool_call_id=tool_call_id,
            source=source,
            option_id=option_id,
            governors_involved=[result.governor_id for result in governor_results],
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )
        context.permission_decisions.append(record)

        task_metadata = context.task.metadata or {}
        context.task.metadata = task_metadata
        decisions_meta = task_metadata.setdefault("permissionDecisions", [])
        decisions_meta.append({
            "toolCallId": tool_call_id,
            "source": source,
            "optionId": option_id,
            "governorsInvolved": record.governors_involved,
            "timestamp": record.timestamp.isoformat(),
            "metadata": record.metadata,
        })

    def _append_governor_feedback(
        self,
        context: TaskExecutionContext,
        phase: str,
        summary_lines: List[str],
        results: List[GovernorResult],
    ) -> None:
        entry = {
            "phase": phase,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": summary_lines,
            "results": self._serialize_governor_results(results),
        }
        context.governor_feedback.append(entry)

        task_metadata = context.task.metadata or {}
        context.task.metadata = task_metadata
        dev_tool = task_metadata.setdefault("development-tool", {})
        feedback_meta = dev_tool.setdefault("governorFeedback", [])
        feedback_meta.append(entry)

    async def _handle_tool_permission(
        self,
        task_id: str,
        context: TaskExecutionContext,
        request: ToolPermissionRequest
    ) -> ToolPermissionDecision:
        """Evaluate permission request via policies and governors."""
        evaluation: PermissionEvaluationResult = await self.governor_manager.evaluate_permission(
            task_id=task_id,
            session_id=request.session_id,
            tool_call=request.tool_call,
            options=request.options,
        )

        tool_call_id = str(request.tool_call.get("id") or request.tool_call.get("toolCallId") or uuid4())
        async with context.lock:
            if evaluation.selected_option_id:
                # Create ToolCall for auto-approval case
                confirmation_options = []
                for opt in request.options:
                    option_id = opt.get("optionId")
                    if option_id:  # Skip options without ID
                        confirmation_options.append(
                            ConfirmationOption(
                                id=option_id,
                                name=opt.get("name", option_id),
                                description=opt.get("description")
                            )
                        )

                # Create confirmation request
                confirmation_request = ConfirmationRequest(
                    options=confirmation_options,
                    details=GenericDetails(
                        description=f"Tool permission required for {request.tool_call.get('toolId', 'unknown_tool')}"
                    )
                )

                # Create ToolCall object for extension serialization
                tool_call = ToolCall(
                    tool_call_id=tool_call_id,
                    status=ToolCallStatus.EXECUTING,
                    tool_name=request.tool_call.get("toolId", "unknown_tool"),
                    input_parameters=request.tool_call.get("parameters", {}),
                    confirmation_request=None  # Cleared after approval
                )

                # Serialize ToolCall to task metadata for extension support
                task_metadata = context.task.metadata or {}
                context.task.metadata = task_metadata
                if "development-tool" not in task_metadata:
                    task_metadata["development-tool"] = {}
                tool_calls = task_metadata["development-tool"].setdefault("tool_calls", {})
                tool_calls[tool_call_id] = tool_call.to_dict()

                self._record_permission_decision(
                    context,
                    tool_call_id,
                    evaluation.decision_source or "policy",
                    evaluation.selected_option_id,
                    evaluation.governor_results,
                    metadata={"summary": evaluation.summary_lines},
                )
                return ToolPermissionDecision(option_id=evaluation.selected_option_id)

            loop = asyncio.get_running_loop()
            decision_future: asyncio.Future[str] = loop.create_future()

            # Build confirmation options from request options
            confirmation_options = []
            for opt in request.options:
                option_id = opt.get("optionId")
                if option_id:  # Skip options without ID
                    confirmation_options.append(
                        ConfirmationOption(
                            id=option_id,
                            name=opt.get("name", option_id),
                            description=opt.get("description")
                        )
                    )

            # Create confirmation request
            confirmation_request = ConfirmationRequest(
                options=confirmation_options,
                details=GenericDetails(
                    description=f"Tool permission required for {request.tool_call.get('toolId', 'unknown_tool')}"
                )
            )

            # Create ToolCall object for extension serialization
            tool_call = ToolCall(
                tool_call_id=tool_call_id,
                status=ToolCallStatus.PENDING,
                tool_name=request.tool_call.get("toolId", "unknown_tool"),
                input_parameters=request.tool_call.get("parameters", {}),
                confirmation_request=confirmation_request
            )

            metadata = {
                "type": "tool-permission",
                "toolCallId": tool_call_id,
                "toolId": request.tool_call.get("toolId"),
                "toolCall": request.tool_call,
                "options": list(request.options),
                "summary": evaluation.summary_lines,
                "policyDecision": self._serialize_policy_decision(evaluation.policy_decision),
                "governorResults": self._serialize_governor_results(evaluation.governor_results),
                "tool_call": tool_call.to_dict(),
            }

            pending = PendingPermission(
                tool_call_id=tool_call_id,
                tool_call=request.tool_call,
                options=list(request.options),
                decision_future=decision_future,
                summary_lines=evaluation.summary_lines,
                governor_results=evaluation.governor_results,
                policy_decision=evaluation.policy_decision,
                metadata=metadata,
            )
            context.pending_permissions[tool_call_id] = pending

            # Serialize ToolCall to task metadata for extension support
            task_metadata = context.task.metadata or {}
            context.task.metadata = task_metadata
            if "development-tool" not in task_metadata:
                task_metadata["development-tool"] = {}
            tool_calls = task_metadata["development-tool"].setdefault("tool_calls", {})
            tool_calls[tool_call_id] = tool_call.to_dict()

            task = context.task
            old_state = task.status.state.value
            task.status.state = TaskState.INPUT_REQUIRED
            task.status.timestamp = current_timestamp()

        notification_payload = {
            "old_state": old_state,
            "new_state": TaskState.INPUT_REQUIRED.value,
            "message": f"Permission required for {request.tool_call.get('toolId', 'tool')}",
            "tool_call_id": tool_call_id,
            "options": request.options,
            "summary": evaluation.summary_lines,
            "policy_decision": metadata["policyDecision"],
            "governor_results": metadata["governorResults"],
            "tool_call": metadata["tool_call"],
        }

        asyncio.create_task(self._send_task_notification(
            task_id,
            EventType.TASK_INPUT_REQUIRED.value,
            notification_payload,
        ))

        logger.info(
            "Permission input required",
            extra={
                "task_id": task_id,
                "tool_call_id": tool_call_id,
                "summary_lines": evaluation.summary_lines,
            },
        )

        return ToolPermissionDecision(future=decision_future)

    async def _run_post_run_governors(
        self,
        task_id: str,
        context: TaskExecutionContext,
        connection: ZedAgentConnection,
        session_id: str,
        agent_output: Dict[str, Any],
        translator: Any,
    ) -> tuple[Dict[str, Any], bool]:
        """Process post-run governors with follow-up handling."""
        policies_payload = {
            "permissionDecisions": [
                {
                    "toolCallId": record.tool_call_id,
                    "source": record.source,
                    "optionId": record.option_id,
                    "governorsInvolved": record.governors_involved,
                    "timestamp": record.timestamp.isoformat(),
                    "metadata": record.metadata,
                }
                for record in context.permission_decisions
            ],
            "taskMetadata": context.task.metadata or {},
        }

        current_output = agent_output
        iteration = 0
        max_iterations = 1
        gov_config = getattr(self.governor_manager, "config", None)
        if gov_config:
            output_settings = getattr(gov_config, "output_settings", None)
            if output_settings is not None:
                max_iter_value = getattr(output_settings, "max_iterations", None)
                try:
                    max_iterations = max(1, int(max_iter_value))
                except (TypeError, ValueError):
                    max_iterations = 1

        while iteration < max_iterations:
            evaluation_call = self.governor_manager.evaluate_post_run(
                task_id=task_id,
                session_id=session_id,
                agent_output=current_output,
                policies=policies_payload,
                iteration=iteration,
            )

            if asyncio.iscoroutine(evaluation_call):
                evaluation = await evaluation_call
            else:
                evaluation = evaluation_call

            summary_lines = self._normalize_sequence(getattr(evaluation, "summary_lines", []))
            governor_results = self._normalize_sequence(getattr(evaluation, "governor_results", []))
            if (not summary_lines or not governor_results):
                fallback_eval = self._get_fallback_post_run_evaluation()
                if fallback_eval and fallback_eval is not evaluation:
                    if not summary_lines:
                        summary_lines = self._normalize_sequence(
                            getattr(fallback_eval, "summary_lines", [])
                        )
                    if not governor_results:
                        governor_results = self._normalize_sequence(
                            getattr(fallback_eval, "governor_results", [])
                        )

            self._append_governor_feedback(context, "post_run", summary_lines, governor_results)

            follow_up_prompts = getattr(evaluation, "follow_up_prompts", None)
            if isinstance(follow_up_prompts, Sequence):
                follow_up_prompts_list = list(follow_up_prompts)
            else:
                follow_up_prompts_list = []

            if follow_up_prompts_list:
                for governor_id, followup_prompt in follow_up_prompts_list:
                    asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_GOVERNOR_FOLLOWUP.value, {
                        "governor_id": governor_id,
                        "prompt": followup_prompt,
                        "iteration": iteration,
                    }))

                    governor_message = Message(
                        role="user",
                        parts=[TextPart(kind="text", text=followup_prompt)],
                        messageId=create_message_id(),
                        taskId=task_id,
                        contextId=context.task.contextId,
                        metadata={"origin": "governor", "governorId": governor_id},
                    )

                    async with context.lock:
                        if context.task.history is None:
                            context.task.history = []
                        context.task.history.append(governor_message)

                    zed_parts = translator.a2a_to_zedacp_message(governor_message)
                    current_output = await connection.prompt(
                        session_id,
                        zed_parts,
                        on_chunk=None,
                        cancel_event=context.cancel_event,
                    )

                    iteration += 1
                    if iteration >= max_iterations:
                        break

                if iteration >= max_iterations:
                    break
                continue

            blocked_result = getattr(evaluation, "blocked", False)
            if not isinstance(blocked_result, bool):
                blocked_result = False

            if blocked_result:
                async with context.lock:
                    old_state = context.task.status.state.value
                    context.task.status.state = TaskState.INPUT_REQUIRED
                    context.task.status.timestamp = current_timestamp()

                asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_FEEDBACK_REQUIRED.value, {
                    "message": "Governor blocked final response",
                    "summary": summary_lines,
                }))

                asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_INPUT_REQUIRED.value, {
                    "old_state": old_state,
                    "new_state": TaskState.INPUT_REQUIRED.value,
                    "message": "Governor requested human review",
                    "input_types": ["text/plain"],
                    "detection_method": "governor_block",
                    "summary": summary_lines,
                }))

                logger.warning(
                    "Governor blocked response",
                    extra={"task_id": task_id, "summary": summary_lines},
                )
                return current_output, True

            break

        return current_output, False

    async def create_task(
        self,
        context_id: str,
        agent_name: str,
        initial_message: Optional[Message] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a new A2A task."""
        async with self.lock:
            task_id = create_task_id()
    
            # Create initial task status
            status = TaskStatus(
                state=TaskState.SUBMITTED,
                timestamp=current_timestamp()
            )
    
            # Create A2A task
            history = None
            if initial_message:
                history = [Message(**initial_message.model_dump(exclude_none=True))]
    
            task = Task(
                id=task_id,
                contextId=context_id,
                status=status,
                history=history,
                artifacts=None,
                metadata=metadata or {},
                kind="task"
            )
    
            # Create execution context
            context = TaskExecutionContext(
                task=task,
                agent_name=agent_name
            )
    
            # Parse AgentSettings from initial_message metadata if present
            if initial_message and initial_message.metadata:
                agent_settings_data = initial_message.metadata.get("agent_settings")
                if agent_settings_data:
                    try:
                        agent_settings = AgentSettings.from_dict(agent_settings_data)
                        if agent_settings.workspace_path:
                            context.working_directory = agent_settings.workspace_path
                            logger.debug(
                                "Set working directory from AgentSettings",
                                extra={"task_id": task_id, "working_directory": context.working_directory}
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to parse AgentSettings from metadata",
                            extra={"task_id": task_id, "error": str(e)}
                        )
    
            # Fallback to current directory if not set
            if not context.working_directory:
                context.working_directory = os.getcwd()
    
            self._active_tasks[task_id] = context
        
            logger.info("Created A2A task",
                       extra={"task_id": task_id, "context_id": context_id, "agent": agent_name})
        
            # Send notification for task creation
            asyncio.create_task(self._send_task_notification(task_id, "task_created", {
                "old_state": None,
                "new_state": TaskState.SUBMITTED.value,
                "context_id": context_id,
                "agent_name": agent_name,
                "creation_event": True
            }))
        
            return task

    async def execute_task(
        self,
        task_id: str,
        agent_command: List[str],
        api_key: Optional[str] = None,
        working_directory: str = ".",
        mcp_servers: Optional[List[Dict]] = None,
        stream_handler: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> Task:
        """Execute an A2A task using ZedACP agent with input-required support."""
        async with self.lock:
            if task_id not in self._active_tasks:
                raise ValueError(f"Unknown task: {task_id}")
            context = self._active_tasks[task_id]
    
        task = context.task
    
        # Use context working_directory if set, otherwise fallback to parameter
        effective_working_directory = context.working_directory or working_directory
    
        try:
            async with context.lock:
                old_state = task.status.state.value
                task.status.state = TaskState.WORKING
                task.status.timestamp = current_timestamp()
    
            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                "old_state": old_state,
                "new_state": TaskState.WORKING.value,
                "message": "Task execution started"
            }))
    
            async def permission_handler(request: ToolPermissionRequest) -> ToolPermissionDecision:
                return await self._handle_tool_permission(task_id, context, request)
    
            # Provide a built-in stub agent when no external command is configured
            if agent_command in (None, ["true"]):
                logger.info(
                    "Executing task via built-in stub agent",
                    extra={"task_id": task_id, "agent_command": agent_command},
                )
                return await self._execute_stub_agent(task_id, context, stream_handler)
    
            # Execute via ZedACP agent
            async with ZedAgentConnection(
                agent_command,
                api_key=api_key,
                permission_handler=permission_handler,
            ) as connection:
                original_permission_handler = getattr(connection, "permission_handler", None)
                if original_permission_handler and original_permission_handler is not permission_handler:
                    async def combined_permission_handler(request: ToolPermissionRequest) -> ToolPermissionDecision:
                        decision = await permission_handler(request)
                        fallback = await original_permission_handler(request)
                        return fallback or decision

                    connection.permission_handler = combined_permission_handler
                await connection.initialize()
    
                # Create ZedACP session for this task
                zed_session_id = await connection.start_session(
                    cwd=effective_working_directory,
                    mcp_servers=mcp_servers or []
                )
    
                async with context.lock:
                    context.zedacp_session_id = zed_session_id
                    context.working_directory = effective_working_directory
    
                # Convert A2A message to ZedACP format if we have history
                if task.history and len(task.history) > 0:
                    user_message = task.history[0]  # First message should be user input
                    translator = _create_translator()
                    zedacp_parts = translator.a2a_to_zedacp_message(user_message)
    
                    # Execute the task with protocol-compliant input-required detection
                    cancel_event = context.cancel_event
    
                    # Chunk handler that accumulates content and handles artifacts
                    accumulated_content = []
    
                    async def on_chunk(text: str) -> None:
                        # Accumulate all chunks for the final response
                        accumulated_content.append(text)
    
                        # Forward to streaming handler if provided
                        if stream_handler:
                            await stream_handler(text)
    
                        # Embed DevelopmentToolEvent for tool call updates in streaming
                        if task.metadata and "development-tool" in task.metadata:
                            dev_tool = task.metadata["development-tool"]
                            if "tool_calls" in dev_tool:
                                # For simplicity, emit a generic update for ongoing execution
                                event = DevelopmentToolEvent(
                                    kind=DevelopmentToolEventKind.TOOL_CALL_UPDATE,
                                    data={"status": "streaming", "chunk_length": len(text)}
                                )
                                # This would be sent via streaming metadata if supported, or batched
    
                        # Also check for artifact creation notifications
                        upper_text = text.upper()
                        if ("ARTIFACT:" in upper_text or
                            "FILE:" in upper_text or
                            "CREATED:" in upper_text):
    
                            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_ARTIFACT.value, {
                                "artifact_type": "detected",
                                "artifact_name": "processing",
                                "detection_text": text[:100]
                            }))
    
                    result = await connection.prompt(
                        zed_session_id,
                        zedacp_parts,
                        on_chunk=on_chunk,
                        cancel_event=cancel_event
                    )
                    # Handle any tool call events emitted before the final response
                    while result.get("toolCalls"):
                        for tool_call in result.get("toolCalls", []):
                            self._ensure_tool_call_metadata(context, tool_call)
                        result = await connection.prompt(
                            zed_session_id,
                            zedacp_parts,
                            on_chunk=on_chunk,
                            cancel_event=cancel_event
                        )
    
                    result, blocked = await self._run_post_run_governors(
                        task_id,
                        context,
                        connection,
                        zed_session_id,
                        result,
                        translator,
                    )
    
                    if blocked:
                        logger.info("Task awaiting feedback due to governor block", extra={"task_id": task_id})
                        return task
    
                    if accumulated_content:
                        full_text = "".join(accumulated_content)
                        result.setdefault("result", {})
                        result["result"]["text"] = full_text
                        logger.debug("Added accumulated content to result",
                                     extra={"content_length": len(full_text), "chunks": len(accumulated_content)})
    
                    is_input_required, reason = self._is_input_required_from_response(result)
                    logger.info("Protocol analysis of ZedACP response",
                                extra={
                                    "task_id": task_id,
                                    "stop_reason": result.get("stopReason"),
                                    "tool_calls_count": len(result.get("toolCalls", [])),
                                    "is_input_required": is_input_required,
                                    "reason": reason
                                })
    
                    if is_input_required:
                        old_state = task.status.state.value
                        task.status.state = TaskState.INPUT_REQUIRED
                        task.status.timestamp = current_timestamp()
    
                        input_types = self._extract_input_types_from_response(result)
    
                        asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_INPUT_REQUIRED.value, {
                            "old_state": old_state,
                            "new_state": TaskState.INPUT_REQUIRED.value,
                            "message": "Additional input required",
                            "input_types": input_types,
                            "detection_method": "protocol_compliant"
                        }))
    
                        input_notification = InputRequiredNotification(
                            taskId=task_id,
                            contextId=task.contextId,
                            message="Additional input required",
                            inputTypes=input_types,
                            timeout=300
                        )
    
                        logger.info("Task requires input (protocol-compliant detection)",
                                    extra={
                                        "task_id": task_id,
                                        "notification": input_notification.model_dump(),
                                        "input_types": input_types,
                                        "detection_reason": reason
                                    })
                        return task
    
                    response_message = translator.zedacp_to_a2a_message(result, task.contextId, task_id)
                    task.history.append(response_message)
                    asyncio.create_task(self._send_message_notification(task_id, response_message, "agent_response"))
    
                    old_state = task.status.state.value
                    task.status.state = TaskState.COMPLETED
                    task.status.timestamp = current_timestamp()
    
                    final_text = result.get("result", {}).get("text")
                    self._finalize_tool_calls_success(
                        context,
                        add_agent_thought=True,
                        tool_output_text=final_text,
                    )
                    dev_tool_meta = context.task.metadata.get("development-tool", {}) if context.task else {}
                    asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                        "old_state": old_state,
                        "new_state": TaskState.COMPLETED.value,
                        "message": "Task completed successfully",
                        "development_tool_metadata": dev_tool_meta
                    }))
    
                    if self.push_notification_manager:
                        asyncio.create_task(self.push_notification_manager.cleanup_by_task_state(
                            task_id, TaskState.COMPLETED.value
                        ))
    
                    logger.info("Task completed successfully",
                                extra={"task_id": task_id, "agent": context.agent_name})
    
                    return task
                else:
                    # No message to execute, finalize any serialized tool calls before exiting
                    task.status.state = TaskState.COMPLETED
                    task.status.timestamp = current_timestamp()
                    self._finalize_tool_calls_success(context, add_agent_thought=True)
                    return task
    
        except PromptCancelled:
            await self._handle_task_cancellation(task_id, task.status.state.value)
            return task
    
        except AgentProcessError as e:
            await self._handle_task_error(task_id, task.status.state.value, e)
            raise
    
        except Exception as e:
            await self._handle_task_error(task_id, task.status.state.value, e, "Unexpected error")
            raise

    async def _execute_stub_agent(
        self,
        task_id: str,
        context: TaskExecutionContext,
        stream_handler: Optional[Callable[[str], Awaitable[None]]],
    ) -> Task:
        """Simulate agent execution when no external command is available."""
        chunk_text = "--END-OF-RESPONSE--"
        user_message = context.task.history[0] if context.task.history else None
        if user_message and user_message.parts:
            user_summary = self._extract_message_content(user_message).strip()
            if user_summary:
                chunk_text = f"{user_summary.strip()} --END-OF-RESPONSE--"

        # Emit streaming chunk for clients
        if stream_handler:
            await stream_handler(chunk_text + " ")

        stub_response = {
            "stopReason": "end_turn",
            "result": {"text": chunk_text + " "},
            "toolCalls": [],
        }

        translator = _create_translator()
        response_message = translator.zedacp_to_a2a_message(
            stub_response,
            context.task.contextId,
            task_id,
        )

        # Tag stub output for auditing
        metadata = response_message.metadata or {}
        metadata["source"] = metadata.get("source", "zedacp")
        metadata["stubAgent"] = True
        response_message.metadata = metadata

        async with context.lock:
            if context.task.history is None:
                context.task.history = []
            context.task.history.append(response_message)

            previous_state = context.task.status.state.value
            context.task.status.state = TaskState.COMPLETED
            context.task.status.timestamp = current_timestamp()

            # Add AgentThought for stub completion
            task_metadata = context.task.metadata or {}
            context.task.metadata = task_metadata
            dev_tool = task_metadata.setdefault("development-tool", {})
            agent_thought = AgentThought(content="Stub agent completed the task simulation successfully.")
            if "thoughts" not in dev_tool:
                dev_tool["thoughts"] = []
            dev_tool["thoughts"].append(agent_thought.to_dict())

        asyncio.create_task(self._send_message_notification(task_id, response_message, "agent_response"))
        dev_tool_meta = context.task.metadata.get("development-tool", {})
        asyncio.create_task(
            self._send_task_notification(
                task_id,
                EventType.TASK_STATUS_CHANGE.value,
                {
                    "old_state": previous_state,
                    "new_state": TaskState.COMPLETED.value,
                    "message": "Task execution completed via stub agent",
                    "development_tool_metadata": dev_tool_meta
                },
            )
        )

        logger.info(
            "Stub agent generated response",
            extra={
                "task_id": task_id,
                "chunk_length": len(chunk_text),
                "history_length": len(context.task.history or []),
            },
        )

        return context.task

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        if task_id in self._active_tasks:
            return self._active_tasks[task_id].task
        return None

    async def get_task_audit_data(self, task_id: str) -> Dict[str, Any]:
        """Return permission and governor history for a task."""
        async with self.lock:
            if task_id not in self._active_tasks:
                raise ValueError(f"Unknown task: {task_id}")
            context = self._active_tasks[task_id]

        async with context.lock:
            decisions = [
                {
                    "toolCallId": record.tool_call_id,
                    "source": record.source,
                    "optionId": record.option_id,
                    "governorsInvolved": record.governors_involved,
                    "timestamp": record.timestamp.isoformat(),
                    "metadata": record.metadata,
                }
                for record in context.permission_decisions
            ]

            pending = [
                {
                    "toolCallId": perm.tool_call_id,
                    "toolId": perm.tool_call.get("toolId"),
                    "options": perm.options,
                    "summary": perm.summary_lines,
                    "created_at": perm.created_at.isoformat(),
                }
                for perm in context.pending_permissions.values()
            ]

            feedback = list(context.governor_feedback)

        return {
            "permissionDecisions": decisions,
            "pendingPermissions": pending,
            "governorFeedback": feedback,
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id not in self._active_tasks:
            return False

        context = self._active_tasks[task_id]

        async with context.lock:
            for pending_id, pending in list(context.pending_permissions.items()):
                if not pending.decision_future.done():
                    pending.decision_future.set_result("deny")
                self._record_permission_decision(
                    context,
                    pending_id,
                    "system",
                    "deny",
                    pending.governor_results,
                    metadata={"reason": "task_cancelled"},
                )
            context.pending_permissions.clear()

        if context.cancel_event:
            context.cancel_event.set()

        # Update task status
        task = context.task
        old_state = task.status.state.value
        task.status.state = TaskState.CANCELLED
        task.status.timestamp = current_timestamp()

        # Send notification for cancellation
        asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
            "old_state": old_state,
            "new_state": TaskState.CANCELLED.value,
            "message": "Task was cancelled by user"
        }))

        logger.info("Cancelled A2A task", extra={"task_id": task_id})
        return True

    async def list_tasks(self, context_id: Optional[str] = None) -> List[Task]:
        """List tasks with optional context filtering."""
        tasks = [ctx.task for ctx in self._active_tasks.values()]

        if context_id:
            tasks = [t for t in tasks if t.contextId == context_id]

        return tasks

    async def provide_input_and_continue(
        self,
        task_id: str,
        user_input: Optional[Message],
        agent_command: List[str],
        api_key: Optional[str] = None,
        working_directory: Optional[str] = None,
        mcp_servers: Optional[List[Dict]] = None,
        permission_option_id: Optional[str] = None,
    ) -> Task:
        """Provide additional input or permission decision for a task."""
        async with self.lock:
            if task_id not in self._active_tasks:
                raise ValueError(f"Unknown task: {task_id}")
            context = self._active_tasks[task_id]

        task = context.task

        if not context.pending_permissions and task.status.state != TaskState.INPUT_REQUIRED:
            raise ValueError(f"Task {task_id} is not in input-required state")

        # Handle permission decisions first, including extension ToolCallConfirmation
        effective_permission_option_id = permission_option_id
        tool_call_id_to_update = None

        if user_input and user_input.metadata:
            dev_tool_meta = user_input.metadata.get("development-tool")
            if dev_tool_meta:
                # Try to deserialize ToolCallConfirmation from metadata
                confirmation_data = dev_tool_meta.get("tool_call_confirmation")
                if confirmation_data:
                    try:
                        confirmation = ToolCallConfirmation.from_dict(confirmation_data)
                        effective_permission_option_id = confirmation.selected_option_id
                        tool_call_id_to_update = confirmation.tool_call_id

                        # Validate the tool_call_id exists in pending permissions
                        if tool_call_id_to_update not in context.pending_permissions:
                            raise ValueError(f"Tool call ID not found in pending permissions: {tool_call_id_to_update}")

                        logger.info(
                            "ToolCallConfirmation deserialized from extension metadata",
                            extra={"task_id": task_id, "tool_call_id": tool_call_id_to_update, "selected_option_id": effective_permission_option_id}
                        )
                    except ValueError:
                        raise
                    except Exception as e:
                        logger.warning(
                            "Failed to deserialize ToolCallConfirmation, falling back to legacy handling",
                            extra={"task_id": task_id, "error": str(e)}
                        )

        if context.pending_permissions:
            if effective_permission_option_id is None:
                raise ValueError("permissionOptionId is required to resolve pending permissions")

            if user_input and any(
                part.kind == "text" and isinstance(getattr(part, "text", None), str) and part.text.strip()
                for part in user_input.parts
            ):
                raise ValueError("Cannot provide message content while a permission decision is pending")

            async with context.lock:
                pending_id, pending = next(iter(context.pending_permissions.items()))
                selected_option = next((opt for opt in pending.options if opt.get("optionId") == effective_permission_option_id), None)
                if not selected_option:
                    raise ValueError(f"Invalid permission option: {effective_permission_option_id}")

                if not pending.decision_future.done():
                    pending.decision_future.set_result(effective_permission_option_id)

                self._record_permission_decision(
                    context,
                    pending_id,
                    "user",
                    effective_permission_option_id,
                    pending.governor_results,
                    metadata={"summary": pending.summary_lines},
                )

                pending_tool_call = pending.tool_call or {}
                del context.pending_permissions[pending_id]
                task.status.state = TaskState.WORKING
                task.status.timestamp = current_timestamp()

                # If we have a tool_call_id_to_update, update the ToolCall status to EXECUTING
                if tool_call_id_to_update:
                    task_metadata = task.metadata or {}
                    dev_tool_meta = task_metadata.setdefault("development-tool", {})
                    tool_calls = dev_tool_meta.setdefault("tool_calls", {})
                    
                    if tool_call_id_to_update in tool_calls:
                        tool_call = ToolCall.from_dict(tool_calls[tool_call_id_to_update])
                    else:
                        tool_call = ToolCall(
                            tool_call_id=tool_call_id_to_update,
                            status=ToolCallStatus.EXECUTING,
                            tool_name=pending_tool_call.get("toolId", "unknown_tool"),
                            input_parameters=pending_tool_call.get("parameters", {}),
                            confirmation_request=None,
                        )

                    tool_call.status = ToolCallStatus.EXECUTING
                    tool_call.confirmation_request = None
                    tool_calls[tool_call_id_to_update] = tool_call.to_dict()

                    # Emit DevelopmentToolEvent for status update
                    event = DevelopmentToolEvent(
                        kind=DevelopmentToolEventKind.TOOL_CALL_UPDATE,
                        data={"tool_call_id": tool_call_id_to_update, "status": "executing"}
                    )
                    notification_metadata = {
                        "development-tool": event.to_dict()
                    }
                else:
                    notification_metadata = {}

            notification_payload = {
                "old_state": TaskState.INPUT_REQUIRED.value,
                "new_state": TaskState.WORKING.value,
                "message": "Permission decision recorded",
                "permission_option_id": effective_permission_option_id,
                **notification_metadata
            }

            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, notification_payload))

            logger.info(
                "Permission decision applied",
                extra={"task_id": task_id, "permission_option_id": effective_permission_option_id, "tool_call_id": tool_call_id_to_update},
            )

            return task

        if user_input is None:
            raise ValueError("user_input is required when no permission decision is pending")

        async with context.lock:
            old_state = task.status.state.value
            task.status.state = TaskState.WORKING
            task.status.timestamp = current_timestamp()

            if task.history is None:
                task.history = []
            task.history.append(user_input)

        asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
            "old_state": old_state,
            "new_state": TaskState.WORKING.value,
            "message": "Task resumed after user input"
        }))

        asyncio.create_task(self._send_message_notification(task_id, user_input, "user_input"))

        try:
            # Use existing ZedACP session if available, otherwise create new connection
            if context.zedacp_session_id and working_directory == context.working_directory:
                async with ZedAgentConnection(agent_command, api_key=api_key) as connection:
                    await connection.initialize()

                    await connection.load_session(context.zedacp_session_id, context.working_directory or ".")

                    translator = _create_translator()
                    zedacp_parts = translator.a2a_to_zedacp_message(user_input)

                    cancel_event = context.cancel_event
                    continuation_content: List[str] = []

                    async def on_chunk(text: str) -> None:
                        continuation_content.append(text)
                        logger.debug(
                            "Continue task output chunk",
                            extra={"task_id": task_id, "chunk": text[:100]},
                        )

                    result = await connection.prompt(
                        context.zedacp_session_id,
                        zedacp_parts,
                        on_chunk=on_chunk,
                        cancel_event=cancel_event,
                    )

                    result, blocked = await self._run_post_run_governors(
                        task_id,
                        context,
                        connection,
                        context.zedacp_session_id,
                        result,
                        translator,
                    )

                    if blocked:
                        return task

                    if continuation_content:
                        full_text = "".join(continuation_content)
                        result.setdefault("result", {})
                        result["result"]["text"] = full_text
                        logger.debug(
                            "Added continuation content to result",
                            extra={"content_length": len(full_text), "chunks": len(continuation_content)},
                        )

                    response_message = translator.zedacp_to_a2a_message(result, task.contextId, task_id)

                    async with context.lock:
                        if task.history is None:
                            task.history = []
                        task.history.append(response_message)

                    if response_message.parts:
                        for part in response_message.parts:
                            if part.kind == "file":
                                asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_ARTIFACT.value, {
                                    "artifact_type": "file",
                                    "artifact_name": part.file.name or "unnamed",
                                    "artifact_size": len(part.file.bytes) if hasattr(part.file, 'bytes') and part.file.bytes else 0
                                }))

                    is_input_required, reason = self._is_input_required_from_response(result)
                    if is_input_required:
                        async with context.lock:
                            old_state = task.status.state.value
                            task.status.state = TaskState.INPUT_REQUIRED
                            task.status.timestamp = current_timestamp()

                        input_types = self._extract_input_types_from_response(result)

                        asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_INPUT_REQUIRED.value, {
                            "old_state": old_state,
                            "new_state": TaskState.INPUT_REQUIRED.value,
                            "message": "Additional input required after continuation",
                            "input_types": input_types,
                            "detection_method": "protocol_compliant"
                        }))

                        logger.info(
                            "Additional input required after continuation",
                            extra={"task_id": task_id, "detection_reason": reason, "input_types": input_types},
                        )
                        return task

                    async with context.lock:
                        old_state = task.status.state.value
                        task.status.state = TaskState.COMPLETED
                        task.status.timestamp = current_timestamp()

                    asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                        "old_state": old_state,
                        "new_state": TaskState.COMPLETED.value,
                        "message": "Task completed after input continuation"
                    }))

                    logger.info("Task completed after input", extra={"task_id": task_id, "agent": context.agent_name})
                    return task

            # Need to restart with new connection
            return await self.execute_task(
                task_id,
                agent_command,
                api_key,
                working_directory or ".",
                mcp_servers,
            )

        except PromptCancelled:
            await self._handle_task_cancellation(task_id, task.status.state.value, "during input continuation")
            return task

        except AgentProcessError as e:
            await self._handle_task_error(task_id, task.status.state.value, e, "during continuation")
            raise

        except Exception as e:
            await self._handle_task_error(task_id, task.status.state.value, e, "during continuation")
            raise

    async def get_task_id_for_tool_call(self, tool_call_id: str) -> Optional[str]:
        """Return the active task ID that contains the given tool call metadata."""
        async with self.lock:
            for current_task_id, context in self._active_tasks.items():
                metadata = context.task.metadata or {}
                dev_tool_meta = metadata.get("development-tool") or {}
                tool_calls = dev_tool_meta.get("tool_calls", {})
                if tool_call_id in tool_calls:
                    return current_task_id
        return None

    async def cleanup_completed_tasks(self) -> int:
        """Clean up old completed tasks."""
        to_remove = []
        for task_id, context in self._active_tasks.items():
            if context.task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
                # Keep for a while for history, then remove
                if context.created_at:
                    age = (datetime.utcnow() - context.created_at).total_seconds()
                    if age > 3600:  # Remove after 1 hour
                        to_remove.append(task_id)

        for task_id in to_remove:
            del self._active_tasks[task_id]

        return len(to_remove)

    async def get_input_required_tasks(self) -> List[Task]:
        """Get all tasks currently in input-required state."""
        input_required_tasks = []
        for context in self._active_tasks.values():
            if context.task.status.state == TaskState.INPUT_REQUIRED:
                input_required_tasks.append(context.task)
        return input_required_tasks


# Global A2A-native task manager
a2a_task_manager = A2ATaskManager()
