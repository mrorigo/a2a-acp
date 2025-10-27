"""
A2A Task Manager

A2A-native task lifecycle management that bridges A2A tasks to ZedACP runs.
Replaces RunManager with proper A2A terminology.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from a2a.models import (
    Task,
    TaskStatus,
    TaskState,
    Message,
    TextPart,
    generate_id,
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
from .models import EventType
from .governor_manager import (
    AutoApprovalDecision,
    GovernorResult,
    PermissionEvaluationResult,
    get_governor_manager,
)

logger = logging.getLogger(__name__)


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
    cancel_event: Optional[asyncio.Event] = None
    created_at: Optional[datetime] = None
    pending_permissions: Dict[str, PendingPermission] = field(default_factory=dict)
    permission_decisions: List[PermissionDecisionRecord] = field(default_factory=list)
    governor_feedback: List[Dict[str, Any]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.cancel_event is None:
            self.cancel_event = asyncio.Event()


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

    @property
    def lock(self) -> asyncio.Lock:
        """Get or create the asyncio lock for thread-safe operations."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _send_task_notification(self, task_id: str, event: str, event_data: Dict[str, Any]) -> None:
        """Send a push notification for a task event."""
        if self.push_notification_manager:
            try:
                await self.push_notification_manager.send_notification(task_id, {
                    "event": event,
                    "task_id": task_id,
                    **event_data
                })
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
        context: str = ""
    ) -> None:
        """Handle task errors with consistent status update, notification, and cleanup."""
        try:
            # Update task status to failed
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id].task
                task.status.state = TaskState.FAILED
                task.status.timestamp = current_timestamp()

            # Send notification for failure
            error_message = f"{context}: {str(error)}" if context else str(error)
            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                "old_state": old_state,
                "new_state": TaskState.FAILED.value,
                "message": error_message,
                "error": str(error)
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
        feedback_meta = task_metadata.setdefault("governorFeedback", [])
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

            metadata = {
                "type": "tool-permission",
                "toolCallId": tool_call_id,
                "toolId": request.tool_call.get("toolId"),
                "toolCall": request.tool_call,
                "options": list(request.options),
                "summary": evaluation.summary_lines,
                "policyDecision": self._serialize_policy_decision(evaluation.policy_decision),
                "governorResults": self._serialize_governor_results(evaluation.governor_results),
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
        max_iterations = max(1, self.governor_manager.config.output_settings.max_iterations)

        while iteration < max_iterations:
            evaluation = await self.governor_manager.evaluate_post_run(
                task_id=task_id,
                session_id=session_id,
                agent_output=current_output,
                policies=policies_payload,
                iteration=iteration,
            )

            self._append_governor_feedback(context, "post_run", evaluation.summary_lines, evaluation.governor_results)

            if evaluation.follow_up_prompts:
                for governor_id, followup_prompt in evaluation.follow_up_prompts:
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

            if evaluation.blocked:
                async with context.lock:
                    old_state = context.task.status.state.value
                    context.task.status.state = TaskState.INPUT_REQUIRED
                    context.task.status.timestamp = current_timestamp()

                asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_FEEDBACK_REQUIRED.value, {
                    "message": "Governor blocked final response",
                    "summary": evaluation.summary_lines,
                }))

                asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_INPUT_REQUIRED.value, {
                    "old_state": old_state,
                    "new_state": TaskState.INPUT_REQUIRED.value,
                    "message": "Governor requested human review",
                    "input_types": ["text/plain"],
                    "detection_method": "governor_block",
                    "summary": evaluation.summary_lines,
                }))

                logger.warning(
                    "Governor blocked response",
                    extra={"task_id": task_id, "summary": evaluation.summary_lines},
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
            if not agent_command or agent_command == ["true"]:
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
                await connection.initialize()

                # Create ZedACP session for this task
                zed_session_id = await connection.start_session(
                    cwd=working_directory,
                    mcp_servers=mcp_servers or []
                )

                async with context.lock:
                    context.zedacp_session_id = zed_session_id
                    context.working_directory = working_directory

                # Convert A2A message to ZedACP format if we have history
                if task.history and len(task.history) > 0:
                    user_message = task.history[0]  # First message should be user input
                    from ..a2a.translator import A2ATranslator
                    translator = A2ATranslator()
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

                    asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                        "old_state": old_state,
                        "new_state": TaskState.COMPLETED.value,
                        "message": "Task completed successfully"
                    }))

                    if self.push_notification_manager:
                        asyncio.create_task(self.push_notification_manager.cleanup_by_task_state(
                            task_id, TaskState.COMPLETED.value
                        ))

                    logger.info("Task completed successfully",
                                extra={"task_id": task_id, "agent": context.agent_name})

                    return task
                else:
                    # No message to execute
                    task.status.state = TaskState.COMPLETED
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

        from a2a.translator import A2ATranslator  # Local import to avoid circular dependency

        stub_response = {
            "stopReason": "end_turn",
            "result": {"text": chunk_text + " "},
            "toolCalls": [],
        }

        translator = A2ATranslator()
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

        asyncio.create_task(self._send_message_notification(task_id, response_message, "agent_response"))
        asyncio.create_task(
            self._send_task_notification(
                task_id,
                EventType.TASK_STATUS_CHANGE.value,
                {
                    "old_state": previous_state,
                    "new_state": TaskState.COMPLETED.value,
                    "message": "Task execution completed via stub agent",
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

        if task.status.state != TaskState.INPUT_REQUIRED:
            raise ValueError(f"Task {task_id} is not in input-required state")

        # Handle permission decisions first
        if context.pending_permissions:
            if permission_option_id is None:
                raise ValueError("permissionOptionId is required to resolve pending permissions")

            if user_input and any(
                part.kind == "text" and isinstance(getattr(part, "text", None), str) and part.text.strip()
                for part in user_input.parts
            ):
                raise ValueError("Cannot provide message content while a permission decision is pending")

            async with context.lock:
                pending_id, pending = next(iter(context.pending_permissions.items()))
                selected_option = next((opt for opt in pending.options if opt.get("optionId") == permission_option_id), None)
                if not selected_option:
                    raise ValueError(f"Invalid permission option: {permission_option_id}")

                if not pending.decision_future.done():
                    pending.decision_future.set_result(permission_option_id)

                self._record_permission_decision(
                    context,
                    pending_id,
                    "user",
                    permission_option_id,
                    pending.governor_results,
                    metadata={"summary": pending.summary_lines},
                )

                del context.pending_permissions[pending_id]
                task.status.state = TaskState.WORKING
                task.status.timestamp = current_timestamp()

            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                "old_state": TaskState.INPUT_REQUIRED.value,
                "new_state": TaskState.WORKING.value,
                "message": "Permission decision recorded",
                "permission_option_id": permission_option_id,
            }))

            logger.info(
                "Permission decision applied",
                extra={"task_id": task_id, "permission_option_id": permission_option_id},
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

                    from ..a2a.translator import A2ATranslator
                    translator = A2ATranslator()
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
