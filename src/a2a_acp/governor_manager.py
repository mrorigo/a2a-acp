from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from fnmatch import fnmatch
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import httpx
import yaml

logger = logging.getLogger(__name__)

GovernorPhase = Literal["permission", "post_run"]
GovernorStatus = Literal["approve", "reject", "needs_attention", "error"]
GovernorType = Literal["script", "http", "python"]
AutoDecisionType = Literal["approve", "reject", "manual"]


def _default_governor_path() -> Path:
    """Return the default governors configuration path."""
    env_path = os.getenv("A2A_GOVERNORS_FILE")
    if env_path:
        return Path(env_path)
    return Path("governors.yaml")


def _default_policy_path() -> Optional[Path]:
    env_path = os.getenv("A2A_AUTO_APPROVAL_FILE")
    if env_path:
        return Path(env_path)
    default_file = Path("auto_approval_policies.yaml")
    return default_file if default_file.exists() else None


@dataclass
class AutoApprovalDecision:
    """Represents the outcome of an auto-approval policy."""

    policy_id: str
    decision_type: AutoDecisionType
    option_id: Optional[str] = None
    reason: Optional[str] = None
    skip_governors: bool = False


@dataclass
class AutoApprovalPolicy:
    """Describes an auto-approval policy rule."""

    id: str
    applies_to: List[str] = field(default_factory=list)
    include_paths: List[str] = field(default_factory=list)
    exclude_paths: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    decision: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, tool_call: Dict[str, Any]) -> Optional[AutoApprovalDecision]:
        """Return a decision if the policy applies to the given tool call."""
        tool_id = tool_call.get("toolId")
        
        # Infer tool ID for diff-based edits if not explicitly provided
        if not tool_id:
            kind = tool_call.get("kind", "")
            content = tool_call.get("content", [])
            if kind == "edit" and any(item.get("type") == "diff" for item in content if isinstance(item, dict)):
                tool_id = "functions.acp_fs__edit_text_file"  # Virtual tool ID for diff edits
                logger.debug("Inferred tool ID for diff edit: %s", tool_id)
            else:
                return None
        
        if self.applies_to and tool_id not in self.applies_to:
            return None

        touched_paths = list(_extract_paths_from_tool_call(tool_call))
        if self.include_paths:
            if not touched_paths:
                return None
            if not any(any(fnmatch(path, pattern) for pattern in self.include_paths) for path in touched_paths):
                return None

        if self.exclude_paths:
            if any(any(fnmatch(path, pattern) for pattern in self.exclude_paths) for path in touched_paths):
                return None

        if self.parameters:
            parameters = tool_call.get("parameters") or {}
            if not _policy_parameter_match(self.parameters, parameters):
                return None

        decision_type = self.decision.get("type", "manual")
        option_id = self.decision.get("optionId")
        reason = self.decision.get("reason")
        skip_governors = bool(self.decision.get("skipGovernors", False))

        if decision_type not in ("approve", "reject", "manual"):
            logger.warning(
                "Policy %s returned unsupported decision type %s",
                self.id,
                decision_type,
            )
            return None

        return AutoApprovalDecision(
            policy_id=self.id,
            decision_type=decision_type,  # type: ignore[arg-type]
            option_id=option_id,
            reason=reason,
            skip_governors=skip_governors,
        )


@dataclass
class GovernorDefinition:
    """Configuration for an individual governor."""

    id: str
    type: GovernorType
    command: Optional[Sequence[str]] = None
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_ms: int = 5000
    requires_manual: bool = False


@dataclass
class GovernorSettings:
    """Global settings for a governor phase."""

    stop_on_first_reject: bool = False
    auto_decision: Optional[str] = None
    max_iterations: int = 3


@dataclass
class GovernorConfig:
    """Top-level configuration for all governors."""

    permission_governors: List[GovernorDefinition] = field(default_factory=list)
    output_governors: List[GovernorDefinition] = field(default_factory=list)
    permission_settings: GovernorSettings = field(default_factory=GovernorSettings)
    output_settings: GovernorSettings = field(default_factory=GovernorSettings)


@dataclass
class GovernorResult:
    """Result payload returned by a governor invocation."""

    governor_id: str
    status: GovernorStatus
    option_id: Optional[str] = None
    score: Optional[float] = None
    messages: List[str] = field(default_factory=list)
    follow_up_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: Optional[int] = None


@dataclass
class PermissionEvaluationResult:
    """Aggregated outcome for a permission request."""

    policy_decision: Optional[AutoApprovalDecision]
    governor_results: List[GovernorResult]
    selected_option_id: Optional[str]
    decision_source: Optional[str]
    requires_manual: bool
    summary_lines: List[str]


@dataclass
class PostRunEvaluationResult:
    """Aggregated outcome for post-run governors."""

    governor_results: List[GovernorResult]
    blocked: bool
    follow_up_prompts: List[Tuple[str, str]]  # (governor_id, prompt)
    summary_lines: List[str]


def _read_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")
            return data
    except Exception as exc:
        logger.error("Failed to read YAML file %s: %s", path, exc)
        raise


def _extract_paths_from_tool_call(tool_call: Dict[str, Any]) -> Iterable[str]:
    content = tool_call.get("content") or []
    for item in content:
        if not isinstance(item, dict):
            continue
        # Handle diff content specifically
        if item.get("type") == "diff":
            path = item.get("path")
            if isinstance(path, str):
                yield path
        else:
            path = item.get("path") or item.get("uri")
            if isinstance(path, str):
                yield path
            elif isinstance(path, list):
                for candidate in path:
                    if isinstance(candidate, str):
                        yield candidate
    parameters = tool_call.get("parameters")
    if isinstance(parameters, dict):
        file_path = parameters.get("path") or parameters.get("file")
        if isinstance(file_path, str):
            yield file_path


def _policy_parameter_match(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key == "command_prefix":
            command = actual.get("command")
            if not isinstance(command, list):
                return False
            if not isinstance(value, list):
                return False
            if command[: len(value)] != value:
                return False
            continue

        if key not in actual or actual[key] != value:
            return False
    return True


def _parse_governor_definition(raw: Dict[str, Any]) -> GovernorDefinition:
    governor_type = raw.get("type", "script")
    if governor_type not in ("script", "http", "python"):
        raise ValueError(f"Unsupported governor type: {governor_type}")

    command = raw.get("command")
    if command is not None and not isinstance(command, (list, tuple)):
        raise ValueError("Governor command must be a list")

    headers = raw.get("headers") or {}
    if headers and not isinstance(headers, dict):
        raise ValueError("Governor headers must be a mapping")

    definition = GovernorDefinition(
        id=str(raw.get("id") or ""),
        type=governor_type,  # type: ignore[arg-type]
        command=command,
        url=raw.get("url"),
        headers=headers,
        timeout_ms=int(raw.get("timeout_ms") or 5000),
        requires_manual=bool(raw.get("requires_manual", False)),
    )
    if not definition.id:
        raise ValueError("Governor definition missing id")
    return definition


def _parse_governor_settings(raw: Dict[str, Any]) -> GovernorSettings:
    if raw is None:
        return GovernorSettings()
    return GovernorSettings(
        stop_on_first_reject=bool(raw.get("stop_on_first_reject", False)),
        auto_decision=raw.get("auto_decision"),
        max_iterations=int(raw.get("max_iterations", 3)),
    )


def load_governor_config(path: Optional[Path] = None) -> GovernorConfig:
    config_path = path or _default_governor_path()
    if not config_path.exists():
        return GovernorConfig()

    data = _read_yaml_file(config_path)

    permission_settings = _parse_governor_settings(data.get("permission_settings") or data.get("settings", {}).get("permission"))
    output_settings = _parse_governor_settings(data.get("output_settings") or data.get("settings", {}).get("output"))

    permission_governors = []
    for entry in data.get("permission_governors", []):
        try:
            permission_governors.append(_parse_governor_definition(entry))
        except Exception as exc:
            logger.error("Failed to parse permission governor entry %s: %s", entry, exc)

    output_governors = []
    for entry in data.get("output_governors", []):
        try:
            output_governors.append(_parse_governor_definition(entry))
        except Exception as exc:
            logger.error("Failed to parse output governor entry %s: %s", entry, exc)

    return GovernorConfig(
        permission_governors=permission_governors,
        output_governors=output_governors,
        permission_settings=permission_settings,
        output_settings=output_settings,
    )


def load_auto_approval_policies(path: Optional[Path] = None) -> List[AutoApprovalPolicy]:
    policies_path = path or _default_policy_path()
    if not policies_path or not policies_path.exists():
        return []

    data = _read_yaml_file(policies_path)
    policies = []
    for entry in data.get("auto_approval_policies", []):
        policy = AutoApprovalPolicy(
            id=str(entry.get("id") or ""),
            applies_to=list(entry.get("applies_to") or []),
            include_paths=list(entry.get("include_paths") or []),
            exclude_paths=list(entry.get("exclude_paths") or []),
            parameters=dict(entry.get("parameters") or {}),
            decision=dict(entry.get("decision") or {}),
        )
        if not policy.id:
            logger.warning("Skipping auto approval policy with missing id: %s", entry)
            continue
        policies.append(policy)
    return policies


class GovernorExecutionError(RuntimeError):
    """Raised when a governor invocation fails."""


class GovernorManager:
    """Central coordinator for auto-approval policies and governors."""

    def __init__(
        self,
        governor_config: Optional[GovernorConfig] = None,
        policies: Optional[List[AutoApprovalPolicy]] = None,
    ) -> None:
        self._config = governor_config or load_governor_config()
        self._policies = policies or load_auto_approval_policies()
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def config(self) -> GovernorConfig:
        return self._config

    @property
    def policies(self) -> List[AutoApprovalPolicy]:
        return self._policies

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def evaluate_permission(
        self,
        *,
        task_id: str,
        session_id: str,
        tool_call: Dict[str, Any],
        options: Sequence[Dict[str, Any]],
        policies_only: bool = False,
    ) -> PermissionEvaluationResult:
        """Evaluate auto-approval policies and permission governors."""
        payload = {
            "phase": "permission",
            "taskId": task_id,
            "sessionId": session_id,
            "toolCall": tool_call,
        }

        policy_decision = self._evaluate_policies(tool_call)

        run_governors = not policies_only
        governor_results: List[GovernorResult] = []
        selected_option_id = None
        decision_source = None
        requires_manual = True
        summary_lines: List[str] = []

        if policy_decision:
            summary_lines.append(f"[policy:{policy_decision.policy_id}] {policy_decision.decision_type}")
            if policy_decision.decision_type in ("approve", "reject"):
                selected_option_id = _resolve_option_id(policy_decision.option_id, options, fallback="approved" if policy_decision.decision_type == "approve" else "abort")
                decision_source = f"policy:{policy_decision.policy_id}"
                requires_manual = False
                run_governors = not policy_decision.skip_governors
            else:
                requires_manual = True

        if run_governors and self.config.permission_governors:
            governor_results = await self._run_governors(
                phase="permission",
                payload=payload,
                governors=self._permission_governors_for_policy(policy_decision),
                settings=self.config.permission_settings,
            )

            summary_lines.extend(_governor_summary_lines(governor_results))

            decision = _aggregate_permission_results(
                governor_results,
                options,
                self.config.permission_settings,
            )

            if decision.selected_option_id and not selected_option_id:
                selected_option_id = decision.selected_option_id
                decision_source = decision.decision_source
                requires_manual = decision.requires_manual
            elif decision.requires_manual is False:
                requires_manual = False

            if not summary_lines and decision.summary_lines:
                summary_lines.extend(decision.summary_lines)
        elif not self.config.permission_governors:
            logger.debug("No permission governors configured")

        return PermissionEvaluationResult(
            policy_decision=policy_decision,
            governor_results=governor_results,
            selected_option_id=selected_option_id,
            decision_source=decision_source,
            requires_manual=requires_manual,
            summary_lines=summary_lines,
        )

    async def evaluate_post_run(
        self,
        *,
        task_id: str,
        session_id: str,
        agent_output: Dict[str, Any],
        policies: Dict[str, Any],
        iteration: int = 0,
    ) -> PostRunEvaluationResult:
        """Execute post-run governors and aggregate their responses."""
        payload = {
            "phase": "post_run",
            "taskId": task_id,
            "sessionId": session_id,
            "agentOutput": agent_output,
            "policies": policies,
        }

        governor_results = await self._run_governors(
            phase="post_run",
            payload=payload,
            governors=self.config.output_governors,
            settings=self.config.output_settings,
        )

        blocked = any(result.status == "reject" for result in governor_results)
        follow_ups = [
            (result.governor_id, result.follow_up_prompt)
            for result in governor_results
            if result.follow_up_prompt
        ]
        summary_lines = _governor_summary_lines(governor_results)

        return PostRunEvaluationResult(
            governor_results=governor_results,
            blocked=blocked,
            follow_up_prompts=follow_ups,
            summary_lines=summary_lines,
        )

    def _evaluate_policies(self, tool_call: Dict[str, Any]) -> Optional[AutoApprovalDecision]:
        for policy in self._policies:
            decision = policy.evaluate(tool_call)
            if decision:
                logger.info(
                    "Auto-approval policy %s produced decision %s",
                    policy.id,
                    decision.decision_type,
                    extra={"policy_id": policy.id, "decision": decision.decision_type},
                )
                return decision
        return None

    def _permission_governors_for_policy(
        self,
        policy_decision: Optional[AutoApprovalDecision],
    ) -> List[GovernorDefinition]:
        if not policy_decision or policy_decision.decision_type != "approve":
            return self.config.permission_governors
        filtered = [
            governor
            for governor in self.config.permission_governors
            if not governor.requires_manual
        ]
        if filtered:
            return filtered
        return self.config.permission_governors

    async def _run_governors(
        self,
        *,
        phase: GovernorPhase,
        payload: Dict[str, Any],
        governors: Sequence[GovernorDefinition],
        settings: GovernorSettings,
    ) -> List[GovernorResult]:
        results: List[GovernorResult] = []
        for governor in governors:
            try:
                result = await self._invoke_governor(governor, payload)
            except GovernorExecutionError as exc:
                logger.error("Governor %s failed: %s", governor.id, exc)
                result = GovernorResult(
                    governor_id=governor.id,
                    status="error",
                    messages=[str(exc)],
                    error=str(exc),
                )

            results.append(result)

            if result.status == "reject" and settings.stop_on_first_reject:
                break
        return results

    async def _invoke_governor(
        self,
        governor: GovernorDefinition,
        payload: Dict[str, Any],
    ) -> GovernorResult:
        started = monotonic()
        hashed_input = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

        if governor.type in ("script", "python"):
            result = await self._run_script_governor(governor, payload)
        elif governor.type == "http":
            result = await self._run_http_governor(governor, payload)
        else:
            raise GovernorExecutionError(f"Unsupported governor type: {governor.type}")

        result.metadata.setdefault("input_hash", hashed_input)
        result.latency_ms = int((monotonic() - started) * 1000)
        logger.info(
            "Governor %s completed",
            governor.id,
            extra={
                "governor_id": governor.id,
                "status": result.status,
                "latency_ms": result.latency_ms,
                "input_hash": hashed_input,
            },
        )
        return result

    async def _run_script_governor(
        self,
        governor: GovernorDefinition,
        payload: Dict[str, Any],
    ) -> GovernorResult:
        if not governor.command:
            raise GovernorExecutionError(f"Governor {governor.id} missing command")

        process = await asyncio.create_subprocess_exec(
            *governor.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert process.stdin
        input_data = json.dumps(payload).encode("utf-8")
        communicate = process.communicate(input=input_data)
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(communicate, timeout=governor.timeout_ms / 1000)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise GovernorExecutionError(f"{governor.id} timed out after {governor.timeout_ms} ms") from exc
        stderr_text = (stderr_bytes or b"").decode("utf-8", errors="ignore").strip()

        if stderr_text:
            logger.debug(
                "Governor %s stderr: %s",
                governor.id,
                stderr_text[:500],
            )

        if process.returncode != 0:
            raise GovernorExecutionError(f"{governor.id} exited with {process.returncode}: {stderr_text or 'no stderr'}")

        response_text = (stdout_bytes or b"").decode("utf-8").strip()
        if not response_text:
            raise GovernorExecutionError(f"{governor.id} did not return a response")

        try:
            response = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise GovernorExecutionError(f"{governor.id} returned invalid JSON: {exc}") from exc

        return _governor_result_from_response(governor.id, response)

    async def _run_http_governor(
        self,
        governor: GovernorDefinition,
        payload: Dict[str, Any],
    ) -> GovernorResult:
        if not governor.url:
            raise GovernorExecutionError(f"Governor {governor.id} missing URL")

        if self._http_client is None:
            self._http_client = httpx.AsyncClient()

        try:
            response = await self._http_client.post(
                governor.url,
                headers=governor.headers,
                json=payload,
                timeout=governor.timeout_ms / 1000,
            )
        except Exception as exc:
            raise GovernorExecutionError(f"{governor.id} HTTP error: {exc}") from exc

        if response.status_code >= 400:
            raise GovernorExecutionError(f"{governor.id} HTTP {response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
        except Exception as exc:
            raise GovernorExecutionError(f"{governor.id} returned invalid JSON: {exc}") from exc

        return _governor_result_from_response(governor.id, data)


def _governor_result_from_response(governor_id: str, response: Dict[str, Any]) -> GovernorResult:
    status = response.get("status", "needs_attention")
    option_id = response.get("optionId")
    score = response.get("score")
    messages = response.get("messages") or []
    follow_up = response.get("followUpPrompt")
    metadata = response.get("metadata") or {}

    if status not in ("approve", "reject", "needs_attention", "error"):
        status = "needs_attention"

    if not isinstance(messages, list):
        messages = [str(messages)]

    return GovernorResult(
        governor_id=governor_id,
        status=status,  # type: ignore[arg-type]
        option_id=option_id,
        score=score if isinstance(score, (int, float)) else None,
        messages=[str(m) for m in messages],
        follow_up_prompt=follow_up if isinstance(follow_up, str) else None,
        metadata=metadata if isinstance(metadata, dict) else {},
        raw_response=response,
    )


def _resolve_option_id(
    preferred_option_id: Optional[str],
    options: Sequence[Dict[str, Any]],
    fallback: str,
) -> Optional[str]:
    option_ids = {opt.get("optionId") for opt in options}
    if preferred_option_id and preferred_option_id in option_ids:
        return preferred_option_id
    if fallback in option_ids:
        return fallback
    return next(iter(option_ids), None)


@dataclass
class _GovernorDecisionAggregate:
    selected_option_id: Optional[str]
    decision_source: Optional[str]
    requires_manual: bool
    summary_lines: List[str]


def _aggregate_permission_results(
    results: Sequence[GovernorResult],
    options: Sequence[Dict[str, Any]],
    settings: GovernorSettings,
) -> _GovernorDecisionAggregate:
    if not results:
        return _GovernorDecisionAggregate(
            selected_option_id=None,
            decision_source=None,
            requires_manual=True,
            summary_lines=[],
        )

    option_ids = {opt.get("optionId") for opt in options}
    reject_result = next((r for r in results if r.status == "reject"), None)
    if reject_result:
        option_id = _resolve_option_id(reject_result.option_id, options, fallback="abort")
        return _GovernorDecisionAggregate(
            selected_option_id=option_id,
            decision_source=f"governor:{reject_result.governor_id}",
            requires_manual=False,
            summary_lines=[
                f"[{reject_result.governor_id}] reject",
            ],
        )

    approvals = [r for r in results if r.status == "approve"]
    if approvals and settings.auto_decision == "all_approve":
        chosen = approvals[0]
        option_id = _resolve_option_id(chosen.option_id, options, fallback="approved")
        if option_id in option_ids:
            return _GovernorDecisionAggregate(
                selected_option_id=option_id,
                decision_source=f"governor:{chosen.governor_id}",
                requires_manual=False,
                summary_lines=[
                    f"[{chosen.governor_id}] approve (auto)",
                ],
            )

    return _GovernorDecisionAggregate(
        selected_option_id=None,
        decision_source=None,
        requires_manual=True,
        summary_lines=[],
    )


def _governor_summary_lines(results: Sequence[GovernorResult]) -> List[str]:
    lines = []
    for result in results:
        if result.status == "approve":
            status_text = "Approve"
        elif result.status == "reject":
            status_text = "Reject"
        elif result.status == "needs_attention":
            status_text = "Needs attention"
        else:
            status_text = "Error"

        message = result.messages[0] if result.messages else ""
        if message:
            lines.append(f"[{result.governor_id}] {status_text}: {message}")
        else:
            lines.append(f"[{result.governor_id}] {status_text}")
    return lines


_manager_instance: Optional[GovernorManager] = None


def get_governor_manager() -> GovernorManager:
    """Return a shared GovernorManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = GovernorManager()
    return _manager_instance


async def shutdown_governor_manager() -> None:
    """Dispose of the shared GovernorManager."""
    global _manager_instance
    if _manager_instance is not None:
        await _manager_instance.close()
    _manager_instance = None


def reset_governor_manager() -> None:
    """Reset the shared GovernorManager for testing."""
    global _manager_instance
    _manager_instance = None
