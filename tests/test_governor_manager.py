import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from src.a2a_acp.governor_manager import (
    GovernorManager,
    GovernorConfig,
    GovernorDefinition,
    GovernorSettings,
    AutoApprovalPolicy,
    GovernorResult,
)


@pytest.mark.asyncio
async def test_permission_auto_approval_policy():
    policy = AutoApprovalPolicy(
        id="docs",
        applies_to=["tool.write"],
        decision={"type": "approve", "optionId": "allow"},
    )
    manager = GovernorManager(
        governor_config=GovernorConfig(),
        policies=[policy],
    )

    result = await manager.evaluate_permission(
        task_id="task-1",
        session_id="sess-1",
        tool_call={"toolId": "tool.write"},
        options=[{"optionId": "allow", "kind": "allow_once"}],
    )

    assert result.selected_option_id == "allow"
    assert result.decision_source == "policy:docs"
    assert result.requires_manual is False


@pytest.mark.asyncio
async def test_permission_governor_rejects():
    governor = GovernorDefinition(id="sec", type="script")
    config = GovernorConfig(
        permission_governors=[governor],
        permission_settings=GovernorSettings(),
    )
    manager = GovernorManager(governor_config=config, policies=[])

    reject_result = GovernorResult(governor_id="sec", status="reject", option_id="deny")

    with patch.object(manager, "_run_governors", new=AsyncMock(return_value=[reject_result])):
        result = await manager.evaluate_permission(
            task_id="task-2",
            session_id="sess-2",
            tool_call={"toolId": "tool.write"},
            options=[{"optionId": "allow", "kind": "allow_once"}, {"optionId": "deny", "kind": "reject_once"}],
        )

    assert result.selected_option_id == "deny"
    assert result.decision_source == "governor:sec"
    assert result.requires_manual is False


@pytest.mark.asyncio
async def test_post_run_followup_and_blocking():
    governor = GovernorDefinition(id="review", type="script")
    config = GovernorConfig(
        output_governors=[governor],
        output_settings=GovernorSettings(max_iterations=2),
    )
    manager = GovernorManager(governor_config=config, policies=[])

    evaluations = [
        [GovernorResult(governor_id="review", status="needs_attention", follow_up_prompt="Add tests")],
        [GovernorResult(governor_id="review", status="reject", messages=["Unsafe output"])],
    ]

    async def fake_run_governors(*args, **kwargs):
        return evaluations.pop(0)

    with patch.object(manager, "_run_governors", new=AsyncMock(side_effect=fake_run_governors)):
        result_followup = await manager.evaluate_post_run(
            task_id="task-3",
            session_id="sess-3",
            agent_output={"result": {"text": "draft"}},
            policies={},
        )
        result_blocked = await manager.evaluate_post_run(
            task_id="task-3",
            session_id="sess-3",
            agent_output={"result": {"text": "draft"}},
            policies={},
        )

    assert result_followup.follow_up_prompts == [("review", "Add tests")]
    assert result_blocked.blocked is True
