# A2A-ACP Development Tool Extension Guide

Comprehensive guide to the `development-tool` A2A extension implementation in A2A-ACP. This extension (URI: `https://developers.google.com/gemini/a2a/extensions/development-tool/v1`) enables structured interactions for development agents, including tool call lifecycles, user confirmations, agent thoughts, and slash commands. It builds on A2A's core task/streaming model, leveraging A2A-ACP's bash tool execution, governance, and push notifications.

## Overview

The development-tool extension standardizes communication for interactive development workflows, allowing clients (e.g., Gemini CLI, VS Code extensions) to:
- Discover and execute slash commands derived from configured tools.
- Receive real-time tool progress via lifecycle updates.
- Handle user confirmations for sensitive operations.
- View agent thoughts for reasoning transparency.
- Integrate with governance (e.g., RAIL) for policy decisions.

A2A-ACP implements this as a compliant A2A server, mapping bash tools to extension schemas. Key benefits:
- **Seamless UX**: Clients render tool progress, diffs, and confirmations natively.
- **Security**: Reuse governors and sandbox for confirmation flows.
- **Extensibility**: Pluggable metadata handlers for future extensions.
- **Backward Compatibility**: Non-extension clients ignore metadata; legacy flows unchanged.

**Version**: A2A-ACP v1.0.0 supports development-tool v1.0.0 (A2A v0.3.0+ required).

## RFC Reference

The extension follows the official RFC: [Gemini CLI A2A Development-Tool Extension](https://developers.google.com/gemini/a2a/extensions/development-tool/v1) (lines 1-510).

Key RFC Components Implemented:
- **Schemas**: `ToolCall`, `ConfirmationRequest`, `ToolOutput`, `ErrorDetails`, `AgentThought`, `DevelopmentToolEvent`, `SlashCommand`, etc. (in [src/a2a_acp/models.py](src/a2a_acp/models.py)).
- **Methods**: `/a2a/commands/get` (discovery), `/a2a/command/execute` (execution) ([src/a2a_acp/main.py](src/a2a_acp/main.py), lines 1537-1616).
- **Events**: `DevelopmentToolEvent` in `TaskStatusUpdateEvent` metadata for lifecycle (kinds: TOOL_CALL_UPDATE, THOUGHT, etc.).
- **Initialization**: `AgentSettings` parsed from initial message metadata (e.g., workspace path).
- **Compliance**: Agent card declares extension URI; supports push notifications for async updates.

For full spec, see the RFC. A2A-ACP passes compliance tests ([tests/test_a2a_extension_compliance.py](tests/test_a2a_extension_compliance.py)).

## Implementation Details

### Architecture

The extension integrates with A2A-ACP's core components:

1. **Schema Layer** ([src/a2a_acp/models.py](src/a2a_acp/models.py), lines 255-664):
   - Dataclasses for RFC objects with `to_dict`/`from_dict` for A2A metadata serialization.
   - Unions for polymorphic fields (e.g., `result: Union[ToolOutput, ErrorDetails]`).
   - Datetime handling as ISO strings; JSON-compatible dicts.

2. **Task Manager Integration** ([src/a2a_acp/task_manager.py](src/a2a_acp/task_manager.py)):
   - `_handle_tool_permission` (lines 351-436): Serializes `PendingPermission` as `ToolCall` (PENDING with `ConfirmationRequest`).
   - `provide_input_and_continue` (lines 986-1031): Deserializes `ToolCallConfirmation` from input, updates to EXECUTING/SUCCEEDED.
   - `on_chunk` (lines 670-689): Embeds `DevelopmentToolEvent` (e.g., TOOL_CALL_UPDATE) in streams.
   - Post-run: Emits `AgentThought` for feedback (lines 478-484).

3. **Tool Execution Enhancements** ([src/a2a_acp/bash_executor.py](src/a2a_acp/bash_executor.py), lines 173-312):
   - `execute_tool`: Streams `live_content` during execution; maps outputs to `ToolOutput` (e.g., `ExecuteDetails` for stdout/stderr).
   - Errors as `ErrorDetails`; supports `FileDiff` for file tools.
   - Lifecycle: PENDING → EXECUTING → SUCCEEDED/FAILED via events.

4. **Endpoints** ([src/a2a_acp/main.py](src/a2a_acp/main.py)):
   - `/a2a/commands/get`: Returns `GetAllSlashCommandsResponse` from [tool_config.py](src/a2a_acp/tool_config.py) tools.
   - `/a2a/command/execute`: Creates task with `ExecuteSlashCommandRequest`; returns `ExecuteSlashCommandResponse` (execution_id = task_id).

5. **Agent Card & Settings**:
   - `/a2a/agent`: Includes extension in `capabilities.extensions` if enabled.
   - `create_task` (task_manager.py, line 545): Parses `AgentSettings` from metadata (e.g., workspace).

6. **Push Notifications** ([src/a2a_acp/push_notification_manager.py](src/a2a_acp/push_notification_manager.py)):
   - Payloads include `DevelopmentToolEvent` for async tool updates.
   - Enhanced with extension kinds (e.g., "tool_call_update").

7. **Configuration** ([src/a2a_acp/settings.py](src/a2a_acp/settings.py)):
   - `DEVELOPMENT_TOOL_EXTENSION_ENABLED: bool = True` (feature flag).

8. **Persistence** ([src/a2a_acp/database.py](src/a2a_acp/database.py)):
   - Metadata JSON storage for `ToolCall` history (lines 152-176).

### Slash Commands Mapping

Bash tools in `tools.yaml` auto-map to slash commands:
- `name`: Slash command (e.g., "web_request" → `/web_request`).
- `parameters`: `SlashCommandArgument` for discovery.
- Execution: Creates task with command as initial message; metadata preserves arguments.

Example: `tools.yaml` tool becomes discoverable via `/a2a/commands/get`.

### Confirmation & Governance Integration

- **Confirmations**: Governor results map to `ConfirmationRequest.options`; auto-approvals skip to EXECUTING.
- **Governance Events**: Post-run `GOVERNANCE_EVENT` with RAIL decisions; remediation suggestions as `THOUGHT`.
- **Correlation**: `tool_call_id` links to `RailDecision.event_id` for audits.

### Streaming & Async Support

- **SSE/WebSocket**: Lifecycle events stream in real-time.
- **Push**: Async clients receive updates with metadata; supports offline confirmation.

## Migration Guide

### From Non-Extension to Extension Usage

1. **Enable Extension**:
   ```bash
   export DEVELOPMENT_TOOL_EXTENSION_ENABLED=true
   # Restart: make run
   ```

2. **Update Client Code**:
   - **Discovery**: Query `/a2a/commands/get` instead of hardcoding tools.
   - **Execution**: Use `/a2a/command/execute` for structured args; monitor via task ID.
   - **Confirmations**: Parse `ConfirmationRequest` from `input_required` metadata; respond with `permissionOptionId`.
   - **Progress**: Handle `TOOL_CALL_UPDATE` in metadata for UI updates (e.g., progress bars).

   **Before (Legacy)**:
   ```javascript
   // Direct message send
   await sendMessage("Run git status");
   ```

   **After (Extension)**:
   ```javascript
   // Discover and execute slash command
   const commands = await fetch('/a2a/commands/get');
   const execution = await fetch('/a2a/command/execute', {
     method: 'POST',
     body: JSON.stringify({ command: 'git_status' })
   });
   const taskId = execution.execution_id;
   // Stream updates with metadata
   const stream = await fetch(`/a2a/message/stream?taskId=${taskId}`);
   ```

3. **Tool Configuration**:
   - Add `requires_confirmation` to `tools.yaml` for PENDING states.
   - Existing tools work unchanged; extension adds metadata.

4. **Push Notifications**:
   - Add extension events to `enabledEvents`: `["tool_call_update", "thought"]`.
   - Handle `DevelopmentToolEvent` in webhook payloads.

5. **Testing**:
   - Run [test_a2a_extension_compliance.py](tests/test_a2a_extension_compliance.py) for RFC flows.
   - Verify metadata in streams: `curl /a2a/message/stream`.

### Potential Breaking Changes

- None: Extension is opt-in; metadata ignored by legacy clients.
- **Validation**: Stricter param types in slash commands (from tool schemas).
- **Events**: New kinds may require filtering in consumers.

**Rollback**: Set `DEVELOPMENT_TOOL_EXTENSION_ENABLED=false`; endpoints 404, metadata omitted.

## Troubleshooting

### Common Issues

1. **Extension Not Declared in Agent Card**:
   - **Cause**: `DEVELOPMENT_TOOL_EXTENSION_ENABLED=false`.
   - **Fix**: Enable and restart server. Verify: `curl /.well-known/agent-card.json | jq .capabilities.extensions`.

2. **Slash Commands Not Found (404)**:
   - **Cause**: Extension disabled or no tools in `tools.yaml`.
   - **Fix**: Enable extension; add tools. Test: `curl /a2a/commands/get`.

3. **No Metadata in Events/Streams**:
   - **Cause**: Extension disabled or client < A2A v0.3.0.
   - **Fix**: Enable; use compatible client. Check logs: `grep "DevelopmentToolEvent" logs/a2a_acp.log`.

4. **Confirmation Not Triggering**:
   - **Cause**: Tool lacks `requires_confirmation: true` in `tools.yaml`.
   - **Fix**: Update config; reload tools (restart or hot-reload). Verify in stream: Look for `status: "pending"`.

5. **Tool Execution Fails with Metadata Errors**:
   - **Cause**: Serialization issue (e.g., non-JSON params).
   - **Fix**: Ensure tool params are serializable. Check [models.py](src/a2a_acp/models.py) for schema compliance.

6. **Push Notifications Missing Extension Data**:
   - **Cause**: `enabledEvents` excludes new kinds.
   - **Fix**: Update config: `["status_change", "tool_call_update"]`. Test webhook.

### Debugging Tips

- **Logs**: Set `LOG_LEVEL=DEBUG`; grep for "development-tool" or "TOOL_CALL_UPDATE".
- **Health Check**: `curl /health` confirms extension status.
- **Tests**: Run `pytest tests/test_a2a_extension_endpoints.py` for endpoint validation.
- **Client Simulation**: Use [tests/dummy_agent.py](tests/dummy_agent.py) with extension flags.

### Performance Considerations

- **Overhead**: ~5-10% latency from metadata serialization; negligible for most workflows.
- **Large Payloads**: Diffs in `FileDiff` limited to 10MB; truncate if needed.
- **Concurrent Tools**: Sandbox isolates executions; monitor via `/metrics/system`.

For support, see [AGENTS.md](AGENTS.md) or file issues.

---
**Extension Guide Complete!** 🔧 Ready for production development workflows.