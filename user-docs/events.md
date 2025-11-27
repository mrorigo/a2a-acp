# Event System

Comprehensive guide to **A2A-ACP's event emission system** for real-time monitoring and integration.

## Overview

A2A-ACP emits structured events throughout the system lifecycle, enabling **real-time monitoring**, **push notifications**, and **event-driven integrations**.

## Event Types

### Task Events

#### Task Status Changes
**Emitted when**: Task state changes (submitted → working → completed/failed/cancelled)

**Trigger locations**:
- `task_manager.py`: Task creation, execution start/completion, cancellation
- `bash_executor.py`: Tool execution completion/failure
- `zed_agent.py`: Agent session lifecycle events

**Event structure**:
```json
{
  "event": "status_change",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "old_state": "working",
    "new_state": "completed",
    "message": "Task completed successfully"
  }
}
```

#### Task Messages
**Emitted when**: New messages are added to task history

**Trigger locations**:
- `task_manager.py`: Agent responses and user input messages
- `bash_executor.py`: Tool execution results as messages

**Event structure**:
```json
{
  "event": "message",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "message_role": "assistant",
    "message_content": "Analysis complete",
    "message_type": "agent_response"
  }
}
```

#### Input Required Events
**Emitted when**: Agent requests user input or confirmation

**Trigger locations**:
- `task_manager.py`: Protocol-compliant input detection
- `bash_executor.py`: Tool confirmation requests

When the input request originates from a tool permission prompt, the notification `data` includes:

- `permission_options`: list of available actions the client may choose
- `tool_call`: structured description of the requested tool call, including diff previews and parameters
- `governor_summary`: bullet-point feedback from all governors that reviewed the request
- `policy_decision`: information about any auto-approval rule that applied

**Event structure**:
```json
{
  "event": "input_required",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "prompt": "Please provide database credentials...",
    "input_types": ["text/plain"],
    "detection_method": "protocol_compliant",
    "permission_options": [
      {"optionId": "approved", "name": "Approve"},
      {"optionId": "abort", "name": "Reject"}
    ],
    "governor_summary": ["[security-diff-check] Needs attention: file touches secrets.env"],
    "tool_call": {"toolId": "functions.acp_fs__write_text_file", "path": "config/secrets.env"},
    "metadata": {
      "development-tool": {
        "kind": "tool_call_update",
        "tool_call": {
          "tool_call_id": "tc_001",
          "status": "pending",
          "confirmation_request": {
            "options": [{"id": "approve", "name": "Allow"}],
            "details": {"description": "Write to secrets file?"}
          }
        }
      }
    }
  }
}
```

#### Governor Follow-up Events
**Emitted when**: A post-run governor injects an automatic follow-up prompt back into the Codex session.

**Trigger locations**:
- `task_manager.py`: `_run_post_run_governors` when evaluating governor responses

**Event structure**:
```json
{
  "event": "task_governor_followup",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "governor_id": "code-reviewer",
    "prompt": "Please include unit tests for the new API endpoint.",
    "iteration": 0
  }
}
```

#### Governor Feedback Required Events
**Emitted when**: Post-run governors block the final response and require a human override.

**Trigger locations**:
- `task_manager.py`: `_run_post_run_governors` when a governor returns `status="reject"`

**Event structure**:
```json
{
  "event": "task_feedback_required",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:32:00Z",
  "data": {
    "message": "Governor blocked final response",
    "summary": ["[compliance-http] Reject: response references internal API keys"]
  }
}
```

#### Artifact Events
**Emitted when**: Files or data structures are created

**Trigger locations**:
- `task_manager.py`: File creation detection in agent responses
- `bash_executor.py`: Tool execution creates output files

**Event structure**:
```json
{
  "event": "artifact",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "artifact_type": "file",
    "artifact_name": "report.pdf",
    "artifact_size": 1024,
    "detection_text": "File created: report.pdf"
  }
}
```

### Tool Execution Events

#### Tool Started
**Emitted when**: Tool execution begins

**Trigger locations**:
- `bash_executor.py`: Before script execution

**Event structure**:
```json
{
  "event": "tool_execution",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "tool_id": "web_request",
    "status": "started",
    "parameters": {"method": "GET", "url": "https://api.example.com"},
    "execution_context": "production",
    "metadata": {
      "development-tool": {
        "kind": "tool_call_update",
        "tool_call": {
          "tool_call_id": "tc_001",
          "status": "executing",
          "tool_name": "web_request"
        }
      }
    }
  }
}
```

#### Tool Completed
**Emitted when**: Tool execution finishes successfully

**Trigger locations**:
- `bash_executor.py`: After successful script execution

**Event structure**:
```json
{
  "event": "tool_execution",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:01Z",
  "data": {
    "tool_id": "web_request",
    "status": "completed",
    "execution_time": 0.5,
    "return_code": 0,
    "output_length": 512,
    "cached": false,
    "output": {
      "content": "HTTP 200 OK",
      "details": {
        "stdout": "Response: {\"status\": \"success\"}\n",
        "exit_code": 0
      }
    },
    "metadata": {
      "development-tool": {
        "kind": "tool_call_update",
        "tool_call": {
          "tool_call_id": "tc_001",
          "status": "succeeded",
          "result": {
            "content": "HTTP 200 OK",
            "details": {"stdout": "Response body...", "exit_code": 0}
          }
        }
      }
    }
  }
}
```

#### Tool Failed
**Emitted when**: Tool execution fails

**Trigger locations**:
- `bash_executor.py`: When script execution returns error

**Event structure**:
```json
{
  "event": "tool_execution",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:01Z",
  "data": {
    "tool_id": "database_query",
    "status": "failed",
    "error": "Connection timeout",
    "execution_time": 5.2,
    "retry_attempt": 2,
    "will_retry": true,
    "metadata": {
      "development-tool": {
        "kind": "tool_call_update",
        "tool_call": {
          "tool_call_id": "tc_002",
          "status": "failed",
          "result": {
            "message": "Connection timeout",
            "code": "DB_TIMEOUT",
            "details": {"attempt": 2}
          }
        }
      }
    }
  }
}
```

## Development Tool Extension Events

The development-tool extension enhances A2A-ACP's event system with structured metadata for tool interactions. When enabled (`DEVELOPMENT_TOOL_EXTENSION_ENABLED=true`), events include `DevelopmentToolEvent` objects in the `metadata` field, enabling clients to render tool progress, confirmations, and thoughts natively.

### DevelopmentToolEvent Metadata Types

`DevelopmentToolEvent` appears in `TaskStatusUpdateEvent` and push notifications, namespaced under the extension URI:

```json
{
  "metadata": {
    "https://developers.google.com/gemini/a2a/extensions/development-tool/v1": {
      "kind": "tool_call_update",  // DevelopmentToolEventKind
      "model": "gemini-1.5-pro",
      "user_tier": "premium",
      "data": { ... }  // ToolCall, AgentThought, etc.
    }
  }
}
```

#### Supported Event Kinds (DevelopmentToolEventKind)

| Kind | Description | Example Use Case |
|------|-------------|------------------|
| `TOOL_CALL_UPDATE` | Tool lifecycle update (PENDING, EXECUTING, SUCCEEDED, FAILED, CANCELLED) | Progress during bash tool execution |
| `TOOL_CALL_CONFIRMATION` | User confirmation received for pending tool | After `tasks/provideInputAndContinue` |
| `TEXT_CONTENT` | Plain text output from tool | Simple stdout capture |
| `STATE_CHANGE` | Task state transition with extension context | From input-required to working |
| `THOUGHT` | Agent reasoning step | "Analyzing parameters for security..." |
| `GOVERNANCE_EVENT` | Policy decision or remediation suggestion | Post-run governor feedback |

#### Tool Lifecycle Event Kinds

Tool calls follow a structured lifecycle, emitted via `TOOL_CALL_UPDATE`:

1. **PENDING**: Confirmation required (if `requires_confirmation: true` in tool config).
   ```json
   {
     "kind": "tool_call_update",
     "tool_call": {
       "tool_call_id": "tc_001",
       "status": "pending",
       "tool_name": "database_query",
       "input_parameters": {"query": "DELETE FROM users"},
       "confirmation_request": {
         "options": [{"id": "approve", "name": "Allow"}],
         "details": {"description": "Dangerous SQL query detected"}
       }
     }
   }
   ```

2. **EXECUTING**: Sandbox execution started.
   ```json
   {
     "kind": "tool_call_update",
     "tool_call": {
       "tool_call_id": "tc_001",
       "status": "executing",
       "live_content": "Connecting to database..."
     }
   }
   ```

3. **SUCCEEDED**: Execution complete with output.
   ```json
   {
     "kind": "tool_call_update",
     "tool_call": {
       "tool_call_id": "tc_001",
       "status": "succeeded",
       "result": {
         "content": "Query executed successfully",
         "details": {"stdout": "10 rows affected", "exit_code": 0}
       }
     }
   }
   ```

4. **FAILED**: Execution error.
   ```json
   {
     "kind": "tool_call_update",
     "tool_call": {
       "tool_call_id": "tc_001",
       "status": "failed",
       "result": {
         "message": "Connection refused",
         "code": "DB_CONN_ERROR"
       }
     }
   }
   ```

5. **CANCELLED**: User rejected or timeout.
   ```json
   {
     "kind": "tool_call_update",
     "tool_call": {
       "tool_call_id": "tc_001",
       "status": "cancelled",
       "result": {"message": "User cancelled confirmation"}
     }
   }
   ```

#### Agent Thoughts (THOUGHT Kind)
Emitted for reasoning transparency:
```json
{
  "kind": "thought",
  "content": "Reviewing query for potential data loss before execution.",
  "timestamp": "2025-10-30T13:40:00Z"
}
```

#### Extension-Specific Event Payloads Examples

1. **Confirmation Flow with Tool Call**:
   Full `input_required` event integrating extension metadata:
   ```json
   {
     "event": "input_required",
     "task_id": "task_123",
     "data": {
       "permission_options": [{"optionId": "approve", "name": "Proceed"}],
       "tool_call": {
         "tool_call_id": "tc_001",
         "status": "pending",
         "confirmation_request": {
           "options": [{"id": "approve", "name": "Allow"}],
           "details": {"description": "Execute shell command?"}
         }
       },
       "metadata": {
         "development-tool": {
           "kind": "tool_call_update",
           "tool_call": { ... }  // Full ToolCall object
         }
       }
     }
   }
   ```

2. **Governance Event (Post-Run Feedback)**:
   Integrates with RAIL for policy decisions:
   ```json
   {
     "event": "task_governor_followup",
     "task_id": "task_123",
     "data": {
       "governor_id": "code-reviewer",
       "prompt": "Add unit tests for new endpoint",
       "metadata": {
         "development-tool": {
           "kind": "governance_event",
           "data": {
             "policy_phase": "post_run",
             "final_decision": "needs_attention",
             "governor_summary": "Add coverage for error handling"
           }
         }
       }
     }
   }
   ```

3. **Push Notification Enhancements**
   Push notifications now include extension metadata for async clients:
   - **Enhanced Payloads**: `DevelopmentToolEvent` embedded in `data.metadata`.
   - **New Event Types**: `tool_call_update`, `thought`, `governance_event` via push configs.
   - **Async Confirmation**: Clients receive PENDING events offline; respond via API when reconnected.

   **Example Push Payload for Tool Update**:
   ```json
   {
     "event": "status_change",
     "task_id": "task_123",
     "timestamp": "2025-10-30T13:40:00Z",
     "data": {
       "old_state": "working",
       "new_state": "working",
       "metadata": {
         "development-tool": {
           "kind": "tool_call_update",
           "tool_call": {
             "status": "succeeded",
             "result": {"content": "Command output..."}
           }
         }
       }
     }
   }
   ```

   **Configuration for Extension Events**:
   Include extension kinds in `enabledEvents`:
   ```json
   {
     "enabledEvents": ["status_change", "input_required", "tool_call_update", "thought"]
   }
   ```

   Enhancements:
   - **Rich Payloads**: Structured `ToolOutput`/`ErrorDetails` for better error handling.
   - **Retry Support**: Failed deliveries retry with exponential backoff, preserving metadata.
   - **Filtering**: Clients filter by `kind` (e.g., ignore `GOVERNANCE_EVENT` if unsupported).
   - **Security**: Metadata signed with HMAC; sensitive params redacted.

For schema details, see [DEVELOPMENT_TOOL_EXTENSION.md](docs/DEVELOPMENT_TOOL_EXTENSION.md).

### Security Events

#### Authentication Events
**Emitted when**: Authentication attempts occur

**Trigger locations**:
- `main.py`: API authentication middleware
- `zed_agent.py`: Agent credential validation

**Event structure:**
```json
{
  "event": "authentication",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "type": "api_key",
    "status": "success",
    "client_ip": "192.168.1.100",
    "user_agent": "A2A-Client/1.0"
  }
}
```

#### Authentication Events
**Emitted when**: Authentication attempts occur

**Trigger locations**:
- `main.py`: API authentication middleware
- `zed_agent.py`: Agent credential validation

**Event structure**:
```json
{
  "event": "authentication",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "type": "api_key",
    "status": "success",
    "client_ip": "192.168.1.100",
    "user_agent": "A2A-Client/1.0"
  }
}
```

#### Audit Events
**Emitted when**: Security-relevant actions occur

**Trigger locations**:
- `audit.py`: All security event logging
- `sandbox.py`: Permission and access control events

**Event structure**:
```json
{
  "event": "security_audit",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "event_type": "tool_execution_started",
    "user_id": "user_123",
    "session_id": "session_456",
    "tool_id": "dangerous_operation",
    "severity": "high",
    "details": {"requires_confirmation": true}
  }
}
```

## Event Emission Flow

### 1. Event Generation

Events are generated at the source component:

```python
# In task_manager.py
async def _send_task_notification(self, task_id: str, event: str, event_data: Dict):
    """Send a push notification for a task event."""
    if self.push_notification_manager:
        await self.push_notification_manager.send_notification(task_id, {
            "event": event,
            "task_id": task_id,
            **event_data
        })
```

### 2. Event Enrichment

Events are enriched with context and metadata:

```python
# Add timestamp, task context, component info
enriched_event = {
    "event": original_event["event"],
    "task_id": task_id,
    "timestamp": generate_timestamp(),
    "component": "task_manager",
    "version": "1.0.0",
    "data": original_event["data"]
}
```

### 3. Event Routing

Events are routed to multiple destinations:

- **Push Notifications**: Immediate HTTP webhook delivery
- **Audit Logging**: Persistent storage for compliance
- **Real-time Streaming**: WebSocket/SSE for live updates
- **Metrics Collection**: Performance and usage analytics

### 4. Event Consumption

Events can be consumed via multiple channels:

- **HTTP Webhooks**: Configured push notification endpoints
- **WebSocket Connections**: Real-time event streaming
- **Server-Sent Events**: Browser-based real-time updates
- **Audit Logs**: Historical event retrieval and analysis

## Event Configuration

### Per-Task Event Configuration

```bash
# Configure events for specific task
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/set",
    "id": "event_config_001",
    "params": {
      "taskId": "task_123",
      "config": {
        "id": "my_events",
        "enabledEvents": ["status_change", "input_required", "tool_execution"],
        "disabledEvents": ["message"],
        "minPriority": "normal"
      }
    }
  }'
```

### Global Event Filtering

```bash
# Environment-based event filtering
export A2A_EVENT_FILTER_LEVEL="normal"  # low, normal, high, critical
export A2A_ENABLE_AUDIT_EVENTS=true
export A2A_ENABLE_PERFORMANCE_EVENTS=true
```

## Event Consumption

### HTTP Webhooks

```bash
# Receive events via HTTP POST
curl -X POST https://your-app.com/events \
  -H "Content-Type: application/json" \
  -H "X-A2A-Signature: hmac_signature" \
  -d @event-payload.json
```

### WebSocket Streaming

```javascript
// Connect to event stream
const ws = new WebSocket('ws://localhost:8001/events');

ws.onmessage = (event) => {
  const a2aEvent = JSON.parse(event.data);
  console.log('Received event:', a2aEvent.event, a2aEvent.data);
};
```

### Server-Sent Events

```bash
# Receive events via SSE
curl -X GET "http://localhost:8001/events/stream" \
  -H "Authorization: Bearer your-token"
```

## Event Timing

### Synchronous Events
**Emitted immediately** when action occurs:

- Task status changes
- Message additions
- Authentication events
- Tool execution results

### Asynchronous Events
**Emitted via background tasks** for non-blocking operation:

- Push notification delivery
- Audit log writing
- Metrics collection
- Cleanup operations

### Event Ordering

Events are emitted in **logical order** but may be delivered asynchronously:

1. **Task Events**: Sequential by task lifecycle
2. **Tool Events**: In execution order within tasks
3. **Security Events**: Immediate for auditing requirements
4. **Notification Events**: Best-effort delivery order

## Event Reliability

### Delivery Guarantees

- **At-least-once delivery**: Events may be duplicated in failure scenarios
- **Ordered by task**: Events for same task maintain logical order
- **Best-effort timing**: Events delivered as soon as possible

### Retry Logic

Failed event deliveries are retried with exponential backoff:

```python
# Configurable retry strategy
retry_config = {
    "max_attempts": 5,
    "initial_delay": 1.0,
    "max_delay": 300.0,
    "backoff_multiplier": 2.0,
    "jitter": True
}
```

## Monitoring Events

### Event Metrics

Monitor event system health:

```bash
# Get event system metrics
curl -X GET "http://localhost:8000/metrics/events" \
  -H "Authorization: Bearer your-token"

# Response includes:
{
  "events_generated": 1500,
  "events_delivered": 1485,
  "events_failed": 15,
  "delivery_success_rate": 99.0,
  "average_delivery_time": 0.25
}
```

### Event Debugging

Enable event debugging for troubleshooting:

```bash
# Enable event debugging
export A2A_DEBUG_EVENTS=true

# View event logs
tail -f logs/a2a_acp.log | grep -E "(EVENT|event)"
```

## Event Security

### Event Signing

Events can be cryptographically signed for verification:

```bash
# HMAC signature in header
X-A2A-Signature: sha256=abc123...
X-A2A-Timestamp: 2024-01-15T10:30:00Z
```

### Event Filtering

Prevent sensitive data leakage:

```python
# Automatic filtering of sensitive fields
sensitive_fields = ["password", "api_key", "token", "secret"]
filtered_event = filter_sensitive_data(original_event, sensitive_fields)
```

## Best Practices

### Event Producers

1. **Consistent Structure**: Use standard event schemas
2. **Rich Context**: Include relevant metadata for debugging
3. **Appropriate Timing**: Emit events at correct lifecycle points
4. **Error Handling**: Don't fail main flow if event emission fails

### Event Consumers

1. **Idempotency**: Handle duplicate events gracefully
2. **Error Handling**: Manage webhook failures and retries
3. **Security**: Validate event signatures and authenticity
4. **Monitoring**: Track event delivery success rates

### Operations Teams

1. **Alerting**: Set up alerts for event system failures
2. **Analytics**: Monitor event patterns and volumes
3. **Compliance**: Ensure audit events meet regulatory requirements
4. **Performance**: Tune event delivery for system load

## Troubleshooting Events

### Events Not Received

**Check webhook configuration:**
```bash
# Verify webhook exists and is active
curl -X POST "http://localhost:8001/tasks/pushNotificationConfig/list" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "task_123"}'
```

**Check event generation:**
```bash
# Enable event debugging
export LOG_LEVEL=DEBUG

# Look for event emission in logs
grep "EVENT_EMITTED" logs/a2a_acp.log
```

### Event Delivery Failures

**Check webhook endpoint:**
```bash
# Test webhook manually
curl -X POST https://your-app.com/webhooks/a2a \
  -H "Content-Type: application/json" \
  -d '{"test": "event"}'
```

**Check network connectivity:**
```bash
# Verify webhook domain is reachable
ping your-app.com
curl -v https://your-app.com/webhooks/a2a
```

### Event Ordering Issues

**Understand event timing:**
- Events are emitted when actions occur, not in strict chronological order
- Use `timestamp` field for precise timing requirements
- Group events by `task_id` for task-specific ordering

---

**Event system documented!** 📡 For real-time integration guides, see [push-notifications.md](push-notifications.md).
