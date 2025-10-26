# API Methods Reference

Complete reference for all A2A JSON-RPC methods implemented by A2A-ACP.

## Core A2A Methods

### `message/send`

Send a message and create a new task.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "message/send",
  "id": "msg_001",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        {"kind": "text", "text": "Hello!"},
        {"kind": "file", "file": {"bytes": "...", "mimeType": "image/png"}}
      ],
      "messageId": "msg_001",
      "contextId": "ctx_123"  // Optional: for conversation continuity
    },
    "metadata": {
      "agent_name": "codex-acp"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "msg_001",
  "result": {
    "id": "task_123",
    "contextId": "ctx_456",
    "status": {
      "state": "completed",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "history": [...],
    "artifacts": [...]
  }
}
```

### `message/stream`

Stream a message for real-time responses.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "message/stream",
  "id": "stream_001",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Explain this code"}],
      "messageId": "msg_001"
    },
    "metadata": {"agent_name": "codex-acp"}
  }
}
```

**Streaming Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "stream_001",
  "result": {
    "id": "task_123",
    "stream": "event-source"  // Server-Sent Events endpoint
  }
}
```

### `tasks/get`

Retrieve task information and history.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/get",
  "id": "get_001",
  "params": {
    "id": "task_123",
    "historyLength": 10  // Optional: limit history entries
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "get_001",
  "result": {
    "id": "task_123",
    "contextId": "ctx_456",
    "status": {
      "state": "completed",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "history": [...],
    "artifacts": [...]
  }
}
```

### `tasks/list`

List tasks with optional filtering.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/list",
  "id": "list_001",
  "params": {
    "contextId": "ctx_456",  // Optional: filter by context
    "limit": 20,            // Optional: max results
    "offset": 0             // Optional: pagination offset
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "list_001",
  "result": {
    "tasks": [
      {
        "id": "task_123",
        "contextId": "ctx_456",
        "status": {"state": "completed"},
        "createdAt": "2024-01-15T10:30:00Z"
      }
    ],
    "total": 1,
    "hasMore": false
  }
}
```

### `tasks/cancel`

Cancel a running task.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/cancel",
  "id": "cancel_001",
  "params": {
    "id": "task_123"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "cancel_001",
  "result": {
    "success": true,
    "message": "Task cancelled"
  }
}
```

### `tasks/provideInputAndContinue`

Resume an `input-required` task by supplying additional content or by selecting a permission decision.

**Request (textual input):**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/provideInputAndContinue",
  "id": "continue_001",
  "params": {
    "taskId": "task_123",
    "input": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Here are the missing details..."}],
      "messageId": "input_msg_001"
    }
  }
}
```

**Request (permission decision):**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/provideInputAndContinue",
  "id": "continue_perm_001",
  "params": {
    "taskId": "task_123",
    "permissionOptionId": "approved"
  }
}
```

- When `permissionOptionId` is supplied the `input` field must be omitted; the option must match one of the `permission_options` advertised in the input-required notification.
- If no permission prompt is pending and `permissionOptionId` is provided, the request fails with a validation error.

## HTTP Endpoints

### `GET /.well-known/agent-card.json`

**Agent Discovery Endpoint** - Standard well-known URL for A2A agent discovery.

This endpoint allows A2A clients to discover agent capabilities without authentication, following well-known URL conventions.

**Request:**
```bash
curl -X GET http://localhost:8001/.well-known/agent-card.json
```

**Response:**
```json
{
  "protocolVersion": "0.3.0",
  "name": "a2a-acp-agent",
  "description": "A2A-ACP Development Agent (with bash tool execution)",
  "url": "http://localhost:8001",
  "preferredTransport": "JSONRPC",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": true
  },
  "securitySchemes": {
    "bearer": {
      "type": "http",
      "scheme": "bearer",
      "description": "JWT bearer token authentication",
      "bearerFormat": "JWT"
    }
  },
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "skills": [...]
}
```

**Authentication**: Not required (public discovery endpoint)

### `GET /a2a/tasks/{task_id}/governor/history`

Fetch the full governance audit trail for a task, including auto-approval decisions, outstanding permission prompts, and governor feedback.

**Request:**
```bash
curl -H "Authorization: Bearer <token>" \
     http://localhost:8001/a2a/tasks/task_123/governor/history
```

**Response:**
```json
{
  "permissionDecisions": [
    {
      "toolCallId": "tool-1",
      "source": "policy:docs",
      "optionId": "approved",
      "governorsInvolved": [],
      "timestamp": "2024-01-15T10:31:00Z",
      "metadata": {
        "summary": ["Auto-approved docs/** change"]
      }
    }
  ],
  "pendingPermissions": [],
  "governorFeedback": [
    {
      "phase": "post_run",
      "timestamp": "2024-01-15T10:32:00Z",
      "summary": ["[code-reviewer] Needs attention: add tests"],
      "results": [
        {
          "governorId": "code-reviewer",
          "status": "needs_attention",
          "messages": ["Add coverage for error handling"],
          "followUpPrompt": "Please include unit tests for the new endpoint."
        }
      ]
    }
  ]
}
```

**Authentication**: Required; same headers as the main A2A RPC endpoint.

## Agent Methods

### `agent/getAuthenticatedExtendedCard`

Get comprehensive agent capabilities and metadata.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "agent/getAuthenticatedExtendedCard",
  "id": "card_001",
  "params": {}
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "card_001",
  "result": {
    "id": "agent_001",
    "name": "codex-acp",
    "description": "OpenAI Codex for A2A-ACP",
    "version": "1.0.0",
    "capabilities": {
      "tasks": true,
      "contexts": true,
      "streaming": true,
      "files": true,
      "toolExecution": true
    },
    "skills": [...],
    "supportedProtocols": ["a2a/0.3.0"],
    "metadata": {
      "provider": "openai",
      "model": "codex"
    }
  }
}
```

## Push Notification Methods

### `tasks/pushNotificationConfig/set`

Configure webhook notifications for a task.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/pushNotificationConfig/set",
  "id": "config_001",
  "params": {
    "taskId": "task_123",
    "config": {
      "id": "webhook_001",
      "url": "https://your-app.com/webhooks/a2a",
      "token": "your-bearer-token",
      "enabledEvents": ["status_change", "message", "input_required"],
      "quietHoursStart": "22:00",
      "quietHoursEnd": "08:00"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "config_001",
  "result": {
    "id": "webhook_001",
    "status": "active",
    "createdAt": "2024-01-15T10:30:00Z"
  }
}
```

### `tasks/pushNotificationConfig/get`

Retrieve notification configuration.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/pushNotificationConfig/get",
  "id": "get_config_001",
  "params": {
    "taskId": "task_123",
    "configId": "webhook_001"
  }
}
```

### `tasks/pushNotificationConfig/list`

List all notification configurations for a task.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/pushNotificationConfig/list",
  "id": "list_config_001",
  "params": {
    "taskId": "task_123"
  }
}
```

### `tasks/pushNotificationConfig/delete`

Delete a notification configuration.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tasks/pushNotificationConfig/delete",
  "id": "delete_config_001",
  "params": {
    "taskId": "task_123",
    "configId": "webhook_001"
  }
}
```

## Notification Event Formats

### Task Status Change

```json
{
  "event": "status_change",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "old_state": "working",
    "new_state": "completed"
  }
}
```

### New Message

```json
{
  "event": "message",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "message_role": "assistant",
    "message_content": "Task completed successfully!"
  }
}
```

### Input Required

```json
{
  "event": "input_required",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "prompt": "Please provide additional clarification...",
    "input_types": ["text/plain"]
  }
}
```

### Artifact Created

```json
{
  "event": "artifact",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "artifact_type": "file",
    "artifact_name": "result.txt",
    "artifact_size": 1024
  }
}
```

## Error Responses

All methods return JSON-RPC 2.0 compliant error responses:

```json
{
  "jsonrpc": "2.0",
  "id": "request_001",
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "details": "Missing required field: message"
    }
  }
}
```

### Common Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32600 | Invalid Request | Malformed JSON-RPC request |
| -32601 | Method not found | Unknown method |
| -32602 | Invalid params | Invalid or missing parameters |
| -32603 | Internal error | Server-side error |
| -32000 | Task not found | Specified task doesn't exist |
| -32001 | Context not found | Specified context doesn't exist |
| -32002 | Agent not available | Zed ACP agent is not responding |

## Authentication

A2A-ACP supports multiple authentication schemes:

### Bearer Token (Recommended)

```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:8001/a2a/rpc
```

### API Key

```bash
curl -H "X-API-Key: your-api-key" \
     http://localhost:8001/a2a/rpc
```

### No Authentication (Development)

```bash
# Only use for local development
curl http://localhost:8001/a2a/rpc
```

## Content Types

### Message Parts

A2A-ACP supports rich content through message parts:

#### Text Content
```json
{
  "kind": "text",
  "text": "Hello, world!"
}
```

#### File Content
```json
{
  "kind": "file",
  "file": {
    "bytes": "base64-encoded-content",
    "mimeType": "image/png",
    "name": "screenshot.png"
  }
}
```

#### Data Content
```json
{
  "kind": "data",
  "data": {
    "content": "structured-data",
    "mimeType": "application/json"
  }
}
```

## Rate Limiting

Default rate limits (configurable):

- **Per-minute limit**: 60 requests
- **Per-hour limit**: 1000 requests
- **Concurrent requests**: 10 simultaneous

Rate limit headers in responses:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1640995200
```

---

**API reference complete!** 📚 For examples and use cases, see the [examples/](examples/) folder.
