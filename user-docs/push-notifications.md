# Push Notifications

A2A-ACP includes comprehensive **push notification support** for real-time task monitoring and event-driven workflows.

## Overview

Push notifications enable **real-time monitoring** of A2A-ACP task events:

- **🔗 HTTP Webhooks**: Configurable webhook endpoints for task events
- **📊 Event Filtering**: Sophisticated filtering by event type, quiet hours, and priority
- **📈 Delivery Analytics**: Real-time tracking of delivery success rates and performance
- **🔄 Retry Logic**: Exponential backoff with configurable retry attempts
- **🧹 Automatic Cleanup**: Lifecycle-based cleanup of notification configurations
- **📡 Real-time Streaming**: WebSocket and Server-Sent Events integration

### Event-Driven Architecture

Push notifications are **triggered by events** emitted throughout the A2A-ACP system:

**Events → Push Notifications → Webhooks → Your Application**

- **Event Generation**: Events emitted when actions occur (task status changes, tool executions, etc.)
- **Event Routing**: Events routed to push notification system for delivery
- **Webhook Delivery**: HTTP POST to configured webhook endpoints
- **Application Integration**: Your application receives and processes notifications

## Quick Setup

### 1. Configure Webhook

```bash
# Set up a webhook endpoint for task status changes
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/set",
    "id": "config_001",
    "params": {
      "taskId": "task_123",
      "config": {
        "id": "webhook_001",
        "url": "https://your-app.com/webhooks/a2a",
        "token": "your-bearer-token",
        "enabledEvents": ["status_change", "message", "artifact"],
        "quietHoursStart": "22:00",
        "quietHoursEnd": "08:00"
      }
    }
  }'
```

### 2. Send a Message

```bash
# Send a message to trigger notifications
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "msg_001",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Analyze this data"}],
        "messageId": "msg_001"
      },
      "metadata": {"agent_name": "codex-acp"}
    }
  }'
```

### 3. Receive Notifications

Your webhook receives real-time notifications:

```json
{
  "event": "status_change",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "old_state": "submitted",
    "new_state": "working"
  }
}
```

## Configuration

### Webhook Configuration

```json
{
  "id": "my_webhook",
  "url": "https://your-app.com/webhooks/a2a",
  "token": "your-bearer-token",
  "enabledEvents": ["status_change", "message", "input_required"],
  "disabledEvents": ["internal_event"],
  "quietHoursStart": "22:00",
  "quietHoursEnd": "08:00",
  "retryConfig": {
    "maxAttempts": 3,
    "initialDelay": 1,
    "maxDelay": 30,
    "backoffMultiplier": 2
  }
}
```

### Environment Variables

```bash
# Push notification settings
export PUSH_NOTIFICATIONS_ENABLED=true
export PUSH_NOTIFICATION_WEBHOOK_TIMEOUT=30
export PUSH_NOTIFICATION_RETRY_ATTEMPTS=3
export PUSH_NOTIFICATION_BATCH_SIZE=10

# Security
export PUSH_NOTIFICATION_HMAC_SECRET="your-secret-key"
export PUSH_NOTIFICATION_RATE_LIMIT_PER_MINUTE=60

# Cleanup
export PUSH_NOTIFICATION_CLEANUP_ENABLED=true
export PUSH_NOTIFICATION_CLEANUP_INTERVAL=3600
```

## Notification Events

### Task Status Changes

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

### New Messages

```json
{
  "event": "message",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "message_role": "assistant",
    "message_content": "Task completed successfully!",
    "message_type": "agent_response"
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
    "input_types": ["text/plain"],
    "detection_method": "protocol_compliant"
  }
}
```

### Artifact Updates

```json
{
  "event": "artifact",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "artifact_type": "file",
    "artifact_name": "result.txt",
    "artifact_size": 1024,
    "detection_text": "File created: result.txt"
  }
}
```

### Tool Executions

```json
{
  "event": "tool_execution",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "tool_id": "web_request",
    "status": "completed",
    "execution_time": 0.5,
    "return_code": 0,
    "output_length": 512
  }
}
```

## Event Filtering

### Event Types

Control which events trigger notifications:

```json
{
  "enabledEvents": [
    "status_change",      // Task state changes
    "message",           // New messages added
    "input_required",    // Agent requests user input
    "artifact",          // Files or data created
    "tool_execution"     // Tool execution events
  ],
  "disabledEvents": [
    "internal_event"     // System internal events
  ]
}
```

### Quiet Hours

Suppress notifications during specified hours:

```json
{
  "quietHoursStart": "22:00",  // 10 PM
  "quietHoursEnd": "08:00",    // 8 AM
  "quietHoursTimezone": "UTC"
}
```

### Priority Filtering

Filter by event priority levels:

```json
{
  "minPriority": "normal",      // minimum priority to trigger notification
  "maxPriority": "critical",   // maximum priority level
  "priorityLevels": {
    "low": ["message"],
    "normal": ["status_change", "artifact"],
    "high": ["input_required"],
    "critical": ["failed", "cancelled"]
  }
}
```

## Authentication

### Bearer Token

```json
{
  "url": "https://api.example.com/webhooks",
  "authentication": {
    "type": "bearer",
    "token": "your-bearer-token-here"
  }
}
```

### API Key

```json
{
  "url": "https://api.example.com/webhooks",
  "authentication": {
    "type": "apikey",
    "headerName": "X-API-Key",
    "apiKey": "your-api-key-here"
  }
}
```

### Custom Authentication

```json
{
  "url": "https://api.example.com/webhooks",
  "authentication": {
    "type": "custom",
    "headerName": "Authorization",
    "headerValue": "Custom your-auth-scheme"
  }
}
```

## Retry Logic

### Exponential Backoff

Failed webhooks are retried with exponential backoff:

```json
{
  "retryConfig": {
    "maxAttempts": 5,           // Maximum retry attempts
    "initialDelay": 1,          // Initial delay in seconds
    "maxDelay": 300,           // Maximum delay between retries
    "backoffMultiplier": 2,     // Delay multiplier per retry
    "jitter": true             // Add random jitter to prevent thundering herd
  }
}
```

### Retry Example

```bash
# Attempt 1: Immediate (1 second delay)
POST https://your-app.com/webhooks/a2a

# Attempt 2: 2 seconds later
POST https://your-app.com/webhooks/a2a

# Attempt 3: 4 seconds later
POST https://your-app.com/webhooks/a2a

# Attempt 4: 8 seconds later
POST https://your-app.com/webhooks/a2a

# Attempt 5: 16 seconds later (final attempt)
POST https://your-app.com/webhooks/a2a
```

## Delivery Analytics

### Metrics Endpoint

```bash
# Get delivery analytics
curl -X GET "http://localhost:8000/metrics/push-notifications" \
  -H "Authorization: Bearer your-token"
```

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "delivery_stats": {
    "total_sent": 150,
    "successful_deliveries": 147,
    "failed_deliveries": 3,
    "success_rate": 98.0,
    "average_response_time": 0.25
  },
  "events_by_type": {
    "status_change": 75,
    "message": 60,
    "input_required": 15
  },
  "webhook_configs": {
    "total": 5,
    "active": 3,
    "expired": 2
  }
}
```

### Webhook Health

Monitor individual webhook health:

```json
{
  "webhook_id": "webhook_001",
  "url": "https://your-app.com/webhooks/a2a",
  "status": "healthy",
  "last_success": "2024-01-15T10:25:00Z",
  "last_failure": null,
  "total_requests": 50,
  "success_rate": 100.0,
  "average_response_time": 0.23
}
```

## Multiple Webhooks

### Configure Multiple Endpoints

```bash
# Webhook for status monitoring
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/set",
    "id": "status_webhook_001",
    "params": {
      "taskId": "task_123",
      "config": {
        "id": "status_monitoring",
        "url": "https://monitoring.example.com/webhooks/status",
        "enabledEvents": ["status_change"]
      }
    }
  }'

# Webhook for user notifications
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/set",
    "id": "user_webhook_001",
    "params": {
      "taskId": "task_123",
      "config": {
        "id": "user_notifications",
        "url": "https://ui.example.com/webhooks/notifications",
        "enabledEvents": ["input_required", "completed"]
      }
    }
  }'
```

### Webhook Management

```bash
# List all webhook configurations
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/list",
    "id": "list_webhooks_001",
    "params": {
      "taskId": "task_123"
    }
  }'

# Update webhook configuration
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/set",
    "id": "update_webhook_001",
    "params": {
      "taskId": "task_123",
      "configId": "status_monitoring",
      "config": {
        "enabledEvents": ["status_change", "failed"]
      }
    }
  }'

# Delete webhook configuration
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/delete",
    "id": "delete_webhook_001",
    "params": {
      "taskId": "task_123",
      "configId": "status_monitoring"
    }
  }'
```

## Use Cases

### 1. Real-time Dashboards

```bash
# Set up webhook for dashboard updates
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/set",
    "id": "dashboard_webhook_001",
    "params": {
      "taskId": "dashboard_task",
      "config": {
        "id": "dashboard_updates",
        "url": "https://dashboard.company.com/api/updates",
        "enabledEvents": ["status_change", "message"],
        "token": "dashboard-api-token"
      }
    }
  }'
```

### 2. CI/CD Integration

```bash
# Notify CI/CD system of task completion
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/set",
    "id": "cicd_webhook_001",
    "params": {
      "taskId": "deployment_task",
      "config": {
        "id": "cicd_notifications",
        "url": "https://ci-cd.company.com/webhooks/a2a",
        "enabledEvents": ["completed", "failed"],
        "token": "cicd-integration-token"
      }
    }
  }'
```

### 3. User Notification System

```bash
# Notify users when input is required
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/set",
    "id": "user_notification_webhook_001",
    "params": {
      "taskId": "user_task_123",
      "config": {
        "id": "user_notifications",
        "url": "https://notifications.company.com/user/123",
        "enabledEvents": ["input_required"],
        "quietHoursStart": "22:00",
        "quietHoursEnd": "08:00"
      }
    }
  }'
```

## Best Practices

### Webhook Development

1. **Idempotency**: Handle duplicate notifications gracefully
2. **Response Time**: Respond within 30 seconds to avoid timeouts
3. **Error Handling**: Log and handle webhook failures appropriately
4. **Security**: Validate webhook signatures if using HMAC

### Configuration Management

1. **Unique IDs**: Use unique, descriptive IDs for webhook configurations
2. **Cleanup**: Regularly clean up expired or unused webhook configurations
3. **Monitoring**: Monitor webhook health and delivery success rates
4. **Testing**: Test webhook configurations before deploying to production

### Security

1. **Token Rotation**: Rotate authentication tokens regularly
2. **Network Security**: Use HTTPS for all webhook endpoints
3. **Signature Validation**: Validate HMAC signatures when provided
4. **Rate Limiting**: Implement rate limiting to prevent abuse

## Troubleshooting

### Common Issues

**"Webhook delivery failed"**
```bash
# Check webhook URL is reachable
curl -I https://your-app.com/webhooks/a2a

# Verify authentication
curl -H "Authorization: Bearer your-token" \
     https://your-app.com/webhooks/a2a
```

**"Notifications not received"**
```bash
# Check if notifications are enabled
curl -X GET "http://localhost:8000/metrics/push-notifications" \
  -H "Authorization: Bearer your-token"

# Verify webhook configuration exists
curl -X POST "http://localhost:8001/tasks/pushNotificationConfig/list" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "task_123"}'
```

**"Authentication failed"**
```bash
# Check webhook authentication configuration
# Verify token is correct and not expired
# Ensure webhook accepts the authentication method being used
```

---

**Push notifications configured!** 📡 For real-time task monitoring, see [interactive-conversations.md](interactive-conversations.md) for input-required workflows.