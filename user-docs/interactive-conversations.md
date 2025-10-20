# Interactive Conversations

A2A-ACP supports **interactive conversations** where agents can pause execution and request user input for clarification, multi-step workflows, or sensitive operations.

## Overview

Interactive conversations enable **multi-turn workflows** where:

- **Agents can ask questions** when they need more information
- **Users provide clarification** without breaking conversation flow
- **Sensitive operations require confirmation** before execution
- **Long-running tasks can be paused** and resumed
- **State is maintained** across interaction boundaries

## How It Works

### 1. Agent Requests Input

When an agent needs user input, it sends an `end_turn` response with empty `toolCalls`:

```json
{
  "stopReason": "end_turn",
  "toolCalls": [],
  "_meta": {
    "input_types": ["text/plain", "application/json"],
    "prompt": "Please provide the missing data..."
  }
}
```

### 2. System Detects Input Required

A2A-ACP automatically detects this pattern and:

- **Changes task state** to `input-required`
- **Emits A2A event** for input-required notification
- **Sends push notification** if webhooks are configured
- **Waits for user response** via `provide_input_and_continue`

### 3. User Provides Input

Users continue the conversation with additional information:

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/provideInputAndContinue",
  "id": "input_001",
  "params": {
    "taskId": "task_123",
    "input": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Here is the missing information..."}],
      "messageId": "input_msg_001"
    }
  }
}
```

### 4. Conversation Continues

The agent receives the input and continues processing:

```json
{
  "jsonrpc": "2.0",
  "id": "input_001",
  "result": {
    "id": "task_123",
    "status": {"state": "completed"},
    "history": [...]
  }
}
```

## Implementation Details

### Input Detection Logic

A2A-ACP uses protocol-compliant detection in `task_manager.py`:

```python
def _is_input_required_from_response(self, response: dict) -> tuple[bool, str]:
    """Protocol-compliant detection of input-required state."""
    stop_reason = response.get("stopReason")
    tool_calls = response.get("toolCalls", [])

    if stop_reason == "end_turn" and not tool_calls:
        return True, "Agent completed turn without actions"
    return False, f"Turn ended with reason: {stop_reason}"
```

### Event Emission for Input Required

**Emitted when**: Input-required state is detected in `task_manager.py`

**Event structure**:
```json
{
  "event": "input_required",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "component": "task_manager",
  "data": {
    "prompt": "Please provide the missing data...",
    "input_types": ["text/plain"],
    "detection_method": "protocol_compliant"
  }
}
```

### State Management

Task states transition smoothly with events emitted at each step:

```
submitted →[task_created]→ working →[input_required]→ input_required →[input_provided]→ working →[task_completed]→ completed
```

### Context Preservation

All conversation context is maintained across input boundaries:

- **Message History**: Complete conversation transcript
- **Agent State**: Zed ACP session state preservation
- **Task Metadata**: All task information and settings
- **Notification Configs**: Webhook configurations persist

## Use Cases

### 1. Information Gathering

Agents can request missing information:

```bash
# Agent needs clarification
"I need to know which database table to query. Please specify: users, orders, or products?"

# User provides clarification
"Use the orders table"

# Agent continues processing
"Querying orders table for active orders..."
```

### 2. Sensitive Operations

Require confirmation for dangerous operations:

```bash
# Agent requests confirmation
"This will delete all data in production database. Continue? (yes/no)"

# User must explicitly confirm
"yes"

# Agent proceeds with operation
"Deleting production data..."
```

### 3. Multi-Step Workflows

Complex operations with multiple user touchpoints:

```bash
# Step 1: Agent analyzes requirements
"Based on your request, I need to create a new API endpoint. What HTTP method should it use?"

# Step 2: User specifies method
"POST"

# Step 3: Agent asks for more details
"What fields should this endpoint accept?"

# Step 4: User provides schema
"name (string), email (string), age (number)"

# Step 5: Agent implements solution
"Creating POST endpoint with specified schema..."
```

## Configuration

### Tool-Level Confirmation

Configure confirmation requirements per tool:

```yaml
# tools.yaml
tools:
  delete_database:
    name: "Delete Database"
    description: "Delete database contents"
    script: "dangerous_delete_command"
    sandbox:
      requires_confirmation: true
      confirmation_message: "This will delete all data. Continue?"
```

### Global Settings

Configure system-wide input handling:

```bash
# Input timeout (seconds)
export A2A_INPUT_TIMEOUT=300

# Maximum input length
export A2A_MAX_INPUT_LENGTH=10000

# Allowed input types
export A2A_ALLOWED_INPUT_TYPES="text/plain,application/json"
```

## API Methods

### Continue After Input

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/provideInputAndContinue",
  "id": "continue_001",
  "params": {
    "taskId": "task_123",
    "input": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Additional information..."}],
      "messageId": "input_001"
    },
    "metadata": {
      "input_type": "clarification"
    }
  }
}
```

### Get Input-Required Tasks

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/getInputRequired",
  "id": "list_input_001",
  "params": {}
}
```

## Event Notifications

### Input Required Event

```json
{
  "event": "input_required",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "prompt": "Please provide the missing data...",
    "input_types": ["text/plain", "application/json"],
    "context": "User needs to specify database table"
  }
}
```

### Input Provided Event

```json
{
  "event": "input_provided",
  "task_id": "task_123",
  "timestamp": "2024-01-15T10:31:00Z",
  "data": {
    "input_type": "text/plain",
    "input_length": 256,
    "resumed_execution": true
  }
}
```

## Best Practices

### For Agent Developers

1. **Clear Prompts**: Provide specific, actionable questions
2. **Context Preservation**: Include relevant context in requests
3. **Input Type Hints**: Specify expected input formats
4. **Error Handling**: Gracefully handle unclear or invalid input

### For Application Developers

1. **User Experience**: Provide clear UI for input collection
2. **State Management**: Track conversation state properly
3. **Timeout Handling**: Handle cases where users don't respond
4. **Error Recovery**: Provide ways to restart or cancel conversations

### For Operations Teams

1. **Monitoring**: Track input-required tasks and response times
2. **Timeouts**: Configure appropriate timeouts for user responses
3. **Notifications**: Set up alerts for long-pending input requests
4. **Analytics**: Monitor conversation patterns and common input requests

## Error Handling

### Timeout Handling

```python
# Configure input timeout
export A2A_INPUT_TIMEOUT=300  # 5 minutes

# Handle timeout in application
if task.status.state == "input-required":
    timeout = task.metadata.get("input_timeout", 300)
    if time_since_input_request > timeout:
        # Handle timeout - cancel or send reminder
        await cancel_task(task.id)
```

### Invalid Input

```python
# Agents should handle invalid input gracefully
try:
    # Process user input
    result = process_user_input(user_input)
except ValidationError as e:
    # Request clarification
    request_clarification(f"Invalid input: {e}. Please try again.")
```

### Conversation Recovery

```python
# If conversation state is lost, provide recovery options
if not conversation_context:
    # Offer to restart or continue from last known state
    await show_recovery_options(task.id)
```

## Monitoring

### Key Metrics

- **Input Request Rate**: How often agents request input
- **Response Time**: How long users take to respond
- **Completion Rate**: How many input-required tasks complete successfully
- **Timeout Rate**: How many input requests time out

### Dashboard Metrics

```bash
# Get input-required statistics
curl -X GET "http://localhost:8000/metrics/interactive-conversations" \
  -H "Authorization: Bearer your-token"

# Response includes:
{
  "input_requests_total": 150,
  "average_response_time": 45.2,
  "completion_rate": 94.5,
  "timeout_rate": 2.1
}
```

## Examples

### Code Review Workflow

```bash
# 1. Agent analyzes code and finds issues
"Found potential security issue in authentication. Need clarification: Is this a public API?"

# 2. User provides context
"Yes, this is a public API for customer use"

# 3. Agent continues with appropriate recommendations
"For public APIs, I recommend implementing rate limiting and API key authentication..."
```

### Database Migration

```bash
# 1. Agent plans migration but needs confirmation
"This migration will modify the users table schema. Backup first? (yes/no)"

# 2. User confirms backup requirement
"yes"

# 3. Agent proceeds with safe migration
"Creating backup... Migration starting..."
```

---

**Interactive conversations enabled!** 💬 For multi-turn workflows, see [push-notifications.md](push-notifications.md) for real-time updates.