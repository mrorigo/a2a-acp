# Development Tool Extension Guide

Complete guide to **A2A-ACP's Development Tool Extension** - a powerful feature that enables structured tool interactions, slash commands, tool lifecycles, and enhanced agent communication for development workflows.

## Overview

The **Development Tool Extension** (v1.0.0) transforms A2A-ACP into a full-featured development agent platform by adding:

- **🔧 Slash Commands**: Auto-discovered bash tools as executable commands
- **📊 Tool Lifecycles**: Real-time progress tracking and status updates  
- **✅ User Confirmations**: Structured permission workflows for sensitive operations
- **💭 Agent Thoughts**: Reasoning transparency during tool execution
- **🔄 Interactive Workflows**: Seamless human-in-the-loop collaboration

This extension is particularly valuable for:
- **Development IDEs** (VS Code, JetBrains) wanting native tool integration
- **CLI Tools** (Gemini CLI) requiring structured tool execution
- **Enterprise Development** workflows needing audit trails and approvals
- **Agent Transparency** where users need to understand agent reasoning

## Quick Start

### 1. Enable the Extension

```bash
# Enable the development tool extension
export DEVELOPMENT_TOOL_EXTENSION_ENABLED=true

# Restart A2A-ACP
make run
```

### 2. Configure Tools

Create or update your `tools.yaml` file:

```yaml
# tools.yaml
tools:
  git_status:
    name: "Git Status"
    description: "Check git repository status"
    script: |
      #!/bin/bash
      git status --porcelain
    parameters: []
    sandbox:
      requires_confirmation: false
      timeout: 10

  database_query:
    name: "Database Query"
    description: "Execute SQL queries safely"
    script: |
      #!/bin/bash
      psql -d {{database}} -c "{{query}}"
    parameters:
      - name: database
        type: string
        required: true
      - name: query
        type: string
        required: true
    sandbox:
      requires_confirmation: true
      confirmation_message: "Execute SQL query? This may modify data."
      timeout: 30
```

### 3. Discover Commands

```bash
# Get all available slash commands
curl -X GET "http://localhost:8001/a2a/commands/get" \
  -H "Authorization: Bearer your-token"
```

**Response:**
```json
{
  "commands": [
    {
      "name": "/git_status",
      "description": "Check git repository status",
      "arguments": []
    },
    {
      "name": "/database_query", 
      "description": "Execute SQL queries safely",
      "arguments": [
        {
          "name": "database",
          "type": "string",
          "required": true
        },
        {
          "name": "query",
          "type": "string", 
          "required": true
        }
      ]
    }
  ]
}
```

### 4. Execute Commands

```bash
# Execute a simple command
curl -X POST "http://localhost:8001/a2a/command/execute" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "command": "git_status",
    "arguments": {}
  }'

# Response includes execution_id for tracking
{
  "execution_id": "task_123",
  "status": "executing"
}
```

## Extension Features

### Slash Commands

Slash commands are **automatically discovered** from your bash tools:

| Tool Configuration | Slash Command | Usage |
|-------------------|---------------|--------|
| `name: "Git Status"` | `/git_status` | `/git_status` |
| `name: "HTTP Request"` | `/http_request` | `/http_request method="GET" url="..."` |
| `name: "Create File"` | `/create_file` | `/create_file path="file.txt" content="..."` |

**Benefits:**
- **Auto-discovery**: No hardcoded command lists
- **Rich Metadata**: Parameter types, descriptions, validation
- **Consistent UX**: Standardized command interface
- **Tool Validation**: Built-in parameter checking

### Tool Lifecycles

Tools follow a structured lifecycle with real-time updates:

#### 1. **PENDING** - Awaiting Confirmation
```json
{
  "kind": "tool_call_update",
  "tool_call": {
    "tool_call_id": "tc_001",
    "status": "pending",
    "tool_name": "database_query",
    "input_parameters": {
      "database": "production",
      "query": "DELETE FROM users WHERE id = 123"
    },
    "confirmation_request": {
      "options": [
        {"id": "approve", "name": "Allow"},
        {"id": "deny", "name": "Reject"}
      ],
      "details": {
        "description": "Dangerous SQL operation detected"
      }
    }
  }
}
```

#### 2. **EXECUTING** - Running in Sandbox
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

#### 3. **SUCCEEDED** - Completed Successfully
```json
{
  "kind": "tool_call_update",
  "tool_call": {
    "tool_call_id": "tc_001", 
    "status": "succeeded",
    "result": {
      "content": "Query executed successfully",
      "details": {
        "stdout": "10 rows affected",
        "exit_code": 0
      }
    }
  }
}
```

#### 4. **FAILED** - Error Occurred
```json
{
  "kind": "tool_call_update",
  "tool_call": {
    "tool_call_id": "tc_001",
    "status": "failed", 
    "result": {
      "message": "Connection refused",
      "code": "DB_CONNECTION_ERROR",
      "details": {
        "stderr": "psql: could not connect to server"
      }
    }
  }
}
```

### User Confirmations

Sensitive tools can request user approval before execution:

```yaml
# tools.yaml configuration
tools:
  deploy_application:
    name: "Deploy Application"
    description: "Deploy to production environment"
    script: |
      #!/bin/bash
      echo "Deploying to production..."
      ./scripts/deploy.sh {{environment}}
    parameters:
      - name: environment
        type: string
        required: true
        enum: ["staging", "production"]
    sandbox:
      requires_confirmation: true
      confirmation_message: "Deploy to production? This action cannot be undone."
      timeout: 300
```

**Confirmation Flow:**
1. Tool execution paused at PENDING state
2. User receives structured confirmation request
3. User approves/denies via API or UI
4. Execution continues or is cancelled

### Agent Thoughts

Agents can share reasoning steps for transparency:

```json
{
  "kind": "thought",
  "content": "Analyzing query for potential data loss before execution...",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Use Cases:**
- **Security Analysis**: "Checking file permissions before modification..."
- **Risk Assessment**: "Evaluating query impact on production database..."
- **Progress Updates**: "Processing large dataset, 50% complete..."
- **Decision Making**: "Choosing optimal algorithm based on input size..."

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEVELOPMENT_TOOL_EXTENSION_ENABLED` | Enable extension features | `true` |
| `A2A_EXTENSION_METADATA_ENABLED` | Include metadata in events | `true` |

### Tool Configuration

Configure tools in `tools.yaml`:

```yaml
# Basic tool (no confirmation)
simple_tool:
  name: "Simple Tool"
  description: "A basic tool with no confirmation"
  script: |
    #!/bin/bash
    echo "Tool executed successfully"
  parameters: []
  sandbox:
    requires_confirmation: false
    timeout: 30

# Confirmation-required tool
sensitive_tool:
  name: "Sensitive Operation" 
  description: "A tool requiring user approval"
  script: |
    #!/bin/bash
    echo "Executing sensitive operation: {{operation}}"
  parameters:
    - name: operation
      type: string
      required: true
  sandbox:
    requires_confirmation: true
    confirmation_message: "Execute sensitive operation?"
    timeout: 60
```

### Extension Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/a2a/commands/get` | GET | Discover available slash commands |
| `/a2a/command/execute` | POST | Execute a slash command |

## Integration Examples

### VS Code Extension

```typescript
// Discover commands
const commands = await fetch('/a2a/commands/get', {
  headers: { 'Authorization': `Bearer ${token}` }
});
const commandList = await commands.json();

// Execute command with progress tracking
const execution = await fetch('/a2a/command/execute', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    command: 'git_status',
    arguments: {}
  })
});

const { execution_id } = await execution.json();

// Stream results with extension metadata
const stream = new EventSource(`/a2a/message/stream?taskId=${execution_id}`, {
  headers: { 'Authorization': `Bearer ${token}` }
});

stream.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  // Handle extension metadata
  if (data.metadata?.['https://developers.google.com/gemini/a2a/extensions/development-tool/v1']) {
    const extData = data.metadata['https://developers.google.com/gemini/a2a/extensions/development-tool/v1'];
    
    switch (extData.kind) {
      case 'tool_call_update':
        handleToolLifecycle(extData.tool_call);
        break;
      case 'thought':
        showAgentThought(extData.content);
        break;
    }
  }
};
```

### Gemini CLI Integration

```bash
#!/bin/bash
# Auto-discover and execute commands

# Get available commands
commands=$(curl -s -H "Authorization: Bearer $A2A_TOKEN" \
  "http://localhost:8001/a2a/commands/get")

# Parse commands and create aliases
echo "$commands" | jq -r '.commands[] | "alias \(.name)=\"a2a-exec \(.name)\""' >> ~/.bashrc

# Function to execute commands
a2a-exec() {
  local cmd=$1
  shift
  
  # Convert arguments to JSON
  local args=$(printf '%s\n' "$@" | jq -R . | jq -s 'split("\n")[:-1] | map(split("=") | {name: .[0], value: .[1]}) | from_entries')
  
  # Execute via extension endpoint
  curl -s -X POST "http://localhost:8001/a2a/command/execute" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $A2A_TOKEN" \
    -d "{\"command\": \"${cmd#\/}\", \"arguments\": $args}"
}
```

### Webhook Integration

Receive extension events via push notifications:

```javascript
// Configure push notifications with extension events
const pushConfig = {
  enabledEvents: [
    "status_change",
    "input_required", 
    "tool_call_update",
    "thought"
  ]
};

// Webhook handler
app.post('/webhooks/a2a', (req, res) => {
  const event = req.body;
  
  // Handle extension events
  if (event.data?.metadata?.['https://developers.google.com/gemini/a2a/extensions/development-tool/v1']) {
    const extData = event.data.metadata['https://developers.google.com/gemini/a2a/extensions/development-tool/v1'];
    
    switch (extData.kind) {
      case 'tool_call_update':
        console.log(`Tool ${extData.tool_call.tool_name}: ${extData.tool_call.status}`);
        break;
      case 'thought':
        console.log(`Agent thought: ${extData.content}`);
        break;
    }
  }
  
  res.status(200).send('OK');
});
```

## Advanced Usage

### Custom Tool Types

Extend tool capabilities with custom types:

```yaml
tools:
  file_diff:
    name: "File Diff Analyzer"
    description: "Analyze file differences"
    script: |
      #!/bin/bash
      diff {{old_file}} {{new_file}} | head -20
    parameters:
      - name: old_file
        type: string
        required: true
      - name: new_file  
        type: string
        required: true
    sandbox:
      requires_confirmation: false
      timeout: 15
      allowed_paths: ["/workspace/**", "/tmp/**"]
```

### Governance Integration

Combine with governance policies:

```yaml
# auto_approval_policies.yaml
auto_approval_policies:
  - id: safe-git-commands
    applies_to: ["git_status", "git_log"] 
    decision:
      type: approve
      optionId: allow
      reason: "Safe read-only git operations"
      
  - id: production-deployments
    applies_to: ["deploy_application"]
    decision:
      type: require_approval
      reason: "Production deployments require human review"
```

### Multi-Agent Extension Support

```bash
# Agent 1: Development tools
export A2A_AGENT_COMMAND="codex-acp --codex-home /codex/dev"
export A2A_TOOLS_CONFIG="/config/dev-tools.yaml"

# Agent 2: Operations tools  
export A2A_AGENT_COMMAND_2="gemini-cli --codex-home /codex/ops"
export A2A_TOOLS_CONFIG_2="/config/ops-tools.yaml"
```

## Troubleshooting

### Extension Not Working

**Check if extension is enabled:**
```bash
curl "http://localhost:8001/.well-known/agent-card.json" | \
  jq '.capabilities.extensions[] | select(.uri | contains("development-tool"))'
```

**Enable extension:**
```bash
export DEVELOPMENT_TOOL_EXTENSION_ENABLED=true
make run
```

### Commands Not Discovered

**Verify tools.yaml exists and is valid:**
```bash
# Check file exists
ls -la tools.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('tools.yaml'))"

# Test command discovery
curl -H "Authorization: Bearer $A2A_TOKEN" \
  "http://localhost:8001/a2a/commands/get"
```

### No Extension Metadata in Events

**Check extension metadata setting:**
```bash
export A2A_EXTENSION_METADATA_ENABLED=true

# Test with verbose logging
export LOG_LEVEL=DEBUG
```

### Tool Execution Failures

**Check tool validation:**
```bash
# Validate tool parameters
curl -X POST "http://localhost:8001/a2a/command/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "invalid_tool",
    "arguments": {}
  }'
```

**Review tool execution logs:**
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG

# Monitor tool execution
tail -f logs/a2a_acp.log | grep -E "(tool|extension|command)"
```

## Migration from Legacy Tools

### Before (Legacy Approach)
```bash
# Direct message with tool execution
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Run git status"}]
      }
    }
  }'
```

### After (Extension Approach)
```bash
# Structured command execution
curl -X POST "http://localhost:8001/a2a/command/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "git_status",
    "arguments": {}
  }'
```

**Benefits of Migration:**
- **Better UX**: Discoverable commands with validation
- **Rich Feedback**: Real-time progress and status updates
- **Structured Data**: JSON-native tool outputs
- **Enhanced Security**: Permission workflows and governance

## Best Practices

### Tool Design
1. **Clear Names**: Use descriptive tool names that become intuitive commands
2. **Parameter Validation**: Define parameter types and validation rules
3. **Confirmation Strategy**: Enable confirmations for sensitive operations
4. **Timeout Management**: Set appropriate timeouts for tool execution
5. **Error Handling**: Provide meaningful error messages and recovery suggestions

### User Experience
1. **Rich Descriptions**: Write clear tool descriptions for command discovery
2. **Progress Updates**: Emit thoughts during long-running operations
3. **Permission Flows**: Design intuitive confirmation messages
4. **Status Communication**: Use appropriate lifecycle states

### Security
1. **Path Restrictions**: Use sandbox path restrictions for file tools
2. **Command Whitelisting**: Limit available commands in production
3. **Parameter Sanitization**: Validate and sanitize all parameters
4. **Audit Trails**: Enable comprehensive logging for compliance

### Performance
1. **Efficient Scripts**: Write optimized bash scripts for tools
2. **Timeout Configuration**: Set realistic timeouts to prevent hanging
3. **Resource Limits**: Configure sandbox resource constraints
4. **Caching**: Consider caching for expensive operations

---

**Development Tool Extension configured!** 🚀 For technical implementation details, see [DEVELOPMENT_TOOL_EXTENSION.md](../docs/DEVELOPMENT_TOOL_EXTENSION.md). For configuration options, see [configuration.md](configuration.md#development-tool-extension-configuration).