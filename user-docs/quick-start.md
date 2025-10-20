# Quick Start Tutorial

Get A2A-ACP running and send your first message in under 5 minutes.

## 1. Prerequisites Check

Ensure you have:
- ✅ Python 3.9+
- ✅ Zed ACP agent installed (e.g., `codex-acp`)
- ✅ API key for your agent

## 2. Installation (1 minute)

```bash
# Clone and install
git clone https://github.com/mrorigo/acp-squared.git
cd a2a-acp
uv sync && uv pip install -e .

# Configure your agent
export A2A_AGENT_COMMAND="/usr/local/bin/codex-acp"
export A2A_AGENT_API_KEY="${OPENAI_API_KEY}"
export A2A_AUTH_TOKEN="your-secret-token"
```

## 3. Start the Server (30 seconds)

```bash
# Start A2A-ACP server
make run

# Server should be running at http://localhost:8001
```

## 4. Test the Connection

### Health Check
```bash
curl -X GET http://localhost:8000/health
# Should return: {"status": "healthy", "version": "1.0.0"}
```

### Get Agent Capabilities
```bash
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "agent/getAuthenticatedExtendedCard",
    "id": "test_001",
    "params": {}
  }'
```

## 5. Send Your First Message

```bash
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "hello_001",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello! Can you help me with a Python script?"}],
        "messageId": "msg_001"
      },
      "metadata": {"agent_name": "codex-acp"}
    }
  }'
```

**Expected Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "hello_001",
  "result": {
    "id": "task_123",
    "contextId": "ctx_456",
    "status": {"state": "completed"},
    "history": [...]
  }
}
```

## 6. Interactive Conversation

A2A-ACP supports stateful conversations with context persistence:

```bash
# First message - creates context
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "ctx_001",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Remember: My name is Alice"}],
        "messageId": "msg_001"
      },
      "metadata": {"agent_name": "codex-acp"}
    }
  }'

# Continue conversation - context preserved!
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "ctx_002",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "What is my name?"}],
        "messageId": "msg_002",
        "contextId": "ctx_456"
      },
      "metadata": {"agent_name": "codex-acp"}
    }
  }'
```

## 7. Set Up Push Notifications (Optional)

Get real-time updates on task events:

```bash
# Configure webhook
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks/pushNotificationConfig/set",
    "id": "webhook_001",
    "params": {
      "taskId": "task_123",
      "config": {
        "id": "my_webhook",
        "url": "https://your-app.com/webhooks/a2a",
        "token": "your-bearer-token",
        "enabledEvents": ["status_change", "message", "input_required"]
      }
    }
  }'
```

## 8. What's Next?

Now that you're up and running:

### 🛠️ Customize Your Setup
- **[Configuration Guide](configuration.md)** - Fine-tune your setup
- **[Tool Execution](tool-execution.md)** - Enable bash-based tool execution
- **[Push Notifications](push-notifications.md)** - Set up real-time notifications

### 🚀 Deploy to Production
- **[Deployment Guide](deployment.md)** - Production deployment instructions
- **[Docker Deployment](docker-deployment.md)** - Containerized deployment
- **[Monitoring](monitoring.md)** - Health checks and observability

### 💻 Build Applications
- **[API Methods](api-methods.md)** - Complete API reference
- **[Interactive Conversations](interactive-conversations.md)** - Handle input-required workflows
- **[Examples](examples/)** - Sample applications and integrations

## 🔒 Security Features

A2A-ACP includes **enterprise-grade security** that works automatically:

### Automatic Protection
Every tool execution is protected by **hard OS-level limits**:
- **512MB memory limit** (prevents memory exhaustion)
- **30-second timeout** (prevents runaway processes)
- **10MB file size limit** (prevents disk space abuse)

### Sandbox Security
- **Command allowlisting** - Tools can only run approved commands
- **Working directory isolation** - Each tool runs in its own sandbox
- **Environment isolation** - Secure API key and credential injection
- **Comprehensive audit logging** - All executions tracked and logged

**These security measures work automatically** - no configuration needed!

## Troubleshooting

**"Agent not found"**
```bash
# Check agent path
which codex-acp
# Verify it's in your PATH or use full path in A2A_AGENT_COMMAND
```

**"Authentication failed"**
```bash
# Check auth token
echo $A2A_AUTH_TOKEN
# Ensure it's set and matches what your client sends
```

**"Connection refused"**
```bash
# Check if server is running
curl http://localhost:8000/health
# Verify port 8000 is not blocked by firewall
```

## Getting Help

- **Configuration Issues**: See [configuration.md](configuration.md)
- **API Questions**: Check [api-methods.md](api-methods.md)
- **Common Problems**: See [troubleshooting.md](troubleshooting.md)
- **Full Documentation**: Return to [index.md](index.md)

---

**Congratulations!** 🎉 You're now running A2A-ACP with a Zed ACP agent. Check out the [examples](examples/) folder for sample applications you can build on top of this!