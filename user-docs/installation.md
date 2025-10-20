# Installation Guide

Get A2A-ACP up and running in minutes with our streamlined installation process.

## Prerequisites

- **Python 3.9+**
- **uv package manager** (recommended) or pip
- **Zed ACP agent** (e.g., `codex-acp`, `claude-code-acp`)

## Quick Install

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/mrorigo/acp-squared.git
cd a2a-acp

# Install with uv (creates virtual environment automatically)
uv sync

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/mrorigo/acp-squared.git
cd a2a-acp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

### Environment Variables (Recommended)

Create a `.env` file or set environment variables:

```bash
# Required: Agent command
export A2A_AGENT_COMMAND="/usr/local/bin/codex-acp"

# Optional: API key for agent authentication
export A2A_AGENT_API_KEY="${OPENAI_API_KEY}"

# Optional: Agent description
export A2A_AGENT_DESCRIPTION="OpenAI Codex for A2A-ACP"

# Optional: Authentication token for A2A-ACP server
export A2A_AUTH_TOKEN="your-secret-token"
```

### Alternative: Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Application will be available at http://localhost:8000
```

## Verification

### Health Check

```bash
# Check if the server is running
curl -X GET http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2024-01-15T10:30:00Z",
#   "version": "1.0.0"
# }
```

### Agent Capabilities

```bash
# Get agent capabilities
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "agent/getAuthenticatedExtendedCard",
    "id": "card_001",
    "params": {}
  }'
```

### Test Message

```bash
# Send a test message
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "msg_001",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello A2A-ACP!"}],
        "messageId": "msg_001"
      },
      "metadata": {"agent_name": "codex-acp"}
    }
  }'
```

## What's Next?

Now that A2A-ACP is installed and running:

1. **Explore the API**: Check out [api-methods.md](api-methods.md) for complete method reference
2. **Configure Push Notifications**: See [push-notifications.md](push-notifications.md) for webhook setup
3. **Deploy to Production**: Follow [deployment.md](deployment.md) for production setup
4. **Troubleshooting**: If you run into issues, check [troubleshooting.md](troubleshooting.md)

## Getting Help

- **Configuration Issues**: See [configuration.md](configuration.md)
- **API Questions**: Check [api-methods.md](api-methods.md)
- **Deployment Help**: See [deployment.md](deployment.md)
- **Common Problems**: Check [troubleshooting.md](troubleshooting.md)

---

**Up and running!** 🎉 Next: [Quick Start Tutorial](quick-start.md)