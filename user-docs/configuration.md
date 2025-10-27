# Configuration Guide

Configure A2A-ACP using environment variables for simple, secure, and flexible setup.

## Core Configuration

### Required Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `A2A_AGENT_COMMAND` | Path to Zed ACP agent binary | `/usr/local/bin/codex-acp` |
| `A2A_AUTH_TOKEN` | Authentication token for A2A-ACP | `your-secret-token` |

### Agent Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `A2A_AGENT_API_KEY` | API key for agent authentication | `${OPENAI_API_KEY}` or `${GEMINI_API_KEY}` |
| `A2A_AGENT_DESCRIPTION` | Human-readable agent description | `OpenAI Codex for A2A-ACP` or `Gemini CLI for A2A-ACP` |

### Optional Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Supported Agents

A2A-ACP supports multiple Zed ACP-compliant agents:

| Agent | Authentication Method | Environment Variable | Description |
|-------|----------------------|---------------------|-------------|
| `codex-acp` | `apikey` | `${OPENAI_API_KEY}` | OpenAI Codex agent |
| `claude-code-acp` | `apikey` | `${ANTHROPIC_API_KEY}` | Anthropic Claude agent |
| `gemini-cli` | `gemini-api-key` | `${GEMINI_API_KEY}` | Google Gemini agent |

The server automatically detects the authentication method based on the agent's capabilities and uses the appropriate API key.

## Environment File Setup

### 1. Copy Template

```bash
cp .env.example .env
```

### 2. Edit Configuration

```bash
# Open .env and configure for OpenAI Codex:
A2A_AGENT_COMMAND="/usr/local/bin/codex-acp"
A2A_AGENT_API_KEY="${OPENAI_API_KEY}"
A2A_AGENT_DESCRIPTION="OpenAI Codex for A2A-ACP"
A2A_AUTH_TOKEN="your-secure-secret-token"

# Or for Gemini CLI:
A2A_AGENT_COMMAND="/opt/homebrew/bin/gemini --experimental-acp"
A2A_AGENT_API_KEY="${GEMINI_API_KEY}"
A2A_AGENT_DESCRIPTION="Gemini CLI for A2A-ACP"
A2A_AUTH_TOKEN="your-secure-secret-token"
```

### 3. Load Environment

```bash
# Load the environment file
source .env

# Or use a tool like direnv:
# echo 'source .env' >> .envrc
# direnv allow
```

## Docker Configuration

### Docker Compose (Recommended)

```yaml
version: '3.8'

services:
  a2a-acp:
    build: .
    ports:
      - "8000:8000"
    environment:
      - A2A_AGENT_COMMAND=/usr/local/bin/codex-acp
      - A2A_AGENT_API_KEY=${OPENAI_API_KEY}
      - A2A_AUTH_TOKEN=${A2A_AUTH_TOKEN}
      - LOG_LEVEL=INFO
    restart: unless-stopped

  # Alternative configuration for Gemini CLI
  # a2a-acp-gemini:
  #   build: .
  #   ports:
  #     - "8000:8000"
  #   environment:
  #     - A2A_AGENT_COMMAND=/opt/homebrew/bin/gemini --experimental-acp
  #     - A2A_AGENT_API_KEY=${GEMINI_API_KEY}
  #     - A2A_AUTH_TOKEN=${A2A_AUTH_TOKEN}
  #     - LOG_LEVEL=INFO
  #   restart: unless-stopped
```

### Docker Run

```bash
# For OpenAI Codex
docker run -d \
  --name a2a-acp-codex \
  -p 8000:8000 \
  -e A2A_AGENT_COMMAND="/usr/local/bin/codex-acp" \
  -e A2A_AGENT_API_KEY="${OPENAI_API_KEY}" \
  -e A2A_AUTH_TOKEN="your-secret-token" \
  -e LOG_LEVEL=INFO \
  your-registry/a2a-acp:latest

# For Gemini CLI
docker run -d \
  --name a2a-acp-gemini \
  -p 8001:8000 \
  -e A2A_AGENT_COMMAND="/opt/homebrew/bin/gemini --experimental-acp" \
  -e A2A_AGENT_API_KEY="${GEMINI_API_KEY}" \
  -e A2A_AUTH_TOKEN="your-secret-token" \
  -e LOG_LEVEL=INFO \
  your-registry/a2a-acp:latest
```

## Push Notifications

Configure push notifications for real-time task monitoring:

```bash
# Enable push notifications
export PUSH_NOTIFICATIONS_ENABLED=true

# Webhook timeout (seconds)
export PUSH_NOTIFICATION_WEBHOOK_TIMEOUT=30

# Retry attempts for failed webhooks
export PUSH_NOTIFICATION_RETRY_ATTEMPTS=3

# HMAC secret for webhook signature verification
export PUSH_NOTIFICATION_HMAC_SECRET="your-hmac-secret"
```

## Advanced Configuration


### Database Settings

```bash
# Database file location
export DATABASE_URL="sqlite:///data/a2a_acp.db"

# Connection pool settings
export DB_POOL_SIZE=10
export DB_MAX_OVERFLOW=20
```

### Performance Tuning

```bash
# Task cleanup interval (seconds)
export TASK_CLEANUP_INTERVAL=3600

# Maximum concurrent tasks
export MAX_CONCURRENT_TASKS=100

# Task timeout (seconds)
export TASK_TIMEOUT=300
```

### Security Settings

```bash
# Maximum request size (MB)
export MAX_REQUEST_SIZE_MB=10

# CORS origins (comma-separated)
export CORS_ORIGINS="https://your-domain.com,https://app.your-domain.com"
```

### Governance Settings

Configure policy-driven auto-approvals and governor pipelines through YAML files. Both files are optional; if omitted no automatic decisions or external governors are executed.

```bash
# Override default locations
export A2A_GOVERNORS_FILE="config/governors.yaml"
export A2A_AUTO_APPROVAL_FILE="config/auto_approval_policies.yaml"
```

#### `governors.yaml`

```yaml
permission_governors:
  - id: security-diff-check
    type: script
    command: ["python3", "governors/security.py"]
    timeout_ms: 5000

output_governors:
  - id: code-reviewer
    type: http
    url: https://governor.example.com/review
    headers:
      Authorization: Bearer ${GOVERNOR_TOKEN}

permission_settings:
  stop_on_first_reject: true
  auto_decision: all_approve

output_settings:
  max_iterations: 3
```

#### `auto_approval_policies.yaml`

```yaml
auto_approval_policies:
  - id: docs-edits
    applies_to: ["functions.acp_fs__write_text_file"]
    include_paths: ["docs/**", "*.md"]
    decision:
      type: approve
      optionId: approved

  - id: safe-shell
    applies_to: ["functions.shell"]
    parameters:
      command_prefix: ["git", "status"]
    decision:
      type: approve
      optionId: approved
```

## Runtime Configuration

### Health Check Endpoint

```bash
# Check system health
curl -X GET "http://localhost:8000/health" \
  -H "Authorization: Bearer your-token"

# Response includes:
# - Overall system status
# - Component health (database, push notifications)
# - Version information
```

### Metrics Endpoints

```bash
# Get push notification metrics
curl -X GET "http://localhost:8000/metrics/push-notifications" \
  -H "Authorization: Bearer your-token"

# Get system metrics
curl -X GET "http://localhost:8000/metrics/system" \
  -H "Authorization: Bearer your-token"
```

## Configuration Validation

### Validate Configuration

```bash
# Check if configuration is valid
python -c "
import os
from src.a2a_acp.settings import Settings

try:
    settings = Settings()
    print('✅ Configuration is valid')
    print(f'Agent: {settings.agent_description}')
    print(f'Auth: {"Enabled" if settings.auth_token else "Disabled"}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

### Environment Check

```bash
# Verify all required environment variables are set
python -c "
import os
required = ['A2A_AGENT_COMMAND', 'A2A_AUTH_TOKEN']
missing = [var for var in required if not os.getenv(var)]
if missing:
    print(f'❌ Missing required variables: {missing}')
else:
    print('✅ All required variables are set')
"
```

## Troubleshooting Configuration

### Common Issues

**"Agent command not found"**
```bash
# Check if agent binary exists and is executable
ls -la $(which codex-acp)
# or
which codex-acp
```

**"Authentication failed"**
```bash
# Verify auth token is set correctly
echo "Auth token: $A2A_AUTH_TOKEN"

# Test health endpoint with authentication
curl -H "Authorization: Bearer $A2A_AUTH_TOKEN" \
     http://localhost:8000/health
```

**"Database connection failed"**
```bash
# Check database file permissions
ls -la data/a2a_acp.db

# Verify SQLite is available
sqlite3 --version
```

## Security Best Practices

### Token Management
- Use strong, randomly generated tokens
- Rotate tokens regularly (monthly recommended)
- Use different tokens for different environments

### Environment Variables
- Never commit `.env` files to version control
- Use `.env.example` as a template
- Consider using secret management systems (Vault, etc.)

### Network Security
- Run behind reverse proxy with TLS termination
- Use internal network for inter-service communication
- Implement rate limiting and DDoS protection

---

**Configuration complete!** 🔧 Next: [Quick Start Tutorial](quick-start.md)
