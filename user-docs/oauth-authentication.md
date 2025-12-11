# OAuth Authentication Guide

Comprehensive guide to **OAuth authentication** in A2A-ACP for ChatGPT/OpenAI integration. This enables secure, token-based authentication for Zed ACP agents through standardized OAuth flows.

## Overview

A2A-ACP supports **OAuth 2.0 authentication** for ChatGPT/OpenAI integration, providing a secure way to authenticate agents without storing API keys directly. This feature is particularly useful for:

- **Production deployments** requiring secure credential management
- **Multi-agent environments** with different authentication requirements
- **Enterprise compliance** needing auditable authentication flows
- **Automated token refresh** for long-running applications

## Supported Authentication Methods

A2A-ACP supports multiple authentication schemes:

| Method | Description | Use Case |
|--------|-------------|----------|
| `apikey` | Direct API key authentication | Simple development setups |
| `gemini-api-key` | Google Gemini-specific API key | Gemini CLI integration |
| `codex-api-key` | OpenAI Codex-specific API key | Legacy Codex deployments |
| `openai-api-key` | OpenAI API key for ChatGPT | Standard OpenAI integration |
| `chatgpt` | **OAuth 2.0 flow** for ChatGPT | Production OAuth deployments |

## OAuth Configuration

### Environment Variables

Configure OAuth using these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `A2A_AGENT_OAUTH_ENABLED` | Enable OAuth authentication | `false` |
| `A2A_OAUTH_CLIENT_ID` | OAuth client identifier | `app_EMoamEEZ73f0CkXaXp7hrann` |
| `A2A_OAUTH_AUTHORIZE_URL` | OAuth authorization endpoint | `https://auth.openai.com/oauth/authorize` |
| `A2A_OAUTH_TOKEN_URL` | OAuth token endpoint | `https://auth.openai.com/oauth/token` |
| `A2A_OAUTH_SCOPE` | OAuth scopes | `openid profile email offline_access` |
| `A2A_OAUTH_REDIRECT` | OAuth redirect URI | `http://localhost:8001/a2a/agents/{agent_id}/auth/callback` |
| `A2A_AGENT_CODEX_HOME` | Agent codex home directory | Required for OAuth |

### Agent Configuration

Configure your agent to use OAuth:

```bash
# Enable OAuth authentication
export A2A_AGENT_OAUTH_ENABLED=true

# Set agent-specific codex home directory
export A2A_AGENT_CODEX_HOME="/var/codex/agent-codex"

# Configure OAuth client (use defaults or set custom)
export A2A_OAUTH_CLIENT_ID="your-client-id"
export A2A_OAUTH_REDIRECT="https://your-domain.com/a2a/agents/codex-acp/auth/callback"

# Agent command with codex home
export A2A_AGENT_COMMAND="/usr/local/bin/codex-acp --codex-home /var/codex/agent-codex"
export A2A_AUTH_TOKEN="your-secure-token"
```

## OAuth Flow

### 1. Start Authentication

Initiate the OAuth flow:

```bash
curl -X POST "http://localhost:8001/a2a/agents/codex-acp/auth/start" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response:**
```json
{
  "authorize_url": "https://auth.openai.com/oauth/authorize?...",
  "state": "oauth_state_123",
  "expires_at": 1710000000000
}
```

### 2. Complete Authorization

1. **Open the authorize_url** in your browser
2. **Complete the ChatGPT login** and authorization
3. **Copy the callback URL** from your browser

### 3. Handle Callback

#### Option A: Automatic Callback
If your server is publicly accessible:

```bash
# The OAuth provider will automatically redirect to:
# http://localhost:8001/a2a/agents/codex-acp/auth/callback?code=xxx&state=yyy
```

#### Option B: Manual Callback
If automatic redirect doesn't work:

```bash
curl -X POST "http://localhost:8001/a2a/agents/codex-acp/auth/manual" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "callback_url": "http://localhost:8001/a2a/agents/codex-acp/auth/callback?code=xxx&state=yyy"
  }'
```

### 4. Verify Authentication

Check authentication status:

```bash
curl "http://localhost:8001/a2a/agents/codex-acp/auth/status" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}"
```

**Response:**
```json
{
  "signed_in": true,
  "expires": 1710000000000,
  "account_id": null,
  "authentication_method": "chatgpt"
}
```

## OAuth Endpoints

### Authentication Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/a2a/agents/{agent_id}/auth/start` | POST | Start OAuth flow |
| `/a2a/agents/{agent_id}/auth/callback` | GET | OAuth callback handler |
| `/a2a/agents/{agent_id}/auth/manual` | POST | Manual callback URL processing |
| `/a2a/agents/{agent_id}/auth/status` | GET | Check authentication status |
| `/a2a/agents/{agent_id}/auth/refresh` | POST | Force token refresh |
| `/a2a/agents/{agent_id}/auth/delete` | DELETE | Remove authentication |
| `/a2a/agents/{agent_id}/auth/check-auth` | POST | Verify agent authentication |

### Request Examples

#### Start OAuth Flow
```bash
curl -X POST "http://localhost:8001/a2a/agents/codex-acp/auth/start" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "optional-custom-client-id",
    "redirect_uri": "optional-custom-redirect",
    "scope": "optional-custom-scope"
  }'
```

#### Check Status
```bash
curl "http://localhost:8001/a2a/agents/codex-acp/auth/status" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}"
```

#### Force Refresh
```bash
curl -X POST "http://localhost:8001/a2a/agents/codex-acp/auth/refresh" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}"
```

#### Remove Authentication
```bash
curl -X DELETE "http://localhost:8001/a2a/agents/codex-acp/auth" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}"
```

#### Verify Agent Authentication
```bash
curl -X POST "http://localhost:8001/a2a/agents/codex-acp/auth/check-auth" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}"
```

## Token Management

### Automatic Refresh
OAuth tokens are automatically refreshed when they expire within 5 minutes.

### Manual Refresh
Force a token refresh:

```bash
curl -X POST "http://localhost:8001/a2a/agents/codex-acp/auth/refresh" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}"
```

### Token Storage
OAuth tokens are stored in the agent's codex home directory:

```bash
# Token file location
${A2A_AGENT_CODEX_HOME}/auth.json

# File permissions (600 = owner read/write only)
chmod 600 ${A2A_AGENT_CODEX_HOME}/auth.json
```

**Token file structure:**
```json
{
  "access_token": "eyJ...",
  "refresh_token": "1//0...",
  "expires": 1710000000000,
  "scope": "openid profile email offline_access",
  "token_type": "Bearer"
}
```

## Multi-Agent OAuth

Configure OAuth for multiple agents:

```bash
# Agent 1: Codex with OAuth
export A2A_AGENT_COMMAND="/usr/local/bin/codex-acp --codex-home /var/codex/agent-1"
export A2A_AGENT_CODEX_HOME="/var/codex/agent-1"

# Agent 2: Different codex home
export A2A_AGENT_COMMAND_2="/usr/local/bin/codex-acp --codex-home /var/codex/agent-2"
export A2A_AGENT_CODEX_HOME_2="/var/codex/agent-2"
```

Each agent gets its own OAuth token file in their respective codex home directory.

## Security Best Practices

### Production Deployment

1. **Use HTTPS**: Always use HTTPS for OAuth redirect URIs in production
2. **Secure Token Storage**: Ensure codex home directories have proper permissions (600)
3. **Environment Isolation**: Use different OAuth clients for dev/staging/production
4. **Token Rotation**: Regularly rotate OAuth client secrets
5. **Audit Logging**: Monitor OAuth events for security compliance

### Environment Configuration

```bash
# Production OAuth settings
export A2A_OAUTH_CLIENT_ID="prod-client-id"
export A2A_OAUTH_REDIRECT="https://a2a.yourcompany.com/a2a/agents/codex-acp/auth/callback"
export A2A_AGENT_CODEX_HOME="/secure/codex/agent-codex"

# Ensure secure permissions
chmod 700 /secure/codex
chmod 600 /secure/codex/agent-codex
```

### Monitoring and Auditing

Monitor OAuth events:

```bash
# Check OAuth events in audit log
sqlite3 audit.db "SELECT * FROM audit_events WHERE event_type LIKE 'oauth_%' ORDER BY timestamp DESC;"

# Monitor authentication status
curl -s "http://localhost:8001/a2a/agents/codex-acp/auth/status" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}" | jq .
```

## Troubleshooting OAuth

### Common Issues

#### "OAuth flows are disabled"
```bash
# Enable OAuth
export A2A_AGENT_OAUTH_ENABLED=true
# Restart server
make run
```

#### "agent_missing_codex_home"
```bash
# Set codex home directory
export A2A_AGENT_CODEX_HOME="/var/codex/agent-codex"
# Ensure directory exists and is writable
mkdir -p /var/codex/agent-codex
chmod 755 /var/codex/agent-codex
```

#### "Invalid redirect URI"
```bash
# Verify redirect URI matches exactly
echo $A2A_OAUTH_REDIRECT
# Must match the URI registered with OpenAI
```

#### "Token expired"
```bash
# Check token status
curl "http://localhost:8001/a2a/agents/codex-acp/auth/status" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}"

# Force refresh if needed
curl -X POST "http://localhost:8001/a2a/agents/codex-acp/auth/refresh" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}"
```

### Debug OAuth Flow

Enable debug logging:

```bash
# Enable OAuth debug logging
export LOG_LEVEL=DEBUG

# Check OAuth events in logs
tail -f logs/a2a_acp.log | grep -i oauth
```

### Manual Testing

Test OAuth endpoints manually:

```bash
# 1. Start OAuth flow
RESPONSE=$(curl -s -X POST "http://localhost:8001/a2a/agents/codex-acp/auth/start" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{}')

echo $RESPONSE | jq .

# Extract authorize_url and state
AUTHORIZE_URL=$(echo $RESPONSE | jq -r '.authorize_url')
STATE=$(echo $RESPONSE | jq -r '.state')

echo "Open in browser: $AUTHORIZE_URL"
echo "State for callback: $STATE"
```

## Integration Examples

### Docker Compose with OAuth

```yaml
version: '3.8'
services:
  a2a-acp-oauth:
    build: .
    ports:
      - "8001:8000"
    environment:
      - A2A_AGENT_OAUTH_ENABLED=true
      - A2A_AGENT_COMMAND=/usr/local/bin/codex-acp --codex-home /codex/agent
      - A2A_AGENT_CODEX_HOME=/codex/agent
      - A2A_OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID}
      - A2A_OAUTH_REDIRECT=https://a2a.example.com/a2a/agents/codex-acp/auth/callback
      - A2A_AUTH_TOKEN=${A2A_AUTH_TOKEN}
      - LOG_LEVEL=INFO
    volumes:
      - codex_data:/codex/agent
    restart: unless-stopped

volumes:
  codex_data:
    driver: local
```

### CI/CD Integration

```bash
#!/bin/bash
# CI script for OAuth authentication

set -e

# Wait for A2A-ACP to be ready
until curl -s http://localhost:8001/health > /dev/null; do
  echo "Waiting for A2A-ACP..."
  sleep 5
done

# Start OAuth flow
echo "Starting OAuth flow..."
START_RESPONSE=$(curl -s -X POST "http://localhost:8001/a2a/agents/codex-acp/auth/start" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{}')

AUTHORIZE_URL=$(echo $START_RESPONSE | jq -r '.authorize_url')
echo "Please complete OAuth: $AUTHORIZE_URL"

# In CI, you might want to use service account or skip OAuth
# For production CI, consider using API key authentication instead
```

## Migration from API Key to OAuth

### Step 1: Backup Current Configuration
```bash
# Backup current environment
cp .env .env.backup

# Document current API key setup
echo "Current setup: API_KEY_AUTH" > oauth_migration.log
```

### Step 2: Configure OAuth
```bash
# Enable OAuth
export A2A_AGENT_OAUTH_ENABLED=true
export A2A_AGENT_CODEX_HOME="/var/codex/agent-codex"

# Update agent command
export A2A_AGENT_COMMAND="/usr/local/bin/codex-acp --codex-home /var/codex/agent-codex"

# Configure OAuth (use your client details)
export A2A_OAUTH_CLIENT_ID="your-oauth-client-id"
export A2A_OAUTH_REDIRECT="https://your-domain.com/a2a/agents/codex-acp/auth/callback"
```

### Step 3: Test OAuth Flow
```bash
# Test OAuth endpoints
curl -X POST "http://localhost:8001/a2a/agents/codex-acp/auth/start" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}"

# Verify authentication
curl "http://localhost:8001/a2a/agents/codex-acp/auth/status" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}"
```

### Step 4: Update Deployment
```bash
# Remove API key environment variables
unset A2A_AGENT_API_KEY

# Test agent functionality
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${A2A_AUTH_TOKEN}" \
  -d '{"jsonrpc":"2.0","method":"message/send","params":{"message":{"role":"user","parts":[{"kind":"text","text":"Hello"}]}}}'
```

## Advanced Configuration

### Custom OAuth Provider

Use a custom OAuth provider:

```bash
export A2A_OAUTH_CLIENT_ID="custom-client-id"
export A2A_OAUTH_AUTHORIZE_URL="https://custom-provider.com/oauth/authorize"
export A2A_OAUTH_TOKEN_URL="https://custom-provider.com/oauth/token"
export A2A_OAUTH_SCOPE="custom-scope1 custom-scope2"
export A2A_OAUTH_REDIRECT="https://your-domain.com/a2a/agents/custom-agent/auth/callback"
```

### OAuth with Development Tool Extension

OAuth works seamlessly with the development tool extension:

```bash
# Enable both features
export A2A_AGENT_OAUTH_ENABLED=true
export DEVELOPMENT_TOOL_EXTENSION_ENABLED=true

# Configure agent
export A2A_AGENT_COMMAND="/usr/local/bin/codex-acp --codex-home /var/codex/agent"
export A2A_AGENT_CODEX_HOME="/var/codex/agent"
```

Both authentication and extension features will be available in the agent card.

---

**OAuth authentication configured!** 🔐 For webhook integration, see [push-notifications.md](push-notifications.md). For extension features, see [development-tool extension](configuration.md#development-tool-extension-configuration).