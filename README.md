# A2A-ACP: Native A2A Protocol Server

**A2A-ACP** turns Zed ACP agents into first-class citizens of enterprise coding workflows. It exposes ACP agents (e.g., `codex-acp`, `claude-code-acp`, `gemini-cli`) through the modern **Agent-to-Agent (A2A) protocol** while embedding policy-driven governance, auto-approvals, and immutable audit trails. Platform teams gain explainable, verified automation without modifying their existing agents.

---

## 🚀 Quick Start

```bash
# 1. Install and configure
git clone https://github.com/mrorigo/a2a-acp.git
cd a2a-acp
uv sync && uv pip install -e .

# 2. Point at your ACP agent
export A2A_AGENT_COMMAND="/usr/local/bin/codex-acp"    # or "/opt/homebrew/bin/gemini --experimental-acp"
export A2A_AGENT_API_KEY="${OPENAI_API_KEY}"           # or "${GEMINI_API_KEY}"
export A2A_AUTH_TOKEN="your-secret-token"             # optional

# 3. Launch the gateway
make run

# 4. Send a request
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "req_001",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello A2A!"}],
        "messageId": "msg_123"
      },
      "metadata": {"agent_name": "codex-acp"}
    }
  }'
```

> 💡 Enable the governance pipeline by providing `A2A_GOVERNORS_FILE` and `A2A_AUTO_APPROVAL_FILE` (see [Configuration Guide](user-docs/configuration.md)).

---

## 📚 Documentation

- **[Quick Start](user-docs/quick-start.md)** – Five-minute setup walkthrough  
- **[API Reference](user-docs/api-methods.md)** – JSON-RPC and REST endpoints  
- **[Configuration Guide](user-docs/configuration.md)** – Environment variables, policies, governors  
- **[Governance Events](user-docs/events.md)** – Event payloads for audit and compliance  
- **[Testing Guide](user-docs/testing.md)** – Running the unit and governance suites

For a strategic overview, read the [Enterprise Whitepaper](WHITEPAPER.md).

---

## ✨ Key Capabilities

- **🛡️ Enterprise Governance** – Auto-approval policies, programmable governors, and audit endpoints (`/a2a/tasks/{id}/governor/history`)
- **🔗 Protocol Bridge** – Full A2A v0.3.0 implementation with seamless Zed ACP translation
- **🧰 Tool Execution** – Direct support for Codex `fs/read_text_file`, `fs/write_text_file`, `shell`, plus custom bash tools
- **💬 Interactive Conversations** – Native handling of input-required workflows and multi-turn tasks
- **📡 Push Notifications** – Webhook delivery with filtering, quiet hours, analytics, and retries
- **📈 Observability** – Structured logging, health checks, and metrics endpoints for production monitoring

---

## 🤝 Supported Agents

- **`codex-acp`** (OpenAI)  
- **`claude-code-acp`** (Anthropic)  
- **`gemini-cli`** (Google)  
- **Any Zed ACP-compliant agent** with session persistence

---

## 📈 Project Status

### ✅ Current Capabilities
- **Governed Execution** – Declarative policies, scripted/Python/HTTP governors, and follow-up prompts
- **A2A Compliance** – Full v0.3.0 coverage, including tasks, contexts, artifacts, and streaming
- **Enterprise Security** – Auth tokens, quota controls, and immutable audit logging
- **Production Tooling** – 240+ tests, SSE streaming, webhook analytics, and bash-based tool execution

### 🔭 On the Horizon
- Integrations with external policy engines (e.g., Rego/CEL)
- Deeper sandboxing for governor execution environments
- Expanded agent marketplace tooling and deployment recipes

---

Start with the [Quick Start](user-docs/quick-start.md), wire in your policies, and let governed ACP agents collaborate with confidence. For questions or contributions, see [CONTRIBUTING.md](CONTRIBUTING.md).
