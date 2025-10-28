# Project Overview

This project is a native A2A (Agent-to-Agent) protocol server, named A2A-ACP. It's written in Python using the FastAPI framework. Its primary purpose is to make Zed ACP agents first-class participants in enterprise coding workflows by exposing them through the A2A JSON-RPC protocol, translating requests to Zed ACP messages, managing ACP subprocesses, and coordinating the full conversation lifecycle.

The server is production-ready, with enterprise-grade security, comprehensive logging, robust error handling, and a built-in **governance pipeline**. In addition to stateful conversations, interactive workflows, push notifications, and tool execution, A2A-ACP supports auto-approval policies, programmable governors (scripted, HTTP, or Python), and detailed permission auditing so that every coding action is explainable and verifiable.

# Key Features

- **Governed Automation** – Declarative policies and programmable governors enforce guardrails while preserving auditability.
- **Permission Mediation** – Auto-resolves agent `session/request_permission` prompts (including Gemini’s `proceed_*` options) with the same policy/governor pipeline used for direct tool calls.
- **Tool Execution Sandbox** – Bash-based executor with resource limits, path controls, and rich MCP error mapping.
- **Multi-Agent Support** – Works with Codex, Claude, Gemini, and any Zed ACP-compliant agent that exposes an A2A session.
- **Push Notifications** – Real-time task state, permission decisions, and artifact alerts delivered via webhooks or SSE.

# Building and Running

The project uses `uv` for dependency management and `make` for running common tasks.

## Installation

To install the production dependencies, run:

```bash
make install
```

To install all dependencies, including development dependencies, run:

```bash
make dev-install
```

## Running the Server

Before running the server, you need to set the following environment variables:

```bash
export A2A_AGENT_COMMAND="/path/to/your/agent"
export A2A_AGENT_API_KEY="your_api_key"
export A2A_AUTH_TOKEN="your_auth_token"
```

To start the A2A-ACP server, run:

```bash
make run
```

The server will be available at `http://localhost:8001` by default. You can change the port by setting the `PORT` environment variable.

Once running, you can inspect governance decisions via:

```bash
curl -H "Authorization: Bearer <token>" \
     http://localhost:8001/a2a/tasks/<task_id>/governor/history
```

## Environment Variables

*   `A2A_AGENT_COMMAND`: The command to execute the Zed ACP agent.
*   `A2A_AGENT_API_KEY`: (Optional) The API key for the Zed ACP agent.
*   `A2A_AUTH_TOKEN`: The bearer token for authenticating with the A2A-ACP server.
*   `PORT`: (Optional) The port to run the server on. Defaults to `8001`.

### Governance Configuration

Governors and auto-approval policies are optional. Provide YAML definitions to enable automated review and audit trails:

```bash
export A2A_GOVERNORS_FILE="config/governors.yaml"
export A2A_AUTO_APPROVAL_FILE="config/auto_approval_policies.yaml"
```

- `governors.yaml` defines permission and post-run governor pipelines (script, HTTP, or Python).
- `auto_approval_policies.yaml` declares rules that approve or reject tool calls without human intervention.

If the files are absent, A2A-ACP runs with no external governors or policy overrides.

## Running Tests

To run the test suite, use the following command:

```bash
make test
```

For the governance pipeline specifically:

```bash
PYTHONPATH=src uv run python -m pytest \
  tests/test_governor_manager.py \
  tests/test_a2a_task_manager.py::TestA2ATaskManager::test_handle_tool_permission_creates_pending
```

# Development Conventions

## Code Style

The project uses `black` and `ruff` for code formatting and linting. To format the code, run:

```bash
make format
```

To lint the code, run:

```bash
make lint
```

## Type Checking

The project uses `mypy` for static type checking. To run the type checker, use:

```bash
make type
```

## Quality Checks

To run all quality checks (formatting, linting, type-checking, and tests) at once, run:

```bash
make quality
```
