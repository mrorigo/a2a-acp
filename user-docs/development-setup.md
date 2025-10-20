# Development Setup

Set up a local development environment for A2A-ACP development and testing.

## Prerequisites

- **Python 3.9+**
- **uv package manager** (recommended)
- **Git**
- **Make** (for build automation)

## Quick Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/mrorigo/acp-squared.git
cd a2a-acp

# Install in development mode
uv sync
uv pip install -e ".[dev]"
```

### 2. Configure Development Environment

```bash
# Copy development environment file
cp .env.example .env

# Edit for development
# A2A_AGENT_COMMAND="python tests/dummy_agent.py"
# A2A_AGENT_DESCRIPTION="Development agent"
# A2A_AUTH_TOKEN="dev-token"
# LOG_LEVEL=DEBUG
```

### 3. Verify Installation

```bash
# Check installation
python -c "import src.a2a_acp.main; print('✅ A2A-ACP imported successfully')"

# Run health check using Makefile
make test

# Or run development server to verify
make run
```

## Development Tools

### Code Quality

The project uses **Black**, **Ruff**, and **MyPy** for code quality. Use the Makefile for consistent tooling:

```bash
# Format code (Black + Ruff)
make format

# Lint code (Ruff)
make lint

# Type checking (MyPy)
make type

# All quality checks (format + lint + type + test)
make quality
```

### Testing

**240+ comprehensive tests** covering unit, integration, protocol, and performance testing:

```bash
# Run all tests (with timeout and quiet output)
make test

# Run with coverage report
make test-coverage

# Run specific test file with verbose output
python -m pytest tests/test_a2a_task_manager.py -v

# Run tests in watch mode during development
python -m pytest tests/ -v --tb=short -x

# Debug specific test with full output
python -m pytest tests/test_a2a_task_manager.py::TestA2ATaskManager::test_create_task_basic -v -s --tb=long
```

### Development Server

```bash
# Run development server (using Makefile)
make run

# Or run manually with debugging
A2A_LOG_LEVEL=DEBUG \
A2A_AGENT_COMMAND="python tests/dummy_agent.py" \
python -m uvicorn src.a2a_acp.main:create_app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

### Source Code

```
src/
├── a2a/                    # A2A Protocol Implementation
│   ├── server.py          # JSON-RPC 2.0 HTTP server
│   ├── models.py          # A2A type definitions (Task, Message, etc.)
│   ├── translator.py      # A2A ↔ Zed ACP translation
│   ├── agent_card.py      # Dynamic AgentCard generation
│   └── agent_manager.py   # Zed ACP agent lifecycle management
└── a2a_acp/               # A2A-ACP Application
    ├── main.py            # FastAPI application entry point
    ├── task_manager.py    # A2A task lifecycle management
    ├── zed_agent.py       # Zed ACP subprocess management
    ├── tool_config.py     # Tool configuration and validation
    ├── bash_executor.py   # Bash script execution engine
    ├── sandbox.py         # Execution environment management
    ├── audit.py           # Security audit logging
    └── push_notification_manager.py # HTTP webhook notifications
```

### Test Structure

```
tests/
├── test_a2a_acp_bridge.py    # Core A2A-ACP functionality tests (40+ tests)
├── test_a2a_server.py        # A2A protocol implementation tests (16+ tests)
├── test_a2a_task_manager.py  # Task management tests (50+ tests)
├── test_tool_config.py       # Tool configuration tests (20+ tests)
├── test_tool_protocol_compliance.py # Protocol compliance tests (15+ tests)
├── test_tool_system_integration.py  # End-to-end integration tests (10+ tests)
├── dummy_agent.py            # Test Zed ACP agent for development
└── run_tool_tests.py         # Comprehensive test runner
```

## Development Workflow

### 1. Feature Development

```bash
# 1. Create feature branch
git checkout -b feature/new-tool-type

# 2. Implement feature with tests
# Edit src/a2a_acp/tool_config.py
# Add tests in tests/test_tool_config.py

# 3. Run tests
make test

# 4. Format and lint
make format
make lint

# 5. Commit changes
git add .
git commit -m "Add new tool type support"
```

### 2. Testing Workflow

```bash
# Run specific test category
python -m pytest tests/test_tool_config.py -v

# Run with debugging
python -m pytest tests/test_a2a_task_manager.py::TestA2ATaskManager::test_create_task_basic -v -s

# Run performance tests
python -m pytest tests/test_push_performance.py -v
```

### 3. Documentation Updates

```bash
# Update documentation for new features
# Edit relevant .md files in user-docs/
# Add examples and API documentation
# Update index.md if adding new documentation files
```

## Debugging Setup

### Enhanced Logging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export A2A_DEBUG_EVENTS=true
export A2A_DEBUG_PROTOCOL=true

# View detailed logs
tail -f logs/a2a_acp.log | grep -E "(DEBUG|ERROR|tool|event)"
```

### Database Inspection

```bash
# Check database contents during development
sqlite3 data/a2a_acp.db "SELECT id, status, created_at FROM tasks ORDER BY created_at DESC LIMIT 5;"

# View recent messages
sqlite3 data/a2a_acp.db "SELECT task_id, role, message_id FROM messages ORDER BY created_at DESC LIMIT 5;"
```

### Network Debugging

```bash
# Monitor network traffic
sudo tcpdump -i lo -A -s 0 port 8000

# Check port availability
lsof -i :8000
netstat -tlnp | grep 8000
```

## Development Best Practices

### Code Organization

1. **Single Responsibility**: Each module has one clear purpose
2. **Type Safety**: Use Pydantic models for all data structures
3. **Async/Await**: Use async patterns throughout for scalability
4. **Error Handling**: Comprehensive error handling with proper logging

### Testing Standards

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Protocol Tests**: Verify A2A and Zed ACP compliance
4. **Mocking**: Use mocks for external dependencies (agents, networks)

### Documentation

1. **Docstrings**: Document all public functions and classes
2. **Type Hints**: Include type hints for all parameters and returns
3. **Examples**: Provide practical examples in documentation
4. **Changelog**: Update documentation when features change

## Common Development Tasks

### Adding New Tool Types

```python
# 1. Add tool type to tool_config.py
class NewToolType(BashTool):
    # Implementation

# 2. Add tests in test_tool_config.py
def test_new_tool_type():
    # Test implementation

# 3. Update documentation in tool-execution.md
# 4. Add examples in tools.yaml
```

### Adding New API Methods

```python
# 1. Add method to A2A server in src/a2a/server.py
async def handle_new_method(params):
    # Implementation

# 2. Add tests in test_a2a_server.py
def test_new_method():
    # Test implementation

# 3. Update api-methods.md documentation
# 4. Add to AgentCard if needed
```

### Debugging Agent Issues

```python
# 1. Use dummy agent for testing
export A2A_AGENT_COMMAND="python tests/dummy_agent.py"

# 2. Enable agent debugging
export A2A_DEBUG_AGENT=true

# 3. Check agent logs
tail -f logs/a2a_acp.log | grep -E "(agent|zed_agent|dummy)"
```

## Performance Profiling

### Memory Profiling

```bash
# Install profiling tools
pip install memory-profiler psutil

# Profile memory usage
python -m memory_profiler src/a2a_acp/main.py

# Monitor process memory
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

### CPU Profiling

```bash
# Install profiling tools
pip install cProfile pstats

# Profile CPU usage
python -m cProfile -o profile.stats src/a2a_acp/main.py

# Analyze results
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(10)
"
```

## IDE Configuration

### VS Code

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "files.associations": {
    "*.yaml": "yaml",
    "*.yml": "yaml"
  }
}
```

### PyCharm

1. **Configure Python Interpreter**: Use the virtual environment
2. **Enable Type Checking**: Configure mypy for type validation
3. **Test Configuration**: Set up pytest for test execution
4. **Documentation Preview**: Install markdown preview plugin

## Troubleshooting Development

### Import Errors

```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify installation
python -c "import src.a2a.models; print('✅ Core imports work')"
```

### Test Failures

```bash
# Run specific failing test with verbose output
python -m pytest tests/test_a2a_task_manager.py::TestA2ATaskManager::test_create_task_basic -v -s

# Check test environment
export A2A_AGENT_COMMAND="python tests/dummy_agent.py"
python -m pytest tests/test_a2a_task_manager.py -v
```

### Database Issues

```bash
# Reset development database
rm -f data/a2a_acp.db
python -c "from src.a2a_acp.database import init_db; init_db()"

# Check database schema
sqlite3 data/a2a_acp.db ".schema"
```

## Getting Help

- **Architecture Questions**: See [architecture.md](architecture.md)
- **API Development**: Check [api-methods.md](api-methods.md)
- **Testing Help**: See [testing.md](testing.md) (when available)
- **Code Standards**: Follow patterns in existing codebase

---

**Development environment ready!** 💻 Next: [Testing Guide](testing.md) for comprehensive testing strategies.