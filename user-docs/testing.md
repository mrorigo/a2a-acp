# Testing Guide

Comprehensive testing strategies and practices for A2A-ACP development.

## Overview

A2A-ACP includes **240+ comprehensive tests** covering:

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction testing
- **Protocol Tests**: A2A and Zed ACP compliance verification
- **Performance Tests**: Load testing and benchmarking
- **End-to-End Tests**: Complete workflow validation

## Test Categories

### 1. A2A-ACP Bridge Tests (`test_a2a_acp_bridge.py`)

**40+ tests** covering core A2A-ACP functionality:

- **Task Management**: Task creation, execution, cancellation
- **Input-Required Workflows**: Multi-turn conversation handling
- **Context Persistence**: Stateful conversation management
- **Error Handling**: Exception scenarios and recovery
- **Event Emission**: Push notification and event testing

**Key Test Areas:**
```python
# Task lifecycle testing
test_create_task_basic()
test_execute_task_success()
test_execute_task_input_required()
test_cancel_task()

# Input-required workflow testing
test_provide_input_and_continue()
test_input_required_detection()
test_cross_protocol_state_sync()
```

### 2. A2A Protocol Tests (`test_a2a_server.py`)

**16+ tests** for A2A protocol compliance:

- **JSON-RPC 2.0**: Request/response format validation
- **Method Implementation**: All A2A method testing
- **Authentication**: Bearer token and API key testing
- **Error Handling**: Protocol-compliant error responses

### 3. Task Manager Tests (`test_a2a_task_manager.py`)

**50+ tests** for task lifecycle management:

- **State Transitions**: Complete state machine validation
- **Context Management**: Conversation persistence testing
- **Message Handling**: Message processing and history
- **Concurrent Operations**: Multi-task coordination

### 4. Tool Configuration Tests (`test_tool_config.py`)

**20+ tests** for tool system functionality:

- **Tool Loading**: YAML configuration parsing and validation
- **Parameter Validation**: Type checking and requirement validation
- **Tool Execution**: Script execution and result handling
- **Security Controls**: Sandboxing and permission testing

### 5. Protocol Compliance Tests (`test_tool_protocol_compliance.py`)

**15+ tests** for protocol compliance:

- **A2A AgentCard Generation**: Skills and capabilities validation
- **Zed ACP Tool Call Format**: Tool call interception and response
- **Event Emission**: Protocol-compliant event structures
- **Cross-Protocol State**: State synchronization between protocols

### 6. Integration Tests (`test_tool_system_integration.py`)

**10+ tests** for end-to-end workflows:

- **Complete Workflows**: Full request → response cycles
- **Event Flow**: Event generation and consumption
- **Error Recovery**: Failure scenarios and recovery
- **Performance Validation**: Load and timing validation

## Running Tests

### All Tests

```bash
# Run complete test suite
make test

# With coverage report
make test-coverage

# With verbose output
python -m pytest tests/ -v
```

### Specific Test Files

```bash
# Test core functionality
python -m pytest tests/test_a2a_acp_bridge.py -v

# Test A2A protocol
python -m pytest tests/test_a2a_server.py -v

# Test task management
python -m pytest tests/test_a2a_task_manager.py -v

# Test tool system
python -m pytest tests/test_tool_config.py tests/test_tool_protocol_compliance.py -v
```

### Specific Test Functions

```bash
# Test single function
python -m pytest tests/test_a2a_task_manager.py::TestA2ATaskManager::test_create_task_basic -v

# Test with debugging
python -m pytest tests/test_a2a_task_manager.py::TestA2ATaskManager::test_execute_task_success -v -s

# Test input-required workflow
python -m pytest tests/test_a2a_task_manager.py -k "input_required" -v
```

### Development Testing

```bash
# Run tests in watch mode (during development)
python -m pytest tests/ -v --tb=short -x

# Stop on first failure for quick debugging
python -m pytest tests/test_a2a_task_manager.py -x -v

# Run only tests that failed previously
python -m pytest tests/ --lf
```

## Test Environment Setup

### Using Dummy Agent

For isolated testing without external dependencies:

```bash
# Set dummy agent for testing
export A2A_AGENT_COMMAND="python tests/dummy_agent.py"
export A2A_AGENT_DESCRIPTION="Test agent for development"

# Run tests with dummy agent
python -m pytest tests/test_a2a_task_manager.py -v
```

### Test Database

Tests use an in-memory database by default:

```python
# Test database configuration in tests
@pytest.fixture
def test_db():
    # Creates temporary database for testing
    # Automatically cleaned up after tests
    pass
```

## Writing Tests

### Test Structure

Follow the established testing patterns:

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestComponent:
    """Test class for specific component."""

    @pytest.fixture
    def component_instance(self):
        """Create component instance for testing."""
        return Component()

    @pytest.mark.asyncio
    async def test_component_method(self, component_instance):
        """Test specific method functionality."""
        # Arrange
        input_data = {"test": "data"}

        # Act
        result = await component_instance.method(input_data)

        # Assert
        assert result.success == True
        assert result.data == expected_data
```

### Mocking External Dependencies

```python
# Mock Zed ACP agent for testing
@patch('src.a2a_acp.task_manager.ZedAgentConnection')
@pytest.mark.asyncio
async def test_with_mocked_agent(self, mock_zed_connection):
    """Test with mocked Zed ACP agent."""

    # Setup mock
    mock_connection = AsyncMock()
    mock_connection.prompt = AsyncMock(return_value={"response": "test"})
    mock_zed_connection.return_value = mock_connection

    # Test implementation
    # ... test code ...

    # Verify interactions
    mock_connection.prompt.assert_called_once()
```

### Async Testing

All tests use `pytest-asyncio` for async function testing:

```python
@pytest.mark.asyncio
async def test_async_functionality():
    """Test async component behavior."""

    # Use async/await patterns
    result = await async_component.method()

    # Assert results
    assert result is not None
```

## Test Coverage

### Coverage Goals

- **Overall Coverage**: >95% codebase coverage
- **Critical Paths**: 100% coverage for error handling and security
- **Protocol Compliance**: Complete coverage of A2A and Zed ACP interactions
- **Integration Points**: Full coverage of component interactions

### Coverage Reporting

```bash
# Generate coverage report
make test-coverage

# View coverage in detail
coverage report -m

# Generate HTML coverage report
coverage html
open htmlcov/index.html
```

## Performance Testing

### Load Testing

```python
# Performance test structure
@pytest.mark.asyncio
async def test_concurrent_task_execution(self):
    """Test system under concurrent load."""

    # Create multiple concurrent tasks
    tasks = [
        task_manager.execute_task(f"task_{i}", ["echo", f"test_{i}"])
        for i in range(100)
    ]

    # Execute concurrently
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    # Validate performance
    assert end_time - start_time < 30  # Should complete in <30 seconds
    assert all(r.status.state == "completed" for r in results)
```

### Memory Testing

```python
# Memory usage validation
@pytest.mark.asyncio
async def test_memory_usage(self):
    """Test memory usage remains stable."""

    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Execute memory-intensive operations
    for i in range(100):
        await task_manager.create_task(f"ctx_{i}", f"agent_{i}")

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    # Assert reasonable memory usage
    assert memory_increase < 50 * 1024 * 1024  # Less than 50MB increase
```

## Protocol Testing

### A2A Compliance Testing

```python
def test_a2a_json_rpc_compliance(self):
    """Test A2A JSON-RPC 2.0 compliance."""

    # Test request format
    request = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": "test_001",
        "params": {...}
    }

    # Validate against A2A specification
    assert request["jsonrpc"] == "2.0"
    assert "method" in request
    assert "id" in request
```

### Zed ACP Compliance Testing

```python
def test_zed_acp_protocol_compliance(self):
    """Test Zed ACP protocol compliance."""

    # Test Zed ACP message format
    zed_message = {
        "sessionId": "sess_123",
        "prompt": [
            {"type": "text", "text": "Hello"}
        ]
    }

    # Validate against Zed ACP specification
    assert "sessionId" in zed_message
    assert "prompt" in zed_message
```

## Integration Testing

### End-to-End Workflows

```python
@pytest.mark.asyncio
async def test_complete_workflow(self):
    """Test complete message → response workflow."""

    # 1. Send message
    message = Message(role="user", parts=[TextPart(kind="text", text="Hello")])
    task = await task_manager.create_task("ctx_123", "test_agent", message)

    # 2. Execute task
    result = await task_manager.execute_task(task.id, ["echo", "response"])

    # 3. Verify complete workflow
    assert result.status.state == "completed"
    assert len(result.history) >= 2  # Request + response
```

### Cross-Component Testing

```python
@pytest.mark.asyncio
async def test_event_driven_workflow(self):
    """Test event-driven component interactions."""

    # Setup event capture
    captured_events = []

    # Execute operation that generates events
    await task_manager.execute_task("task_123", ["echo", "test"])

    # Verify events were emitted correctly
    # Event verification logic
```

## Test Utilities

### Dummy Agent for Testing

```python
# tests/dummy_agent.py
class DummyAgentConnection:
    """Mock Zed ACP agent for testing."""

    async def prompt(self, session_id, messages):
        """Return predictable responses for testing."""
        return {
            "stopReason": "end_turn",
            "toolCalls": []
        }

    async def initialize(self):
        """Mock initialization."""
        return {"capabilities": {"tools": True}}

# Usage in tests
export A2A_AGENT_COMMAND="python tests/dummy_agent.py"
```

### Test Data Factories

```python
# Test data creation utilities
def create_test_message(text="Hello", role="user"):
    """Create test message with defaults."""
    return Message(
        role=role,
        parts=[TextPart(kind="text", text=text)],
        messageId=f"msg_{generate_id()}"
    )

def create_test_task(context_id="ctx_123", agent_name="test_agent"):
    """Create test task with defaults."""
    return Task(
        id=f"task_{generate_id()}",
        contextId=context_id,
        status=TaskStatus(state=TaskState.SUBMITTED),
        history=[],
        artifacts=[],
        metadata={}
    )
```

## Debugging Tests

### Verbose Test Output

```bash
# Run with detailed output
python -m pytest tests/test_a2a_task_manager.py -v -s --tb=long

# Debug specific test
python -m pytest tests/test_a2a_task_manager.py::TestA2ATaskManager::test_execute_task_input_required -v -s --tb=long
```

### Test Database Inspection

```bash
# Check test database state
sqlite3 :memory: "SELECT name FROM sqlite_master WHERE type='table';"

# Or for file-based tests
sqlite3 /tmp/test_a2a_acp.db "SELECT * FROM tasks LIMIT 5;"
```

### Mock Verification

```python
# Verify mock interactions
mock_connection.prompt.assert_called_once()
call_args = mock_connection.prompt.call_args

# Verify call parameters
assert call_args[0][0] == "expected_session_id"  # session_id
assert call_args[0][1] == expected_messages     # messages
```

## Continuous Integration

### CI Test Configuration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install uv
      - run: uv sync
      - run: uv pip install -e ".[dev]"
      - run: make test-coverage
      - uses: codecov/codecov-action@v1
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
```

## Best Practices

### Test Organization

1. **One Test File per Component**: `test_task_manager.py` for task management
2. **Descriptive Test Names**: `test_create_task_with_message()` not `test_create()`
3. **Arrange-Act-Assert**: Clear test structure with setup, execution, verification
4. **Independent Tests**: Tests should not depend on each other

### Mocking Strategy

1. **External Dependencies**: Mock Zed ACP agents, networks, file systems
2. **Internal Components**: Use dependency injection for testability
3. **State Management**: Mock databases and persistent state for isolation
4. **Time Dependencies**: Mock time-sensitive operations for predictable tests

### Async Testing

1. **Proper Decorators**: Use `@pytest.mark.asyncio` for async tests
2. **Await All Async Calls**: Ensure all async operations are awaited
3. **Event Loop Management**: Handle event loops properly in tests
4. **Concurrent Testing**: Use `asyncio.gather()` for concurrent operation testing

### Error Testing

1. **Exception Types**: Test for correct exception types and messages
2. **Error Recovery**: Test error recovery and fallback mechanisms
3. **Boundary Conditions**: Test edge cases and error boundaries
4. **Resource Cleanup**: Ensure resources are cleaned up in error scenarios

## Troubleshooting Tests

### Common Test Issues

**"Async def functions not supported"**
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Use proper test markers
@pytest.mark.asyncio
async def test_async_function():
    pass
```

**"Event loop issues"**
```bash
# Mock components that create event loops
@patch('component_that_creates_loops')
def test_with_mocked_loops(self, mock_component):
    pass
```

**"Database connection errors"**
```bash
# Use in-memory databases for tests
@pytest.fixture
def test_db():
    # Setup in-memory database
    # Return connection for testing
    pass
```

**"Import errors"**
```bash
# Check Python path in tests
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Verify imports work
python -c "import a2a_acp.a2a.models; print('Imports OK')"
```

---

**Testing strategy documented!** 🧪 For development workflow, see [development-setup.md](development-setup.md).
