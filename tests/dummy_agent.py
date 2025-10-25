#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
import threading
import random
import asyncio
from typing import Any, Dict, Optional, List, Tuple

SESSION_ID = "session-test"
TOOL_TESTING_ENABLED = True  # Enable tool call simulation for testing
cancel_event = threading.Event()
current_request_id_lock = threading.Lock()
current_request_id: Optional[int] = None


def set_current_request(request_id: Optional[int]) -> None:
    global current_request_id
    with current_request_id_lock:
        current_request_id = request_id


def get_current_request() -> Optional[int]:
    with current_request_id_lock:
        return current_request_id


def send(payload: Dict[str, Any]) -> None:
    message = json.dumps(payload)
    sys.stdout.write(message + "\n")
    sys.stdout.flush()
    print(f"dummy agent: Sending response: {message}", file=sys.stderr, flush=True)


def handle_initialize(message: Dict[str, Any]) -> None:
    send({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {}}})


def handle_session_new(message: Dict[str, Any]) -> None:
    send({"jsonrpc": "2.0", "id": message["id"], "result": {"sessionId": SESSION_ID}})


def handle_session_prompt(message: Dict[str, Any]) -> None:
    cancel_event.clear()
    set_current_request(message["id"])

    prompt_data = message.get("params", {}).get("prompt", [])
    print(f"dummy agent prompt: {prompt_data}", file=sys.stderr, flush=True)

    # Extract text from structured prompt format
    prompt_text = ""
    if isinstance(prompt_data, list):
        # Handle structured prompt format from ZedACP
        for item in prompt_data:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    prompt_text += item.get("text", "")
                elif "text" in item:  # Also handle direct text field
                    prompt_text += item.get("text", "")
            elif isinstance(item, str):
                prompt_text += item
    elif isinstance(prompt_data, str):
        # Handle string format
        prompt_text = prompt_data
    else:
        # Handle other formats
        prompt_text = str(prompt_data)

    def worker() -> None:
        try:
            start = time.time()
            words = prompt_text.split()
            words.append("--END-OF-RESPONSE--")
            check_count = 0
            while True:
                # Check for cancellation more frequently
                if cancel_event.wait(0.05):  # Check every 50ms instead of 100ms
                    print("dummy agent prompt cancelled", file=sys.stderr, flush=True)
                    set_current_request(None)
                    return
                if words:
                    word = words.pop(0)
                    send(
                        {
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "sessionId": SESSION_ID,
                                "update": {
                                    "sessionUpdate": "agent_message_chunk",
                                    "content": {"type": "text", "text": f"{word} "}
                                }
                            },
                        }
                    )
                    # Also check for cancellation after sending each message
                    if cancel_event.is_set():
                        print("dummy agent prompt cancelled after sending message", file=sys.stderr, flush=True)
                        set_current_request(None)
                        return
                check_count += 1
                # Run for a shorter time to make cancellation more testable
                if time.time() - start >= 2.0:  # Reduced from 5.0 to 2.0 seconds
                    break
            send({"jsonrpc": "2.0", "id": message["id"], "result": {"stopReason": "stop"}})
            set_current_request(None)
        except Exception as exc:  # pragma: no cover - debug aid
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": SESSION_ID,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": f"error: {exc}"}
                        }
                    },
                }
            )

    t = threading.Thread(target=worker, name="dummy-agent-prompt", daemon=True)
    t.start()




def handle_session_cancel(_: Dict[str, Any]) -> None:
    print("dummy agent: cancel received", file=sys.stderr, flush=True)
    cancel_event.set()
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": SESSION_ID,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "cancel acknowledged"}
                }
            },
        }
    )
    send({"jsonrpc": "2.0", "method": "session/cancelled", "params": {"sessionId": SESSION_ID}})
    request_id = get_current_request()
    if request_id is not None:
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": 499, "message": "cancelled"},
            }
        )
        set_current_request(None)


# Tool Simulation Configuration
TOOL_SIMULATIONS = {
    "web_request": {
        "execution_time": 0.5,
        "success_rate": 0.9,
        "requires_confirmation": False,
        "output": "HTTP 200: Success"
    },
    "dangerous_operation": {
        "execution_time": 1.0,
        "success_rate": 0.8,
        "requires_confirmation": True,
        "confirmation_message": "This operation is dangerous. Proceed?",
        "output": "Dangerous operation completed"
    },
    "failing_tool": {
        "execution_time": 0.2,
        "success_rate": 0.1,  # High failure rate for testing
        "error_message": "Simulated tool failure"
    },
    "database_query": {
        "execution_time": 0.8,
        "success_rate": 0.95,
        "requires_confirmation": False,
        "output": "Query executed successfully: 42 rows returned"
    },
    "file_operation": {
        "execution_time": 0.3,
        "success_rate": 0.9,
        "requires_confirmation": False,
        "output": "File operation completed successfully"
    }
}

# Global state for tool call simulation
active_tool_calls: Dict[str, Dict[str, Any]] = {}
tool_call_counter = 0


def simulate_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate tool execution with configurable behavior."""
    global tool_call_counter
    tool_call_counter += 1

    tool_id = tool_call.get("toolId", "unknown")
    tool_config = TOOL_SIMULATIONS.get(tool_id, {
        "execution_time": 0.5,
        "success_rate": 0.8,
        "requires_confirmation": False,
        "output": f"Tool {tool_id} executed"
    })

    call_id = f"call_{tool_call_counter}"
    active_tool_calls[call_id] = {
        "id": call_id,
        "tool_id": tool_id,
        "status": "started",
        "parameters": tool_call.get("parameters", {}),
        "start_time": time.time()
    }

    # Check if tool requires confirmation
    if tool_config.get("requires_confirmation", False):
        return {
            "toolCallId": tool_call.get("id", call_id),
            "status": "pending_confirmation",
            "content": [{"type": "text", "text": tool_config["confirmation_message"]}],
            "rawOutput": tool_config["confirmation_message"]
        }

    # Simulate execution
    time.sleep(tool_config["execution_time"])

    # Simulate success/failure based on success_rate
    success = random.random() < tool_config["success_rate"]

    if success:
        return {
            "toolCallId": tool_call.get("id", call_id),
            "status": "completed",
            "content": [{"type": "text", "text": tool_config["output"]}],
            "rawOutput": tool_config["output"]
        }
    else:
        error_msg = tool_config.get("error_message", f"Tool {tool_id} failed")
        return {
            "toolCallId": tool_call.get("id", call_id),
            "status": "failed",
            "content": [{"type": "text", "text": f"Error: {error_msg}"}],
            "rawOutput": f"Error: {error_msg}"
        }


def handle_session_prompt_with_tools(message: Dict[str, Any]) -> None:
    """Enhanced session/prompt handler with tool call simulation."""
    cancel_event.clear()
    set_current_request(message["id"])

    prompt_data = message.get("params", {}).get("prompt", [])
    print(f"dummy agent prompt: {prompt_data}", file=sys.stderr, flush=True)

    # Extract text from structured prompt format
    prompt_text = ""
    if isinstance(prompt_data, list):
        # Handle structured prompt format from ZedACP
        for item in prompt_data:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    prompt_text += item.get("text", "")
                elif "text" in item:  # Also handle direct text field
                    prompt_text += item.get("text", "")
            elif isinstance(item, str):
                prompt_text += item
    elif isinstance(prompt_data, str):
        # Handle string format
        prompt_text = prompt_data
    else:
        # Handle other formats
        prompt_text = str(prompt_data)

    # Check if prompt contains tool requests
    if TOOL_TESTING_ENABLED and "tool" in prompt_text.lower():
        # Simulate tool call responses
        simulate_tool_calls(message["id"])
        return

    # Fall back to original prompt handling
    handle_session_prompt(message)


def simulate_tool_calls(request_id: int) -> None:
    """Simulate ZedACP tool calls in response to tool requests."""
    def worker() -> None:
        try:
            # Simulate various tool call scenarios
            tool_scenarios = [
                {"toolId": "web_request", "parameters": {"method": "GET", "url": "https://api.example.com"}},
                {"toolId": "dangerous_operation", "parameters": {"action": "delete"}},
                {"toolId": "database_query", "parameters": {"query": "SELECT * FROM users"}},
                {"toolId": "file_operation", "parameters": {"path": "/tmp/test.txt", "operation": "read"}}
            ]

            for scenario in tool_scenarios:
                if cancel_event.is_set():
                    break

                # Send tool call notification
                send({
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": SESSION_ID,
                        "update": {
                            "sessionUpdate": "tool_call",
                            "toolCalls": [{
                                "id": f"call_{random.randint(1000, 9999)}",
                                "toolId": scenario["toolId"],
                                "parameters": scenario["parameters"]
                            }]
                        }
                    }
                })

                # Wait a bit between tool calls
                time.sleep(0.2)

                # Send simulated tool result
                result = simulate_tool_call(scenario)
                send({
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": SESSION_ID,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallUpdates": [result]
                        }
                    }
                })

                time.sleep(0.1)

            # Send completion notification
            send({"jsonrpc": "2.0", "id": request_id, "result": {"stopReason": "stop"}})
            set_current_request(None)

        except Exception as exc:
            send({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": SESSION_ID,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": f"error: {exc}"}
                    }
                }
            })
            set_current_request(None)

    t = threading.Thread(target=worker, name="dummy-agent-tool-simulation", daemon=True)
    t.start()


def handle_permission_response(message: Dict[str, Any]) -> None:
    """Handle permission request responses for tool confirmation."""
    # This would handle responses to session/request_permission calls
    # For now, just log the response
    print(f"Permission response: {message}", file=sys.stderr, flush=True)


HANDLERS = {
    "initialize": handle_initialize,
    "session/new": handle_session_new,
    "session/prompt": handle_session_prompt_with_tools,
    "session/cancel": handle_session_cancel,
    "session/request_permission": handle_permission_response,
}


def main() -> None:
    print("dummy agent: Starting up", file=sys.stderr, flush=True)
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        try:
            message = json.loads(raw)
            method = message.get("method")
            print(f"dummy agent: Received {method} request: {json.dumps(message)}", file=sys.stderr, flush=True)

            handler = HANDLERS.get(method)
            if handler is None:
                print(f"dummy agent: No handler for method {method}", file=sys.stderr, flush=True)
                continue

            handler(message)

            if method == "session/cancel":
                print("dummy agent: Received cancel, exiting", file=sys.stderr, flush=True)
                break

        except json.JSONDecodeError as e:
            print(f"dummy agent: Failed to parse JSON: {raw}, error: {e}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"dummy agent: Error processing message: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
