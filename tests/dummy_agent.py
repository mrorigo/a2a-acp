from __future__ import annotations

import json
import sys
import time
import threading
from typing import Any, Dict, Optional

SESSION_ID = "session-test"
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
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


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
            if isinstance(item, dict) and item.get("type") == "text":
                prompt_text += item.get("text", "")
    else:
        # Handle legacy string format
        prompt_text = str(prompt_data)

    def worker() -> None:
        try:
            start = time.time()
            words = prompt_text.split()
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


HANDLERS = {
    "initialize": handle_initialize,
    "session/new": handle_session_new,
    "session/prompt": handle_session_prompt,
    "session/cancel": handle_session_cancel,
}


def main() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        message = json.loads(raw)
        method = message.get("method")
        handler = HANDLERS.get(method)
        if handler is None:
            continue
        handler(message)
        if method == "session/cancel":
            break


if __name__ == "__main__":
    main()
