#!/usr/bin/env python3
"""
Simple test script for the A2A server implementation.

Tests basic functionality without requiring a full ZedACP agent setup.
"""

import asyncio
import json
import aiohttp
import sys
from typing import Dict, Any

from .server import create_a2a_server


async def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")

    server = create_a2a_server()
    app = server.get_fastapi_app()

    # Test the health endpoint directly
    from fastapi.testclient import TestClient
    client = TestClient(app)

    response = client.get("/health")
    print(f"Health check status: {response.status_code}")
    print(f"Health check response: {response.json()}")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "protocol": "A2A", "version": "0.1.0"}
    print("✅ Health check test passed\n")


async def test_jsonrpc_request():
    """Test a basic JSON-RPC 2.0 request."""
    print("Testing JSON-RPC 2.0 request...")

    server = create_a2a_server()
    app = server.get_fastapi_app()

    from fastapi.testclient import TestClient
    client = TestClient(app)

    # Test a message/send request
    request_data = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Hello, A2A!"
                    }
                ],
                "messageId": "test_msg_123"
            }
        }
    }

    response = client.post("/", json=request_data)
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text}")

    if response.status_code == 200:
        response_data = response.json()
        print(f"Parsed response: {response_data}")

        # Should get either a success response or an error response
        assert "jsonrpc" in response_data
        assert response_data["jsonrpc"] == "2.0"
        assert "id" in response_data
        assert response_data["id"] == 1
        print("✅ JSON-RPC request test passed\n")
    else:
        print(f"❌ JSON-RPC request test failed with status {response.status_code}")
        return False

    return True


async def test_method_not_found():
    """Test error handling for unknown methods."""
    print("Testing method not found error...")

    server = create_a2a_server()
    app = server.get_fastapi_app()

    from fastapi.testclient import TestClient
    client = TestClient(app)

    # Test an unknown method
    request_data = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "unknown/method",
        "params": {}
    }

    response = client.post("/", json=request_data)
    response_data = response.json()

    print(f"Error response: {response_data}")

    # Should get a method not found error
    assert response_data["jsonrpc"] == "2.0"
    assert response_data["id"] == 2
    assert "error" in response_data
    assert response_data["error"]["code"] == -32601  # Method not found
    assert "Method not found" in response_data["error"]["message"]

    print("✅ Method not found error test passed\n")


async def run_tests():
    """Run all A2A server tests."""
    print("🚀 Starting A2A Server Tests\n")
    print("=" * 50)

    try:
        await test_health_check()
        await test_jsonrpc_request()
        await test_method_not_found()

        print("=" * 50)
        print("🎉 All A2A server tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)