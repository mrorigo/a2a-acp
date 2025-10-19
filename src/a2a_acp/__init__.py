"""
A2A-ACP: Native A2A Protocol Server

A complete A2A v0.3.0 protocol implementation that bridges ZedACP agents
to modern A2A clients using JSON-RPC 2.0 over HTTP.

This replaces the legacy ACP IBM ACP bridge with a native A2A implementation.
"""

from .main import create_app

__all__ = ["create_app"]