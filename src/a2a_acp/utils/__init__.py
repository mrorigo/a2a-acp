"""
A2A-ACP utils package

This package contains small, focused helper modules required across the project.
By importing publicly useful utilities here (e.g. `atomic_write_json`), callers
can import concise symbols from `a2a_acp.utils`.

Example:
    from a2a_acp.utils import atomic_write_json

Note:
- Avoid importing heavy dependencies in this module to keep import times low.
- Keep public API surface minimal and explicit via `__all__`.
"""

from .file_utils import atomic_remove, atomic_write_json

__all__ = ["atomic_write_json", "atomic_remove"]
