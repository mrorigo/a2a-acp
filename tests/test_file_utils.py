# a2a-acp/tests/test_file_utils.py
"""
Tests for atomic file write and remove utilities in a2a_acp.utils.file_utils.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from a2a_acp.utils.file_utils import atomic_remove, atomic_write_json


def _file_permissions(path: Path) -> int:
    """Return the permission bits of the file as a 0oNNN integer."""
    return path.stat().st_mode & 0o777


def test_atomic_write_json_creates_file_and_sets_mode(tmp_path: Path):
    """
    Verify atomic_write_json writes JSON to the destination path and restricts
    permissions to 0600 (owner read/write).
    """
    dest = tmp_path / "auth.json"
    payload = {
        "type": "oauth",
        "access": "access-token-123",
        "refresh": "refresh-token-abc",
        "expires": 1670000000000,
    }

    atomic_write_json(dest, payload)

    assert dest.exists(), "Destination auth.json should have been created"
    with dest.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == payload, "JSON content should match the serialized payload"

    # Permission check (owner read/write only)
    assert _file_permissions(dest) == 0o600


def test_atomic_write_overwrites_existing_file_and_cleans_tmp(tmp_path: Path):
    """
    Ensure that atomic_write_json replaces an existing file and does not leave
    temporary files behind.
    """
    dest = tmp_path / "auth.json"
    old_content = {"old": True}
    dest.write_text(json.dumps(old_content), encoding="utf-8")

    # First write with new token
    first_payload = {"type": "oauth", "access": "first", "refresh": "r1", "expires": 1}
    atomic_write_json(dest, first_payload)
    assert json.loads(dest.read_text(encoding="utf-8")) == first_payload

    # Overwrite again
    second_payload = {
        "type": "oauth",
        "access": "second",
        "refresh": "r2",
        "expires": 2,
    }
    atomic_write_json(dest, second_payload)
    assert json.loads(dest.read_text(encoding="utf-8")) == second_payload

    # Only the final file should remain
    entries = list(tmp_path.iterdir())
    assert len(entries) == 1 and entries[0].name == "auth.json"


def test_atomic_remove_removes_file_and_missing_ok(tmp_path: Path):
    """
    atomic_remove should delete an existing file; missing_ok=True should not
    raise when the file does not exist.
    """
    dest = tmp_path / "tmp.json"
    dest.write_text("data", encoding="utf-8")
    assert dest.exists()

    atomic_remove(dest)
    assert not dest.exists()

    # Missing ok should not raise
    atomic_remove(dest, missing_ok=True)


def test_atomic_remove_missing_not_ok_raises(tmp_path: Path):
    """
    atomic_remove with missing_ok=False should raise a FileNotFoundError when the file is missing.
    """
    dest = tmp_path / "does_not_exist.json"
    assert not dest.exists()

    with pytest.raises(FileNotFoundError):
        atomic_remove(dest, missing_ok=False)
