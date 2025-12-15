# a2a-acp/src/a2a_acp/utils/file_utils.py
"""
Atomic file write and removal utilities.

This module implements helpers for performing atomic JSON writes to a file
(i.e. write to a tmp file in the same directory, fsync, rename to target)
and a safe removal helper.

These utilities intentionally avoid logging or returning sensitive contents.

Design notes:
- Writes are atomic via os.replace (POSIX rename semantics).
- Writes are fsynced to the temporary file before rename for durability.
- On POSIX, we attempt to fsync the directory to ensure rename durability; we
  gracefully degrade if that is not available (e.g., on some platforms).
- Files are chmod'ed to the requested `mode` (default 0o600).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

__all__ = ["atomic_write_json", "atomic_remove"]


def _fsync_directory(path: Path) -> None:
    """
    Attempt an fsync on a directory to flush the file system metadata so that an
    atomic rename is durable on crash. If the platform doesn't support O_DIRECTORY,
    or the operation fails for other reasons, the error is swallowed (we still
    succeeded in creating the file).
    """
    try:
        # POSIX-specific: open the directory and fsync it
        dir_fd = os.open(str(path), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        # Not critical — log at debug level and continue.
        logger.debug(
            "Directory fsync unavailable on this platform or failed", exc_info=True
        )


def atomic_write_json(
    dest_path: Union[str, Path],
    data: Any,
    mode: int = 0o600,
    *,
    ensure_dir: bool = True,
    tmp_suffix: Optional[str] = None,
) -> None:
    """
    Atomically write `data` (JSON-serializable) into `dest_path`.

    Steps:
    1. Ensure parent directory exists (if `ensure_dir` is True).
    2. Write content to a unique temporary file in the same directory.
    3. Flush & fsync the temporary file.
    4. os.replace() the temp file into the desired `dest_path`.
    5. chmod the resulting file to `mode`.
    6. Attempt to fsync the containing directory for improved durability.

    Args:
        dest_path: Path to write (str or Path).
        data: JSON-serializable object to write.
        mode: File mode to set (unix style), defaults to 0o600.
        ensure_dir: If True, create parent dirs if they don't exist.
        tmp_suffix: Optional suffix for the temporary file name.
    Raises:
        PermissionError, FileNotFoundError, OSError on IO errors.
    """
    dest = Path(dest_path)
    parent = dest.parent

    if ensure_dir:
        parent.mkdir(parents=True, exist_ok=True)

    # Prepare serialized bytes upfront so we can write and fsync the bytes
    # exactly as they'll appear on disk.
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )

    # Create a NamedTemporaryFile in the same directory. Use delete=False so we can
    # close it and then os.replace the file path on all platforms.
    tmp_file = None
    try:
        if tmp_suffix:
            tmp_file = tempfile.NamedTemporaryFile(
                mode="wb", dir=str(parent), delete=False, suffix=tmp_suffix
            )
        else:
            tmp_file = tempfile.NamedTemporaryFile(
                mode="wb", dir=str(parent), delete=False
            )
        try:
            # Write data, flush, fsync
            tmp_file.write(serialized)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        finally:
            tmp_file.close()

        # Atomically replace destination (POSIX atomic rename semantics)
        os.replace(tmp_file.name, str(dest))

        # Ensure file modes are set to restrict access (owner r/w only)
        try:
            os.chmod(str(dest), mode)
        except Exception:
            # On some platforms (e.g., certain Windows setups), chmod may not
            # behave exactly the same as Unix. We avoid failing the write if chmod
            # is unsupported — the security guidance in the plan suggests
            # production systems consider KMS or ACL controls.
            logger.debug(
                "chmod failed, platform may not support setting Unix-style mode",
                exc_info=True,
            )

        # Attempt to fsync parent directory metadata as well to make the rename durable.
        _fsync_directory(parent)

    except Exception as exc:
        # If something goes wrong, attempt to remove the temporary file if it exists.
        logger.exception("Failed to perform atomic write to %s", str(dest))
        if tmp_file is not None and tmp_file.name:
            try:
                os.remove(tmp_file.name)
            except Exception:
                logger.debug(
                    "Failed to remove temporary file %s", tmp_file.name, exc_info=True
                )
        # Re-raise so caller can act on failure.
        raise exc


def atomic_remove(path: Union[str, Path], *, missing_ok: bool = True) -> None:
    """
    Remove a file at `path` if it exists.

    Args:
        path: Path to remove.
        missing_ok: If True, silently ignore if file doesn't exist.
    Raises:
        OSError (or subclass) if removal fails and missing_ok is False.
    """
    p = Path(path)
    try:
        p.unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise
    except Exception:
        logger.exception("Failed to remove file: %s", str(p))
        raise
