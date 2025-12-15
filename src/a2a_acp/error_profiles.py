"""Error profile normalization helpers for ACP-compatible payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class ErrorProfile(str, Enum):
    """Enumeration of supported MCP error formats."""

    ACP_BASIC = "acp-basic"
    EXTENDED_JSON = "extended-json"


@dataclass(frozen=True)
class ErrorContract:
    """Normalized representation of an ACP error payload."""

    code: int
    message: str
    detail: Optional[Any] = None
    retryable: Optional[bool] = None
    diagnostics: Optional[Dict[str, Any]] = None

    def to_mcp_error(self) -> Dict[str, Any]:
        """Serialize the contract into an MCP error payload."""
        payload: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.retryable is not None:
            payload["retryable"] = self.retryable
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


def parse_error_profile(raw_profile: Optional[str]) -> ErrorProfile:
    """Parse an error profile string into the enum, validating supported values."""
    if not raw_profile:
        return ErrorProfile.ACP_BASIC
    try:
        return ErrorProfile(raw_profile)
    except ValueError as exc:
        supported = ", ".join(profile.value for profile in ErrorProfile)
        raise ValueError(
            f"Unsupported A2A error profile '{raw_profile}'. Supported profiles: {supported}"
        ) from exc


def _normalize_detail_for_profile(
    detail: Optional[Any],
    profile: ErrorProfile,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Normalise raw detail into a profile-compliant representation.

    Returns:
        Tuple of (detail_value, supplemental diagnostics) where diagnostics contains
        the raw structured data when it must be suppressed from the MCP payload.
    """
    if detail is None:
        return None, None

    if profile is ErrorProfile.ACP_BASIC:
        if isinstance(detail, str):
            return detail, None

        supplemental: Dict[str, Any] = {"raw_detail": detail}
        try:
            detail_string = json.dumps(detail, separators=(",", ":"))
        except (TypeError, ValueError):
            detail_string = str(detail)
        return detail_string, supplemental

    # EXTENDED_JSON preserves structured data verbatim
    return detail, None


def build_acp_error(
    *,
    profile: ErrorProfile,
    code: int,
    message: str,
    detail: Optional[Any] = None,
    retryable: Optional[bool] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> ErrorContract:
    """Construct an ErrorContract honoring the selected profile."""
    normalized_detail, suppressed_detail = _normalize_detail_for_profile(
        detail, profile
    )

    merged_diagnostics: Dict[str, Any] = {}
    if diagnostics:
        merged_diagnostics.update(diagnostics)
    if suppressed_detail:
        merged_diagnostics.setdefault("suppressed_detail", suppressed_detail)

    return ErrorContract(
        code=code,
        message=message,
        detail=normalized_detail,
        retryable=retryable,
        diagnostics=merged_diagnostics or None,
    )
