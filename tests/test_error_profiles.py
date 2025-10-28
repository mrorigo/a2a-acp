import pytest

from a2a_acp.error_profiles import (
    ErrorProfile,
    build_acp_error,
    parse_error_profile,
)


def test_build_acp_error_converts_detail_for_basic_profile():
    contract = build_acp_error(
        profile=ErrorProfile.ACP_BASIC,
        code=-32002,
        message="Resource not found",
        detail={"path": "/tmp/example.txt"},
        retryable=False,
    )

    mcp_error = contract.to_mcp_error()

    assert mcp_error["detail"] == '{"path":"/tmp/example.txt"}'
    assert contract.diagnostics is not None
    assert "suppressed_detail" in contract.diagnostics
    assert contract.diagnostics["suppressed_detail"]["raw_detail"] == {"path": "/tmp/example.txt"}


def test_build_acp_error_preserves_extended_detail():
    detail_payload = {"path": "/tmp/example.txt", "hint": "check permissions"}
    contract = build_acp_error(
        profile=ErrorProfile.EXTENDED_JSON,
        code=-32002,
        message="Resource not found",
        detail=detail_payload,
        retryable=True,
    )

    mcp_error = contract.to_mcp_error()

    assert mcp_error["detail"] == detail_payload
    assert contract.diagnostics is None


def test_parse_error_profile_validates_supported_values():
    assert parse_error_profile(None) is ErrorProfile.ACP_BASIC
    assert parse_error_profile("extended-json") is ErrorProfile.EXTENDED_JSON

    with pytest.raises(ValueError):
        parse_error_profile("unsupported-profile")
