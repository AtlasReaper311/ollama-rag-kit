import pytest
from fastapi import HTTPException

from app.write_policy import require_writes_enabled, write_freeze_enabled


def test_write_policy_allows_reads_when_not_frozen():
    require_writes_enabled(False)


def test_write_policy_returns_explicit_503_when_frozen():
    with pytest.raises(HTTPException) as exc_info:
        require_writes_enabled(True)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "write_frozen"


def test_write_freeze_env_is_explicit_and_fail_closed(monkeypatch):
    monkeypatch.setenv("WRITE_FREEZE", "true")
    assert write_freeze_enabled() is True
    monkeypatch.setenv("WRITE_FREEZE", "not-a-boolean")
    with pytest.raises(RuntimeError, match="WRITE_FREEZE"):
        write_freeze_enabled()
