"""Shared application-level maintenance gate for RAG writes."""

import os

from fastapi import HTTPException


WRITE_FROZEN_DETAIL = {
    "error": "write_frozen",
    "message": "RAG memory and failure-capture writes are temporarily disabled",
}


def write_freeze_enabled() -> bool:
    """Read the explicit freeze switch without changing protected settings."""
    value = os.getenv("WRITE_FREEZE", "false").strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise RuntimeError("WRITE_FREEZE must be an explicit boolean")
    return value in {"1", "true", "yes", "on"}


def require_writes_enabled(write_frozen: bool) -> None:
    """Fail closed before a route can persist session or failure data."""
    if write_frozen:
        raise HTTPException(status_code=503, detail=WRITE_FROZEN_DETAIL)
