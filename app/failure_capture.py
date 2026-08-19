"""Draft-only failure capture for live Ramone RAG answers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from pydantic import BaseModel, Field


SCHEMA_VERSION = "atlas-eval/pending-failure-case/v1"
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "auth",
    "cookie",
    "credential",
    "password",
    "secret",
    "session",
    "token",
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(authorization:\s*bearer\s+\S+|x-api-key:\s*\S+|sk-[a-z0-9_-]{16,}|"
    r"gh[opsu]_[a-z0-9_]{16,}|pat_[a-z0-9_]{16,})"
)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class FailureCaptureError(ValueError):
    """A submitted failure report is incomplete or unsafe to store."""


class FailureSource(BaseModel):
    """Public source evidence displayed with the reported answer."""

    id: str = Field(default="", max_length=300)
    preview: str = Field(default="", max_length=800)


class FailureCaptureRequest(BaseModel):
    """Browser-reported failure for draft eval-case creation."""

    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=8000)
    reason: str = Field(
        default="reported as incorrect or ungrounded by a live user",
        min_length=1,
        max_length=500,
    )
    expected_behavior: str = Field(
        default=(
            "Answer only from cited public Atlas Systems source evidence; "
            "say when the available context does not prove the claim."
        ),
        min_length=1,
        max_length=1000,
    )
    sources: list[FailureSource] = Field(default_factory=list, max_length=10)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_array(values: list[str]) -> str:
    if not values:
        return "[]"
    lines = ["["]
    for value in values:
        lines.append(f"  {_toml_str(value)},")
    lines.append("]")
    return "\n".join(lines)


def _walk_for_secrets(value, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _SECRET_KEY_PARTS):
                raise FailureCaptureError(f"secret-like field name at {path}")
            _walk_for_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_for_secrets(child, f"{path}[{index}]")
    elif isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        raise FailureCaptureError(f"secret-like value at {path}")


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    return slug[:48].strip("-") or "reported-live-answer"


def _source_context(sources: list[FailureSource], answer: str) -> str:
    if not sources:
        return f"Reported live answer had no displayed source cards.\n\nAnswer:\n{answer}"

    lines = ["Displayed source cards:"]
    for index, source in enumerate(sources, start=1):
        label = source.id or f"source-{index}"
        preview = source.preview or "No preview provided."
        lines.append(f"[{index}] {label}: {preview}")
    lines.extend(["", "Reported live answer:", answer])
    return "\n".join(lines)


def render_pending_case(
    request: FailureCaptureRequest,
    *,
    model: str,
    route: str,
    captured_at: str,
    case_id: str,
) -> str:
    """Render a pending eval-harness TOML case from a safe report."""
    context = _source_context(request.sources, request.answer)
    forbidden = [f"claim:{request.reason}"]
    lines = [
        "# Draft-only pending eval case generated from sanitized live-failure evidence.",
        "# Review, edit, and move to cases/ before running it as accepted eval coverage.",
        f"id = {_toml_str(case_id)}",
        'task_type = "ramone-rag-generation"',
        'description = "Reported live Ramone RAG answer issue"',
        "think = false",
        f"prompt = {_toml_str(request.question)}",
        f"context = {_toml_str(context)}",
        f"required = {_toml_array([request.expected_behavior])}",
        f"forbidden = {_toml_array(forbidden)}",
        'format_checks = ["non_empty"]',
        "",
        "[pending]",
        f"schema_version = {_toml_str(SCHEMA_VERSION)}",
        f"captured_at = {_toml_str(captured_at)}",
        'status = "draft-only"',
        "auto_promotion = false",
        'source = "ramone-edge"',
        f"model = {_toml_str(model)}",
        f"route = {_toml_str(route)}",
        'tags = ["live-report", "draft-only", "ramone-rag-generation"]',
    ]
    return "\n".join(lines) + "\n"


def write_failure_capture(
    request: FailureCaptureRequest,
    *,
    out_dir: Path,
    model: str,
    route: str = "/ask/stream",
) -> tuple[str, Path]:
    """Write a draft pending eval case and return its id and path."""
    raw = request.model_dump(mode="json")
    _walk_for_secrets(raw)

    captured_at = _utc_now()
    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    case_id = (
        f"pending-ramone-rag-generation-{_slug(request.reason)}-"
        f"{captured_at[:10].replace('-', '')}-{digest}"
    )
    rendered = render_pending_case(
        request,
        model=model,
        route=route,
        captured_at=captured_at,
        case_id=case_id,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.toml"
    if out_path.exists():
        raise FailureCaptureError(f"pending case already exists: {out_path.name}")
    out_path.write_text(rendered, encoding="utf-8")
    return case_id, out_path
