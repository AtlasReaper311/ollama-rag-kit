from pathlib import Path

import pytest

from app.failure_capture import (
    FailureCaptureError,
    FailureCaptureRequest,
    write_failure_capture,
)


def test_write_failure_capture_creates_pending_eval_case(tmp_path: Path):
    request = FailureCaptureRequest(
        question="What backs Ramone answers?",
        answer="Ramone answers from public source cards.",
        reason="answer omitted a required citation",
        sources=[{"id": "doc.md#0", "preview": "Public source card preview."}],
    )

    case_id, path = write_failure_capture(
        request,
        out_dir=tmp_path,
        model="qwen3.5-mtp",
    )

    assert case_id.startswith("pending-ramone-rag-generation-")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert 'task_type = "ramone-rag-generation"' in text
    assert 'status = "draft-only"' in text
    assert "auto_promotion = false" in text
    assert 'model = "qwen3.5-mtp"' in text
    assert "Public source card preview." in text


def test_write_failure_capture_rejects_secret_like_values(tmp_path: Path):
    request = FailureCaptureRequest(
        question="What happened?",
        answer="authorization: bearer abcdefghijklmnopqrstuvwxyz",
        reason="reported bad answer",
    )

    with pytest.raises(FailureCaptureError, match="secret-like value"):
        write_failure_capture(request, out_dir=tmp_path, model="qwen3.5-mtp")
