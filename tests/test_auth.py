"""Tests for the ATLAS_SECRET middleware.

These tests intentionally use a tiny app rather than the full FastAPI
that ollama-rag-kit boots, because the goal is to prove the middleware
in isolation. The full app is exercised via integration in CI.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth import AtlasSecretMiddleware, load_required_secret

VALID_SECRET = "this-is-a-thirty-two-or-more-char-secret-value"


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AtlasSecretMiddleware, secret=VALID_SECRET)

    @app.get("/")
    def root() -> dict:
        return {"ok": True}

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/ask")
    def ask() -> dict:
        return {"ok": True}

    return app


def test_open_paths_bypass_auth():
    client = TestClient(_build_app())
    for path in ("/", "/docs", "/openapi.json", "/redoc"):
        res = client.get(path)
        # /docs and /redoc render HTML; /openapi.json returns JSON;
        # all should not be 401.
        assert res.status_code != 401, f"unexpected 401 on open path {path}"


def test_protected_path_rejects_missing_header():
    client = TestClient(_build_app())
    res = client.post("/ask", json={})
    assert res.status_code == 401
    assert res.json() == {"error": "unauthorized"}


def test_protected_path_rejects_wrong_secret():
    client = TestClient(_build_app())
    res = client.post(
        "/ask",
        headers={"x-atlas-secret": "definitely-the-wrong-secret"},
        json={},
    )
    assert res.status_code == 401


def test_protected_path_accepts_correct_secret():
    client = TestClient(_build_app())
    res = client.post(
        "/ask",
        headers={"x-atlas-secret": VALID_SECRET},
        json={},
    )
    assert res.status_code == 200
    assert res.json() == {"ok": True}


def test_health_requires_secret():
    """The Worker fronting this service injects the secret on every call
    including /health for the wake probe, so /health must not be open."""
    client = TestClient(_build_app())
    res = client.get("/health")
    assert res.status_code == 401
    res = client.get("/health", headers={"x-atlas-secret": VALID_SECRET})
    assert res.status_code == 200


def test_constructor_rejects_empty_secret():
    app = FastAPI()
    with pytest.raises(RuntimeError, match="non-empty secret"):
        app.add_middleware(AtlasSecretMiddleware, secret="")
        # Trigger app build so middleware constructor runs.
        TestClient(app).get("/")


def test_load_required_secret_raises_when_unset(monkeypatch):
    monkeypatch.delenv("ATLAS_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="ATLAS_SECRET is not set"):
        load_required_secret()


def test_load_required_secret_rejects_short_value(monkeypatch):
    monkeypatch.setenv("ATLAS_SECRET", "too-short")
    with pytest.raises(RuntimeError, match="at least 32 characters"):
        load_required_secret()


def test_load_required_secret_accepts_valid_value(monkeypatch):
    monkeypatch.setenv("ATLAS_SECRET", VALID_SECRET)
    assert load_required_secret() == VALID_SECRET
