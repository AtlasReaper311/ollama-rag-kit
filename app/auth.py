"""Shared-secret authentication for ramone-edge ↔ ollama-rag-kit.

The Cloudflare Tunnel hostname is technically discoverable (a CNAME on
the apex zone), so a known endpoint is not the same as an authenticated
one. The Worker holds an `UPSTREAM_SECRET` and injects it as the
`X-Atlas-Secret` header on every proxied request; this middleware
rejects any request that does not match.

The secret is loaded from `ATLAS_SECRET` in the environment. Constant-
time comparison is used so a timing oracle cannot recover the secret
byte-by-byte. If the secret is unset the service refuses to start; we
fail closed rather than allow an unauthenticated public endpoint to
appear by accident.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Endpoints that bypass auth. Keep this list short and obvious.
# `/` is the docs redirect; `/docs` and `/openapi.json` are FastAPI's
# generated interactive docs, useful only from localhost during dev.
# In production these are still reachable only via the tunnel, which
# the Worker fronts, so leaving the redirect open is harmless.
_OPEN_PATHS = frozenset({"/", "/docs", "/openapi.json", "/redoc"})


class AtlasSecretMiddleware(BaseHTTPMiddleware):
    """Enforce a shared secret header on every protected request."""

    def __init__(self, app: FastAPI, secret: str) -> None:
        super().__init__(app)
        if not secret:
            # Belt-and-braces: the constructor should already fail at
            # app boot, but check again in case a config change leaves
            # the secret blank at runtime.
            raise RuntimeError(
                "AtlasSecretMiddleware requires a non-empty secret; "
                "set ATLAS_SECRET in the environment.",
            )
        self._secret_bytes = secret.encode("utf-8")

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in _OPEN_PATHS:
            return await call_next(request)

        presented = request.headers.get("x-atlas-secret", "")
        if not presented or not hmac.compare_digest(
            presented.encode("utf-8"), self._secret_bytes,
        ):
            client = request.client.host if request.client else "unknown"
            logger.warning(
                "rejected unauthenticated request: path=%s client=%s",
                request.url.path,
                client,
            )
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
            )
        return await call_next(request)


def load_required_secret() -> str:
    """Return the ATLAS_SECRET from the environment, or raise.

    Called from the app factory so a misconfigured deployment dies at
    boot with a clear message rather than serving traffic with no auth.
    """
    secret = os.environ.get("ATLAS_SECRET", "").strip()
    if not secret:
        raise RuntimeError(
            "ATLAS_SECRET is not set. ollama-rag-kit refuses to start "
            "without a shared secret while exposed through a tunnel.",
        )
    if len(secret) < 32:
        raise RuntimeError(
            "ATLAS_SECRET must be at least 32 characters. "
            "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(48))'",
        )
    return secret
