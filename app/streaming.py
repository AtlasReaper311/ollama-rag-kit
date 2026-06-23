"""Server-Sent Events endpoint for streaming answers to ramone-edge.

The non-streaming `/ask` endpoint stays in place for direct integrations
and the OpenAPI documentation. The streaming endpoint mirrors the same
retrieval logic but emits events as soon as Ollama returns each token,
so the public Worker can pipe bytes to the browser in real time.

Event shapes (one JSON object per `data:` line, blank line separates events):

    data: {"type":"sources", "sources":[{"id":"...","preview":"..."}, ...]}
    data: {"type":"token",   "text":"..."}
    data: {"type":"done"}
    data: {"type":"error",   "reason":"..."}

`sources` is emitted first so the client can render provenance before
the answer starts streaming. `done` is always emitted, even on error,
so the client knows when to stop reading.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator

import httpx
from chromadb.api.models.Collection import Collection
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings
from app.retriever import retrieve

logger = logging.getLogger(__name__)

router = APIRouter()


class AskStreamRequest(BaseModel):
    """A streaming question. Same contract as AskRequest, separate class
    so future divergence does not break the non-streaming endpoint."""

    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


def _sse(event: dict) -> bytes:
    """Format a single SSE event as bytes."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")


async def _stream_answer(
    http: httpx.AsyncClient,
    settings: Settings,
    collection: Collection,
    body: AskStreamRequest,
) -> AsyncIterator[bytes]:
    """Produce the SSE byte stream for a single question.

    Retrieval happens first; sources are sent before any tokens. Then
    Ollama is called with `stream: true` and each delta is unwrapped
    into a token event. Network errors surface as an error event so
    the client always reaches a terminal state.
    """
    indexed = collection.count()
    if indexed == 0:
        yield _sse({"type": "error", "reason": "no_documents_indexed"})
        yield _sse({"type": "done"})
        return

    top_k = min(body.top_k or settings.top_k, indexed)

    try:
        chunks = await retrieve(http, settings, collection, body.question, top_k)
    except httpx.HTTPError as exc:
        logger.exception("retrieval failed")
        yield _sse({"type": "error", "reason": f"retrieval_failed:{type(exc).__name__}"})
        yield _sse({"type": "done"})
        return

    # Emit sources first. The client uses these to render citations
    # while the model is still generating.
    sources_payload = [
    {
        "id": f"{c.source}#{c.chunk_index}",
        "preview": c.text[:120],
    }
    for c in chunks
]
    yield _sse({"type": "sources", "sources": sources_payload})

    # Build the prompt the same way the non-streaming path does, then
    # call Ollama with stream: true. We re-implement the call here
    # rather than refactoring retriever.generate_answer because the
    # generator yields rather than returns; mixing the two control
    # flows would muddy both.
    prompt = _build_prompt(body.question, chunks)

    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": settings.temperature,
            "num_ctx": settings.num_ctx,
        },
    }

    started = time.perf_counter()
    try:
        async with http.stream(
            "POST",
            f"{settings.ollama_host}/api/generate",
            json=payload,
            timeout=httpx.Timeout(60.0, read=None),
        ) as response:
            if response.status_code >= 400:
                detail = await response.aread()
                logger.error(
                    "ollama generate failed: status=%s body=%s",
                    response.status_code,
                    detail[:500],
                )
                yield _sse(
                    {"type": "error", "reason": f"ollama_status_{response.status_code}"},
                )
                yield _sse({"type": "done"})
                return

            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    # Ollama occasionally emits non-JSON during keepalive;
                    # ignore rather than fail the whole stream.
                    continue
                text = chunk.get("response", "")
                if text:
                    yield _sse({"type": "token", "text": text})
                if chunk.get("done"):
                    break
    except (httpx.HTTPError, asyncio.CancelledError) as exc:
        logger.exception("streaming generation failed")
        yield _sse(
            {"type": "error", "reason": f"generation_failed:{type(exc).__name__}"},
        )
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("stream completed in %d ms (sources=%d)", elapsed_ms, len(chunks))
        yield _sse({"type": "done"})


def _build_prompt(question: str, chunks) -> str:
    """The same RAG prompt shape as the non-streaming path, kept here so
    the streaming endpoint is independent of retriever's private helpers.
    A small duplication is preferable to leaking generator concerns
    into the synchronous code path."""
    context_blocks = "\n\n".join(
        f"[{i + 1}] (source: {c.source}) {c.text}" for i, c in enumerate(chunks)
    )
    return (
        "You are Ramone, the live infrastructure assistant for Atlas Systems. "
        "Answer the user's question using only the context blocks below. "
        "If the answer is not in the context, say so plainly and suggest where on "
        "atlas-systems.uk they might find it. Cite the block number in square "
        "brackets when you use information from a block.\n\n"
        f"Context:\n{context_blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


@router.post("/ask/stream")
async def ask_stream(body: AskStreamRequest, request: Request) -> StreamingResponse:
    """Streaming Q&A endpoint consumed by ramone-edge.

    The Worker translates this to the browser one-to-one, so any header
    or body change here is a contract change on the public API.
    """
    app = request.app
    settings: Settings = app.state.settings
    collection = app.state.collection
    http: httpx.AsyncClient = app.state.http

    if collection is None:
        raise HTTPException(status_code=503, detail="collection_unavailable")

    return StreamingResponse(
        _stream_answer(http, settings, collection, body),
        media_type="text/event-stream",
        headers={
            # Disable buffering for proxies that look at these headers.
            "cache-control": "no-cache, no-transform",
            "x-accel-buffering": "no",
            "connection": "keep-alive",
        },
    )
