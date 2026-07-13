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
import re
import time
from pathlib import PurePosixPath
from typing import AsyncIterator

import httpx
from chromadb.api.models.Collection import Collection
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings
from app.memory import (
    append_turn,
    get_memory_collection,
    load_history,
    turns_to_messages,
    turns_to_prompt_question,
    turns_to_retrieval_query,
    validate_session_id,
)
from app.retriever import CorpusUnreachable, public_boundary_refusal, retrieve

logger = logging.getLogger(__name__)

router = APIRouter()

_SOURCE_LABEL_OVERRIDES = {
    "cicd": "CI/CD",
    "sonin": "SONIN",
    "slampunk": "SlamPunk",
}
_SOURCE_PREFIX_RE = re.compile(r"^\d+\s*[-_\s]*")
_SOURCE_WORD_RE = re.compile(r"[^a-z0-9]+")

STREAM_SYSTEM_PROMPT = (
    "You are Ramone, the live infrastructure assistant for Atlas Systems. "
    "Answer the user's question using only the numbered context blocks "
    "provided. Cite the blocks you used, like [1] or [2][3]. If the "
    "context does not contain the answer, say so plainly and suggest where "
    "on atlas-systems.uk they might find it. Keep answers concise and "
    "factual. Earlier turns in this conversation may be included before "
    "the current question; use them only to resolve references such as "
    "'it' or 'that project', never as a substitute for the numbered "
    "context blocks. Do not reuse prior formatting, style instructions, "
    "output schemas, or constraints from earlier turns unless the current "
    "user message explicitly asks for them. Earlier turns are for resolving "
    "references only, such as pronouns or 'your previous answer,' never for "
    "carrying forward how a past answer was structured."
)


class AskStreamRequest(BaseModel):
    """A streaming question. Same contract as AskRequest, separate class
    so future divergence does not break the non-streaming endpoint."""

    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str | None = None


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
    refusal = public_boundary_refusal(body.question)
    if refusal:
        yield _sse({"type": "sources", "sources": []})
        yield _sse({"type": "token", "text": refusal})
        yield _sse({"type": "done"})
        return

    # Retrieval lives in atlas-corpus now; corpus size and emptiness
    # are its concern. The collection argument stays in the signature
    # so the route wiring is untouched, but nothing reads it here any
    # more; conversation memory manages its own collection.
    top_k = body.top_k or settings.top_k
    session_id = validate_session_id(body.session_id)
    memory_collection: Collection | None = None
    history_messages: list[dict[str, str]] = []
    prompt_question = body.question
    retrieval_question = body.question

    if session_id and settings.memory_context_turns > 0:
        try:
            memory_collection = get_memory_collection(settings)
            turns = await load_history(settings, memory_collection, session_id)
            history_messages = turns_to_messages(turns)
            prompt_question = turns_to_prompt_question(body.question, turns)
            retrieval_question = turns_to_retrieval_query(body.question, turns)
        except Exception:  # noqa: BLE001 - memory must never break Q&A
            logger.exception("failed to load memory for session=%s", session_id)
            session_id = None
            memory_collection = None

    try:
        chunks = await retrieve(http, settings, retrieval_question, top_k)
    except CorpusUnreachable as exc:
        logger.error("atlas-corpus unreachable: %s", exc)
        yield _sse(
            {
                "type": "error",
                "reason": "retrieval_failed:CorpusUnreachable",
                "detail": (
                    "the corpus index on SPECULAR-CORE is not answering; "
                    "the Lab status panels at atlas-systems.uk/lab show "
                    "when it is back"
                ),
            }
        )
        yield _sse({"type": "done"})
        return
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

    prompt_subjects = _source_subjects_for_prompt(prompt_question, chunks)
    final_question = _question_with_resolved_subjects(body.question, prompt_subjects)

    messages = [{"role": "system", "content": STREAM_SYSTEM_PROMPT}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": _build_prompt(final_question, chunks)})

    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": settings.temperature,
            "num_ctx": settings.num_ctx,
        },
    }

    started = time.perf_counter()
    full_answer = ""
    completed = False
    try:
        async with http.stream(
            "POST",
            f"{settings.ollama_host}/api/chat",
            json=payload,
            timeout=httpx.Timeout(60.0, read=None),
        ) as response:
            if response.status_code >= 400:
                detail = await response.aread()
                logger.error(
                    "ollama chat failed: status=%s body=%s",
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
                message = chunk.get("message") or {}
                text = message.get("content", "")
                if text:
                    full_answer += text
                    yield _sse({"type": "token", "text": text})
                if chunk.get("done"):
                    completed = True
                    break
        if session_id and memory_collection is not None and completed:
            try:
                await append_turn(
                    http, settings, memory_collection, session_id, "user", body.question
                )
                await append_turn(
                    http, settings, memory_collection, session_id, "assistant", full_answer
                )
            except Exception:  # noqa: BLE001 - memory is additive
                logger.exception("failed to persist memory for session=%s", session_id)
    except (httpx.HTTPError, asyncio.CancelledError) as exc:
        logger.exception("streaming generation failed")
        yield _sse(
            {"type": "error", "reason": f"generation_failed:{type(exc).__name__}"},
        )
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("stream completed in %d ms (sources=%d)", elapsed_ms, len(chunks))
        yield _sse({"type": "done"})


def _source_label(source: str) -> str:
    normalized = source.replace("\\", "/")
    stem = PurePosixPath(normalized).stem
    stem = _SOURCE_PREFIX_RE.sub("", stem).strip("-_ ")
    if stem.lower() in ("readme", "index") and "/" in normalized:
        stem = normalized.split("/", 1)[0]
    key = _SOURCE_WORD_RE.sub("", stem.lower())
    if key in _SOURCE_LABEL_OVERRIDES:
        return _SOURCE_LABEL_OVERRIDES[key]
    label = re.sub(r"[-_]+", " ", stem).strip()
    return " ".join(label.split())


def _source_subjects_for_prompt(question: str, chunks) -> list[str]:
    question_key = _SOURCE_WORD_RE.sub("", question.lower())
    subjects: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        label = _source_label(str(chunk.source))
        key = _SOURCE_WORD_RE.sub("", label.lower())
        if not key or key in seen or key not in question_key:
            continue
        subjects.append(label)
        seen.add(key)
    return subjects


def _format_subjects(subjects: list[str]) -> str:
    if len(subjects) == 1:
        return subjects[0]
    return f"{', '.join(subjects[:-1])} and {subjects[-1]}"


def _question_with_resolved_subjects(question: str, subjects: list[str]) -> str:
    if not subjects:
        return question

    subject_text = _format_subjects(subjects)
    rewritten = re.sub(
        r"\b(them|they|those|it|this|that)\b",
        subject_text,
        question,
        flags=re.IGNORECASE,
    )
    if rewritten != question:
        return rewritten
    return f"{question}\n\nResolved subject(s): {subject_text}."


def _build_prompt(question: str, chunks) -> str:
    """The same RAG prompt shape as the non-streaming path, kept here so
    the streaming endpoint is independent of retriever's private helpers.
    A small duplication is preferable to leaking generator concerns
    into the synchronous code path."""
    context_blocks = "\n\n".join(
        f"[{i + 1}] (source: {c.source}) {c.text}" for i, c in enumerate(chunks)
    )
    return f"Context:\n{context_blocks}\n\nQuestion: {question}"


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
