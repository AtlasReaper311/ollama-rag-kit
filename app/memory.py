"""Cross-session conversation memory for Ramone.

Every turn of a conversation (the user's question, Ramone's answer) is
stored as its own document in a dedicated Chroma collection, keyed by a
client-generated session_id. On each new question, the most recent
turns for that session are loaded and spliced into the Ollama messages
array ahead of the current, RAG-grounded question, giving Ramone real
conversational continuity instead of answering every question cold.

The collection is deliberately separate from the document collection
(settings.collection_name). Mixing conversation turns into the document
index would let a user's own earlier question get retrieved back as
"evidence" for a later answer, which is a correctness bug, not just an
untidy schema.

Nothing here runs on a schedule. Growth is bounded two ways instead:
  - a hard cap per session (memory_max_turns), enforced on every write
  - a TTL (memory_ttl_seconds), enforced lazily on every read

That is the only reaper a single-instance, zero-cost service needs.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import chromadb
import httpx
from chromadb.api.models.Collection import Collection
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.ingest import embed_texts

logger = logging.getLogger(__name__)

# UUID v4 only. A malformed session_id is rejected outright rather than
# sanitised. By the time any other function in this module sees one,
# it is guaranteed well-formed, so nothing downstream re-validates.
_SESSION_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_REFERENCE_CUE_RE = re.compile(
    r"\b("
    r"it|its|that|this|these|those|they|them|their|he|him|his|"
    r"she|her|hers|previous|earlier|above|last|former|latter|"
    r"first|second|third|same|also|too|again|more|followup|follow-up"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class Turn:
    """One stored turn: either the user's question or Ramone's answer."""

    role: str  # "user" | "assistant"
    content: str
    turn_index: int
    created_at: float


def validate_session_id(raw: str | None) -> str | None:
    """Return a normalised session_id, or None if raw is missing/malformed.

    Called at the API boundary so nothing downstream ever has to
    re-validate: by the time the rest of this module sees a session_id,
    it is guaranteed to be a well-formed UUID v4 string.
    """
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lower()
    if not _SESSION_ID_RE.match(candidate):
        return None
    return candidate


def get_memory_collection(settings: Settings) -> Collection:
    """Connect to (or create) the dedicated conversation-memory collection.

    A second, independent HttpClient rather than reusing however
    main.py connects to the document collection. Chroma's HttpClient is
    a thin REST wrapper with no persistent socket to manage, so a
    second instance costs nothing meaningful and keeps this module
    decoupled from main.py's internals.
    """
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return client.get_or_create_collection(
        name=settings.memory_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def question_needs_history(question: str) -> bool:
    """Return True when a question likely depends on prior turns.

    Conversation history is only for reference resolution. Injecting it
    into every request gives old one-off formatting instructions another
    chance to influence unrelated answers. This deliberately simple
    detector keeps common follow-up shapes while leaving standalone
    questions as clean RAG calls.
    """
    if not isinstance(question, str):
        return False
    q = question.strip().lower()
    if not q:
        return False
    if _REFERENCE_CUE_RE.search(q):
        return True
    return any(
        phrase in q
        for phrase in (
            "your previous answer",
            "the previous answer",
            "your last answer",
            "the last answer",
            "what about",
            "how about",
            "compare that",
            "compare this",
            "how does that compare",
            "how does this compare",
            "what is the difference",
            "what's the difference",
            "how is that different",
            "how is this different",
            "same as",
            "tell me more",
            "go on",
            "continue",
        )
    )


async def load_history(
    settings: Settings,
    collection: Collection,
    session_id: str,
) -> list[Turn]:
    """Return the most recent turns for a session, oldest first.

    Skips the Chroma round trip entirely when memory_context_turns is
    0, the documented killswitch for turning memory off in production
    with an env var flip, no redeploy required.

    Chroma's .get() has no server-side ORDER BY, so turns are sorted
    client-side by turn_index. Fine at this scale: memory_max_turns
    caps any one session at a couple of dozen rows, worst case.

    Turns older than memory_ttl_seconds are treated as if they do not
    exist: lazy expiry, no scheduled job required.
    """
    if settings.memory_context_turns <= 0:
        return []

    result = await run_in_threadpool(
        collection.get,
        where={"session_id": session_id},
        include=["documents", "metadatas"],
    )

    cutoff = time.time() - settings.memory_ttl_seconds
    turns: list[Turn] = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        created_at = float(meta.get("created_at", 0))
        if created_at < cutoff:
            continue
        turns.append(
            Turn(
                role=str(meta.get("role", "user")),
                content=doc,
                turn_index=int(meta.get("turn_index", 0)),
                created_at=created_at,
            )
        )

    turns.sort(key=lambda t: t.turn_index)
    return turns[-settings.memory_context_turns :]


def turns_to_messages(turns: list[Turn]) -> list[dict[str, str]]:
    """Convert stored Turns into a reference-only history block.

    Prior user messages may contain one-off formatting instructions or
    prompt-injection attempts. Replaying them as live ``role=user``
    messages lets the model keep obeying old constraints on unrelated
    questions. A quoted system-side history block preserves enough
    continuity for pronouns and "your previous answer" without granting
    past turns fresh instruction authority.
    """
    assistant_turns = [turn for turn in turns if turn.role == "assistant"]
    if not assistant_turns:
        return []

    lines = [
        "Reference-only previous assistant answers follow.",
        "Use them only to resolve references in the current user message, "
        "such as pronouns or 'your previous answer.'",
        "Do not copy their formatting, style, output schemas, or labels "
        "unless the current user message explicitly asks for that format.",
        "",
    ]
    for turn in assistant_turns:
        lines.append(f"Previous assistant answer: {turn.content}")

    return [{"role": "system", "content": "\n".join(lines)}]


async def append_turn(
    http: httpx.AsyncClient,
    settings: Settings,
    collection: Collection,
    session_id: str,
    role: str,
    content: str,
) -> None:
    """Embed and store one turn, then prune the session back to the cap.

    Called after generation completes, not per-token: the assistant's
    turn is not a coherent unit of content until the answer is
    finished, and re-embedding a growing partial string on every token
    would waste GPU cycles for zero benefit.
    """
    if settings.memory_context_turns <= 0:
        return

    content = content[: settings.memory_max_message_chars]
    next_index = await _next_turn_index(collection, session_id)
    embedding = (await embed_texts(http, settings, [content]))[0]

    await run_in_threadpool(
        collection.upsert,
        ids=[f"{session_id}:{next_index}"],
        embeddings=[embedding],
        documents=[content],
        metadatas=[
            {
                "session_id": session_id,
                "role": role,
                "turn_index": next_index,
                "created_at": time.time(),
            }
        ],
    )

    await _prune_session(collection, settings, session_id)


async def _next_turn_index(collection: Collection, session_id: str) -> int:
    result = await run_in_threadpool(
        collection.get, where={"session_id": session_id}, include=["metadatas"]
    )
    if not result["metadatas"]:
        return 0
    return max(int(m.get("turn_index", 0)) for m in result["metadatas"]) + 1


async def _prune_session(collection: Collection, settings: Settings, session_id: str) -> None:
    """Delete the oldest turns once a session exceeds memory_max_turns.

    Bounds worst-case storage per session regardless of how long a
    single visitor keeps talking to Ramone, independent of the TTL
    reaper above, which only catches abandoned sessions.
    """
    result = await run_in_threadpool(
        collection.get, where={"session_id": session_id}, include=["metadatas"]
    )
    ids = result["ids"]
    metas = result["metadatas"]
    if len(ids) <= settings.memory_max_turns:
        return

    paired = sorted(zip(ids, metas), key=lambda p: int(p[1].get("turn_index", 0)))
    overflow = len(paired) - settings.memory_max_turns
    stale_ids = [doc_id for doc_id, _ in paired[:overflow]]
    await run_in_threadpool(collection.delete, ids=stale_ids)
    logger.info("memory pruned session=%s removed=%d", session_id, len(stale_ids))
