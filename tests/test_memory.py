"""Tests for app.memory: session_id validation, history loading, and
turn pruning.

Uses a lightweight in-memory fake Collection rather than a running
Chroma server. This exercises the real filtering, sorting, TTL, and
pruning logic in memory.py without needing Docker up for the test run.

Requires pytest-asyncio (async def tests below). If it's not already a
dev dependency:

    pip install pytest pytest-asyncio --break-system-packages
"""

from __future__ import annotations

import time

import pytest

from app import memory
from app.config import Settings


class FakeCollection:
    """Minimal stand-in for chromadb.api.models.Collection.Collection.

    Implements exactly the three methods memory.py calls (get, upsert,
    delete) against a plain dict, keyed by document id.
    """

    def __init__(self):
        self._docs: dict[str, dict] = {}

    def get(self, where=None, include=None):
        session_id = (where or {}).get("session_id")
        items = [
            (doc_id, row)
            for doc_id, row in self._docs.items()
            if session_id is None or row["metadata"]["session_id"] == session_id
        ]
        return {
            "ids": [doc_id for doc_id, _ in items],
            "documents": [row["document"] for _, row in items],
            "metadatas": [row["metadata"] for _, row in items],
        }

    def upsert(self, ids, embeddings, documents, metadatas):
        for doc_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self._docs[doc_id] = {"embedding": embedding, "document": document, "metadata": metadata}

    def delete(self, ids):
        for doc_id in ids:
            self._docs.pop(doc_id, None)


@pytest.fixture
def settings():
    return Settings(
        memory_max_turns=4,
        memory_context_turns=6,
        memory_ttl_seconds=3600,
        memory_max_message_chars=4000,
    )


@pytest.fixture
def collection():
    return FakeCollection()


# validate_session_id


def test_accepts_valid_uuid_v4():
    assert memory.validate_session_id("110e8400-e29b-41d4-a716-446655440000") is not None


def test_rejects_none():
    assert memory.validate_session_id(None) is None


def test_rejects_non_uuid_string():
    assert memory.validate_session_id("not-a-uuid") is None


def test_rejects_uuid_v1_shape():
    # Version nibble must be 4; this is a plausible-looking but wrong shape.
    assert memory.validate_session_id("110e8400-e29b-11d4-a716-446655440000") is None


def test_normalises_to_lowercase():
    raw = "110E8400-E29B-41D4-A716-446655440000"
    assert memory.validate_session_id(raw) == raw.lower()


# load_history


@pytest.mark.asyncio
async def test_load_history_returns_empty_when_disabled(settings, collection):
    settings.memory_context_turns = 0
    result = await memory.load_history(settings, collection, "sid-1")
    assert result == []


@pytest.mark.asyncio
async def test_load_history_orders_oldest_first(settings, collection):
    now = time.time()
    collection._docs = {
        "sid-1:1": {
            "document": "second",
            "metadata": {"session_id": "sid-1", "role": "assistant", "turn_index": 1, "created_at": now},
        },
        "sid-1:0": {
            "document": "first",
            "metadata": {"session_id": "sid-1", "role": "user", "turn_index": 0, "created_at": now},
        },
    }
    result = await memory.load_history(settings, collection, "sid-1")
    assert [t.content for t in result] == ["first", "second"]


@pytest.mark.asyncio
async def test_load_history_ignores_other_sessions(settings, collection):
    now = time.time()
    collection._docs = {
        "sid-1:0": {
            "document": "mine",
            "metadata": {"session_id": "sid-1", "role": "user", "turn_index": 0, "created_at": now},
        },
        "sid-2:0": {
            "document": "not mine",
            "metadata": {"session_id": "sid-2", "role": "user", "turn_index": 0, "created_at": now},
        },
    }
    result = await memory.load_history(settings, collection, "sid-1")
    assert [t.content for t in result] == ["mine"]


@pytest.mark.asyncio
async def test_load_history_drops_expired_turns(settings, collection):
    settings.memory_ttl_seconds = 60
    stale = time.time() - 3600
    collection._docs = {
        "sid-1:0": {
            "document": "old",
            "metadata": {"session_id": "sid-1", "role": "user", "turn_index": 0, "created_at": stale},
        },
    }
    result = await memory.load_history(settings, collection, "sid-1")
    assert result == []


@pytest.mark.asyncio
async def test_load_history_caps_to_context_turns(settings, collection):
    settings.memory_context_turns = 2
    now = time.time()
    for i in range(5):
        collection._docs[f"sid-1:{i}"] = {
            "document": f"turn {i}",
            "metadata": {"session_id": "sid-1", "role": "user", "turn_index": i, "created_at": now},
        }
    result = await memory.load_history(settings, collection, "sid-1")
    assert [t.content for t in result] == ["turn 3", "turn 4"]


# append_turn + pruning


@pytest.mark.asyncio
async def test_append_turn_stores_document(settings, collection, monkeypatch):
    async def fake_embed(http, settings, texts):
        return [[0.0, 0.0] for _ in texts]

    monkeypatch.setattr(memory, "embed_texts", fake_embed)
    await memory.append_turn(None, settings, collection, "sid-1", "user", "hello")

    result = collection.get(where={"session_id": "sid-1"})
    assert result["documents"] == ["hello"]
    assert result["metadatas"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_append_turn_skips_when_memory_disabled(settings, collection, monkeypatch):
    async def fake_embed(http, settings, texts):
        raise AssertionError("disabled memory should not embed")

    monkeypatch.setattr(memory, "embed_texts", fake_embed)
    settings.memory_context_turns = 0

    await memory.append_turn(None, settings, collection, "sid-1", "user", "hello")

    result = collection.get(where={"session_id": "sid-1"})
    assert result["ids"] == []


@pytest.mark.asyncio
async def test_append_turn_prunes_oldest_over_cap(settings, collection, monkeypatch):
    async def fake_embed(http, settings, texts):
        return [[0.0, 0.0] for _ in texts]

    monkeypatch.setattr(memory, "embed_texts", fake_embed)
    settings.memory_max_turns = 3

    for i in range(5):
        await memory.append_turn(None, settings, collection, "sid-1", "user", f"turn {i}")

    remaining = collection.get(where={"session_id": "sid-1"})
    assert len(remaining["ids"]) == 3
    kept = sorted(int(m["turn_index"]) for m in remaining["metadatas"])
    assert kept == [2, 3, 4]


def test_turns_to_messages_shape():
    turns = [
        memory.Turn(role="user", content="use bullets", turn_index=0, created_at=0.0),
        memory.Turn(role="assistant", content="hi", turn_index=1, created_at=0.0),
    ]
    messages = memory.turns_to_messages(turns)
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert "Reference-only previous assistant answers" in messages[0]["content"]
    assert "Previous assistant answer: hi" in messages[0]["content"]
    assert "use bullets" not in messages[0]["content"]


def test_turns_to_messages_returns_empty_without_history():
    assert memory.turns_to_messages([]) == []


def test_turns_to_messages_returns_empty_without_assistant_turns():
    turns = [memory.Turn(role="user", content="hi", turn_index=0, created_at=0.0)]
    assert memory.turns_to_messages(turns) == []


def test_question_needs_history_detects_referential_followup():
    assert memory.question_needs_history("What does it do?") is True
    assert memory.question_needs_history("In your previous answer, what was second?") is True


def test_question_needs_history_rejects_standalone_question():
    assert memory.question_needs_history("How does the CI/CD pipeline work?") is False
