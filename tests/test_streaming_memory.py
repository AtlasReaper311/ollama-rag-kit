"""Integration-style tests for streaming.py's use of conversation memory.

These tests keep Chroma and Ollama mocked, but exercise the real
_stream_answer orchestration: history loading, /api/chat payload
construction, SSE output, and post-generation memory writes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app import streaming
from app.config import Settings
from app.memory import Turn


VALID_SESSION = "110e8400-e29b-41d4-a716-446655440000"


class FakeDocumentCollection:
    def count(self):
        return 1


class FakeChatResponse:
    def __init__(self, lines=None, status_code=200, body=b""):
        self.lines = lines or []
        self.status_code = status_code
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aread(self):
        return self.body


class FakeHttp:
    def __init__(self, response):
        self.response = response
        self.payload = None

    def stream(self, method, url, json, timeout):  # noqa: A002 - mirrors httpx
        self.payload = {"method": method, "url": url, "json": json, "timeout": timeout}
        return self.response


def _decode_sse(chunks):
    events = []
    for chunk in chunks:
        text = chunk.decode("utf-8").strip()
        if not text.startswith("data: "):
            continue
        events.append(json.loads(text.removeprefix("data: ")))
    return events


@pytest.mark.asyncio
async def test_streaming_injects_history_into_chat_payload_and_writes_completed_turns(
    monkeypatch,
):
    settings = Settings(memory_context_turns=6, top_k=4)
    memory_collection = object()
    writes = []

    async def fake_load_history(settings, collection, session_id):
        assert collection is memory_collection
        assert session_id == VALID_SESSION
        return [
            Turn(
                role="user",
                content="What is the second component?",
                turn_index=0,
                created_at=0.0,
            ),
            Turn(
                role="assistant",
                content="Second: ChromaDB.",
                turn_index=1,
                created_at=0.0,
            ),
        ]

    async def fake_retrieve(http, settings, collection, question, top_k):
        return [
            SimpleNamespace(
                source="doc.md",
                chunk_index=0,
                text="ChromaDB stores embeddings for retrieval.",
            )
        ]

    async def fake_append_turn(http, settings, collection, session_id, role, content):
        writes.append((collection, session_id, role, content))

    monkeypatch.setattr(streaming, "get_memory_collection", lambda settings: memory_collection)
    monkeypatch.setattr(streaming, "load_history", fake_load_history)
    monkeypatch.setattr(streaming, "retrieve", fake_retrieve)
    monkeypatch.setattr(streaming, "append_turn", fake_append_turn)

    http = FakeHttp(
        FakeChatResponse(
            lines=[
                json.dumps({"message": {"content": "It stores "}, "done": False}),
                json.dumps({"message": {"content": "embeddings."}, "done": True}),
            ]
        )
    )
    body = streaming.AskStreamRequest(
        question="What does it do?",
        session_id=VALID_SESSION,
    )

    chunks = [
        chunk
        async for chunk in streaming._stream_answer(
            http, settings, FakeDocumentCollection(), body
        )
    ]
    events = _decode_sse(chunks)

    messages = http.payload["json"]["messages"]
    assert messages[1]["role"] == "system"
    assert "Reference-only previous assistant answers" in messages[1]["content"]
    assert "What is the second component?" not in messages[1]["content"]
    assert "Previous assistant answer: Second: ChromaDB." in messages[1]["content"]
    assert messages[-1]["role"] == "user"
    assert "Question: What does it do?" in messages[-1]["content"]
    assert [event["type"] for event in events] == ["sources", "token", "token", "done"]
    assert writes == [
        (memory_collection, VALID_SESSION, "user", "What does it do?"),
        (memory_collection, VALID_SESSION, "assistant", "It stores embeddings."),
    ]


@pytest.mark.asyncio
async def test_streaming_does_not_write_memory_when_generation_fails(monkeypatch):
    settings = Settings(memory_context_turns=6, top_k=4)
    memory_collection = object()
    writes = []

    async def fake_load_history(settings, collection, session_id):
        return []

    async def fake_retrieve(http, settings, collection, question, top_k):
        return [
            SimpleNamespace(
                source="doc.md",
                chunk_index=0,
                text="ChromaDB stores embeddings for retrieval.",
            )
        ]

    async def fake_append_turn(http, settings, collection, session_id, role, content):
        writes.append((role, content))

    monkeypatch.setattr(streaming, "get_memory_collection", lambda settings: memory_collection)
    monkeypatch.setattr(streaming, "load_history", fake_load_history)
    monkeypatch.setattr(streaming, "turns_to_messages", lambda turns: [])
    monkeypatch.setattr(streaming, "retrieve", fake_retrieve)
    monkeypatch.setattr(streaming, "append_turn", fake_append_turn)

    http = FakeHttp(FakeChatResponse(status_code=500, body=b"model unavailable"))
    body = streaming.AskStreamRequest(
        question="What does it do?",
        session_id=VALID_SESSION,
    )

    chunks = [
        chunk
        async for chunk in streaming._stream_answer(
            http, settings, FakeDocumentCollection(), body
        )
    ]
    events = _decode_sse(chunks)

    assert events[0]["type"] == "sources"
    assert events[1] == {"type": "error", "reason": "ollama_status_500"}
    assert events[-1]["type"] == "done"
    assert writes == []


@pytest.mark.asyncio
async def test_streaming_always_injects_history_for_valid_session(monkeypatch):
    settings = Settings(memory_context_turns=6, top_k=4)
    memory_collection = object()
    writes = []

    async def fake_load_history(settings, collection, session_id):
        assert collection is memory_collection
        assert session_id == VALID_SESSION
        return [
            Turn(
                role="user",
                content="Use only First and Second labels.",
                turn_index=0,
                created_at=0.0,
            ),
            Turn(
                role="assistant",
                content="First: Ollama. Second: ChromaDB.",
                turn_index=1,
                created_at=0.0,
            ),
        ]

    async def fake_retrieve(http, settings, collection, question, top_k):
        return [
            SimpleNamespace(
                source="doc.md",
                chunk_index=0,
                text="The CI/CD pipeline uses reusable workflows.",
            )
        ]

    async def fake_append_turn(http, settings, collection, session_id, role, content):
        writes.append((role, content))

    monkeypatch.setattr(streaming, "get_memory_collection", lambda settings: memory_collection)
    monkeypatch.setattr(streaming, "load_history", fake_load_history)
    monkeypatch.setattr(streaming, "retrieve", fake_retrieve)
    monkeypatch.setattr(streaming, "append_turn", fake_append_turn)

    http = FakeHttp(
        FakeChatResponse(
            lines=[json.dumps({"message": {"content": "Pipeline answer."}, "done": True})]
        )
    )
    body = streaming.AskStreamRequest(
        question="How does the CI/CD pipeline work?",
        session_id=VALID_SESSION,
    )

    chunks = [
        chunk
        async for chunk in streaming._stream_answer(
            http, settings, FakeDocumentCollection(), body
        )
    ]
    events = _decode_sse(chunks)

    messages = http.payload["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "system"
    assert "Reference-only previous assistant answers" in messages[1]["content"]
    assert "First: Ollama. Second: ChromaDB." in messages[1]["content"]
    assert "Use only First and Second labels." not in messages[1]["content"]
    assert messages[2]["role"] == "user"
    assert "Question: How does the CI/CD pipeline work?" in messages[2]["content"]
    assert [event["type"] for event in events] == ["sources", "token", "done"]
    assert writes == [
        ("user", "How does the CI/CD pipeline work?"),
        ("assistant", "Pipeline answer."),
    ]
