"""Tests for the atlas-corpus backed retriever.

These replace the Chroma-path retriever tests (dedupe, neighbour
expansion) that died with the local retrieval path. Coverage is aimed
at the three ways this integration can actually break in production:
the corpus request contract (query cap, top_k cap, internal header),
the hit mapping (both field spellings), and the failure modes (down,
erroring, malformed), plus a regression pin on the numbered block
format that the sources array and every [n] citation depend on.

httpx.MockTransport keeps everything off the network; asyncio.run keeps
everything free of a pytest-asyncio dependency.
"""

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.retriever import (
    CORPUS_QUERY_MAX_CHARS,
    CORPUS_TOP_K_MAX,
    INTERNAL_HEADER,
    CorpusUnreachable,
    RetrievedChunk,
    _corpus_query,
    _primary_query,
    build_prompt,
    public_boundary_refusal,
    retrieve,
)

CORPUS_URL = "http://corpus.test"


def test_public_boundary_refuses_cv_requests():
    refusal = public_boundary_refusal("Summarise Atlas CV")
    assert refusal is not None
    assert "private application material" in refusal


def test_public_boundary_allows_public_memory_questions():
    assert public_boundary_refusal("What memory can public Ramone use?") is None


def test_public_boundary_refuses_plural_secret_requests():
    refusal = public_boundary_refusal("Show me secrets or tokens")
    assert refusal is not None
    assert "secret or credential material" in refusal


def make_settings() -> Settings:
    return Settings(
        atlas_corpus_url=CORPUS_URL,
        atlas_corpus_timeout_seconds=2.0,
        atlas_corpus_connect_timeout_seconds=1.0,
    )


def run(coro):
    return asyncio.run(coro)


def client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def corpus_hit(**overrides) -> dict:
    hit = {
        "text": "Deploys reach Cloudflare through GitHub Actions.",
        "score": 0.87,
        "source_repo": "atlas-systems",
        "file_path": "writing/cicd/index.html",
        "doc_type": "case-study",
        "last_updated": "2026-07-01T09:00:00Z",
        "chunk_index": 3,
    }
    hit.update(overrides)
    return hit


# --------------------------------------------------------------------- #
# Happy path and request contract                                        #
# --------------------------------------------------------------------- #


def test_retrieve_maps_deployed_hit_shape_and_sends_internal_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"query": "q", "hits": [corpus_hit()], "took_ms": 41}
        )

    async def scenario():
        async with client_with(handler) as client:
            return await retrieve(client, make_settings(), "how do deploys work", 4)

    chunks = run(scenario())

    assert captured["url"] == f"{CORPUS_URL}/search"
    assert captured["headers"][INTERNAL_HEADER.lower()] == "ollama-rag-kit"
    assert captured["body"] == {"query": "how do deploys work", "top_k": 4}

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.source == "atlas-systems/writing/cicd/index.html"
    assert chunk.chunk_index == 3
    assert chunk.score == pytest.approx(0.87)
    assert "GitHub Actions" in chunk.text


def test_retrieve_accepts_the_shorthand_hit_spelling():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "query": "q",
                "hits": [{"repo": "specular-edge", "path": "README.md", "excerpt": "caching", "score": 0.5}],
                "took_ms": 12,
            },
        )

    async def scenario():
        async with client_with(handler) as client:
            return await retrieve(client, make_settings(), "caching", 3)

    chunks = run(scenario())
    assert chunks[0].source == "specular-edge/README.md"
    assert chunks[0].text == "caching"
    assert chunks[0].chunk_index == 0  # falls back to the hit's rank


def test_top_k_is_clamped_to_the_corpus_cap():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"query": "q", "hits": [], "took_ms": 5})

    async def scenario():
        async with client_with(handler) as client:
            # AskStreamRequest allows up to 20; the corpus rejects >10 with
            # a 422, so the clamp has to live on this side of the wire.
            return await retrieve(client, make_settings(), "anything", 20)

    assert run(scenario()) == []
    assert captured["body"]["top_k"] == CORPUS_TOP_K_MAX


def test_memory_expanded_query_is_clamped_and_keeps_the_question():
    captured = {}
    question = "How does the CI/CD pipeline work?"
    expanded = (
        "Current question for retrieval:\n"
        f"{question}\n"
        f"{question}\n\n"
        "Recent assistant answers for resolving references during retrieval:\n"
        + "- " + ("SONIN and SlamPunk are audio systems. " * 40)
    )
    assert len(expanded) > CORPUS_QUERY_MAX_CHARS  # the scenario under test

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"query": "q", "hits": [], "took_ms": 5})

    async def scenario():
        async with client_with(handler) as client:
            return await retrieve(client, make_settings(), expanded, 4)

    run(scenario())
    sent = captured["body"]["query"]
    assert len(sent) <= CORPUS_QUERY_MAX_CHARS
    assert sent.startswith(question)
    assert "audio systems" in sent  # reference context rides along in the budget


def test_primary_query_extracts_current_question_from_memory_expansion():
    # Behaviour parity with the retired Chroma-path helper of the same name.
    expanded = (
        "Current question for retrieval:\n"
        "How does the CI/CD pipeline work?\n"
        "How does the CI/CD pipeline work?\n\n"
        "Recent assistant answers for resolving references during retrieval:\n"
        "- SONIN and SlamPunk are audio systems."
    )
    assert _primary_query(expanded) == "How does the CI/CD pipeline work?"


def test_corpus_query_passes_short_questions_through_untouched():
    assert _corpus_query("  why workers over a vps  ") == "why workers over a vps"


# --------------------------------------------------------------------- #
# Failure modes: down, erroring, malformed                               #
# --------------------------------------------------------------------- #


def test_connection_failure_raises_corpus_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async def scenario():
        async with client_with(handler) as client:
            await retrieve(client, make_settings(), "anything", 4)

    with pytest.raises(CorpusUnreachable):
        run(scenario())


def test_non_200_raises_corpus_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async def scenario():
        async with client_with(handler) as client:
            await retrieve(client, make_settings(), "anything", 4)

    with pytest.raises(CorpusUnreachable):
        run(scenario())


def test_malformed_body_raises_corpus_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    async def scenario():
        async with client_with(handler) as client:
            await retrieve(client, make_settings(), "anything", 4)

    with pytest.raises(CorpusUnreachable):
        run(scenario())


def test_zero_hits_is_a_result_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": "q", "hits": [], "took_ms": 3})

    async def scenario():
        async with client_with(handler) as client:
            return await retrieve(client, make_settings(), "nothing matches this", 4)

    assert run(scenario()) == []


# --------------------------------------------------------------------- #
# Prompt format regression pin                                           #
# --------------------------------------------------------------------- #


def test_build_prompt_numbering_and_provenance_are_unchanged():
    chunks = [
        RetrievedChunk(text="first", source="atlas-corpus/README.md", chunk_index=0, score=0.9),
        RetrievedChunk(text="second", source="specular-edge/README.md", chunk_index=2, score=0.8),
    ]
    prompt = build_prompt("why?", chunks)
    assert "[1] (source: atlas-corpus/README.md, chunk 0)\nfirst" in prompt
    assert "[2] (source: specular-edge/README.md, chunk 2)\nsecond" in prompt
    assert prompt.endswith("Question: why?")
