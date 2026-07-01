import pytest

from app import retriever
from app.config import Settings


class FakeCollection:
    def __init__(
        self,
        documents=None,
        metadatas=None,
        distances=None,
        source_docs=None,
        query_responses=None,
    ):
        self.n_results = None
        self.get_sources = []
        self.query_count = 0
        self.documents = documents or [
            "About profile chunk with SONIN and SlamPunk.",
            "Duplicate about profile export with the same facts.",
            "Full stack overview chunk.",
            "SlamPunk detailed case-study chunk.",
            "FAQ chunk.",
        ]
        self.metadatas = metadatas or [
            {"source": "98_-_about.md", "chunk_index": 2},
            {"source": "98 - about.md", "chunk_index": 2},
            {"source": "13_-_full_stack.md", "chunk_index": 12},
            {"source": "06 - slampunk.md", "chunk_index": 1},
            {"source": "99_-_faq.md", "chunk_index": 3},
        ]
        self.distances = distances or [0.1, 0.11, 0.2, 0.3, 0.4]
        self.source_docs = source_docs or {}
        self.query_responses = query_responses or []

    def query(self, query_embeddings, n_results, include):
        self.n_results = n_results
        self.query_count += 1
        if self.query_responses:
            return self.query_responses.pop(0)
        return {
            "documents": [self.documents],
            "metadatas": [self.metadatas],
            "distances": [self.distances],
        }

    def get(self, where, include):
        source = where["source"]
        self.get_sources.append(source)
        rows = self.source_docs.get(source, [])
        return {
            "documents": [text for text, _meta in rows],
            "metadatas": [meta for _text, meta in rows],
        }


async def fake_embed_texts(client, settings, texts):
    return [[0.0, 0.0] for _ in texts]


def test_query_terms_normalize_cicd_slash_phrase():
    assert "cicd" in retriever._query_terms("How does the CI/CD pipeline work?")


def test_primary_query_extracts_current_question_from_memory_expansion():
    expanded = (
        "Current question for retrieval:\n"
        "How does the CI/CD pipeline work?\n"
        "How does the CI/CD pipeline work?\n\n"
        "Recent assistant answers for resolving references during retrieval:\n"
        "- SONIN and SlamPunk are audio systems."
    )

    assert retriever._primary_query(expanded) == "How does the CI/CD pipeline work?"


@pytest.mark.asyncio
async def test_retrieve_diversifies_duplicate_source_exports(monkeypatch):
    monkeypatch.setattr(retriever, "embed_texts", fake_embed_texts)
    collection = FakeCollection()

    chunks = await retriever.retrieve(
        client=None,
        settings=Settings(),
        collection=collection,
        question="tell me more about them",
        top_k=4,
    )

    assert collection.n_results == 32
    assert [chunk.source for chunk in chunks] == [
        "98_-_about.md",
        "13_-_full_stack.md",
        "06 - slampunk.md",
        "99_-_faq.md",
    ]


@pytest.mark.asyncio
async def test_retrieve_boosts_exact_source_name_matches(monkeypatch):
    monkeypatch.setattr(retriever, "embed_texts", fake_embed_texts)
    collection = FakeCollection(
        documents=[
            "Generic architectural reasoning overview with a strong embedding match.",
            "Another generic nearby chunk.",
            "Later SONIN implementation detail.",
            "Later SlamPunk implementation detail.",
        ],
        metadatas=[
            {"source": "13_-_full_stack.md", "chunk_index": 0},
            {"source": "98_-_about.md", "chunk_index": 1},
            {"source": "05_-_sonin.md", "chunk_index": 15},
            {"source": "06 - slampunk.md", "chunk_index": 20},
        ],
        distances=[0.01, 0.02, 0.25, 0.26],
        source_docs={
            "05_-_sonin.md": [
                (
                    "SONIN is an autonomous Max/MSP generative audio-visual system.",
                    {"source": "05_-_sonin.md", "chunk_index": 0},
                ),
            ],
            "06 - slampunk.md": [
                (
                    "SlamPunk uses a 15-stem dynamic audio engine.",
                    {"source": "06 - slampunk.md", "chunk_index": 0},
                ),
            ],
        },
    )

    chunks = await retriever.retrieve(
        client=None,
        settings=Settings(),
        collection=collection,
        question="Tell me about SONIN and SlamPunk.",
        top_k=2,
    )

    assert [chunk.source for chunk in chunks] == [
        "05_-_sonin.md",
        "06 - slampunk.md",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 0]
    assert collection.get_sources == ["05_-_sonin.md", "06 - slampunk.md"]


@pytest.mark.asyncio
async def test_retrieve_current_question_overrides_stale_memory(monkeypatch):
    monkeypatch.setattr(retriever, "embed_texts", fake_embed_texts)
    collection = FakeCollection(
        query_responses=[
            {
                "documents": [[
                    "SONIN is an autonomous digital instrument.",
                    "SlamPunk is a dynamic audio engine.",
                ]],
                "metadatas": [[
                    {"source": "05_-_sonin.md", "chunk_index": 0},
                    {"source": "06 - slampunk.md", "chunk_index": 0},
                ]],
                "distances": [[0.05, 0.06]],
            },
            {
                "documents": [[
                    "The CI/CD pipeline uses GitHub Actions and Cloudflare.",
                    "Worker deploys use wrangler dry-run and tests.",
                ]],
                "metadatas": [[
                    {"source": "10_- cicd.md", "chunk_index": 0},
                    {"source": "04 - decisions-public.md", "chunk_index": 3},
                ]],
                "distances": [[0.2, 0.21]],
            },
        ],
        source_docs={
            "10_- cicd.md": [
                (
                    "The CI/CD pipeline uses GitHub Actions and Cloudflare.",
                    {"source": "10_- cicd.md", "chunk_index": 0},
                ),
            ],
        },
    )
    expanded = (
        "Current question for retrieval:\n"
        "How does the CI/CD pipeline work?\n"
        "How does the CI/CD pipeline work?\n\n"
        "Recent assistant answers for resolving references during retrieval:\n"
        "- SONIN is an autonomous instrument. SlamPunk is a dynamic audio engine."
    )

    chunks = await retriever.retrieve(
        client=None,
        settings=Settings(),
        collection=collection,
        question=expanded,
        top_k=2,
    )

    assert collection.query_count == 2
    assert [chunk.source for chunk in chunks] == [
        "10_- cicd.md",
        "04 - decisions-public.md",
    ]
