"""Retrieval and answer generation.

retrieve() finds the most relevant chunks; build_prompt() assembles them
into numbered context blocks; generate_answer() asks the LLM to answer
strictly from that context, optionally continuing a remembered
conversation. Keeping the steps separate makes each one swappable: a
reranker can slot in after retrieve(), a different prompt strategy only
touches build_prompt(), and memory is just another input to
generate_answer() rather than a rewrite of it.
"""

import logging
import re
from dataclasses import dataclass

import httpx
from chromadb.api.models.Collection import Collection
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.ingest import embed_texts

logger = logging.getLogger(__name__)

_SOURCE_KEY_RE = re.compile(r"[^a-z0-9]+")
_QUERY_TERM_RE = re.compile(r"[a-z0-9][a-z0-9/_-]+")
_QUERY_STOPWORDS = {
    "about",
    "answer",
    "assistant",
    "briefly",
    "context",
    "current",
    "does",
    "from",
    "history",
    "more",
    "previous",
    "question",
    "recent",
    "reference",
    "references",
    "resolving",
    "tell",
    "them",
    "these",
    "this",
    "turns",
    "user",
    "what",
    "with",
}

# The system prompt is the contract that makes this RAG rather than
# open-ended chat: answer from context, cite by block number, admit when
# the documents do not contain the answer.
SYSTEM_PROMPT = (
    "You are a retrieval assistant for a private document collection. "
    "Answer the user's question using ONLY the numbered context blocks "
    "provided. Cite the blocks you used, like [1] or [2][3]. If the "
    "context does not contain the answer, say so plainly instead of "
    "guessing. Keep answers concise and factual. Earlier turns in this "
    "conversation may be included before the current question; use them "
    "only to resolve references such as 'it' or 'that project', never as "
    "a substitute for the numbered context blocks. Do not reuse prior "
    "formatting, style instructions, output schemas, or constraints from "
    "earlier turns unless the current user message explicitly asks for "
    "them. Earlier turns are for resolving references only, such as "
    "pronouns or 'your previous answer,' never for carrying forward how a "
    "past answer was structured."
)


@dataclass
class RetrievedChunk:
    """One chunk returned by vector search, with provenance."""

    text: str
    source: str
    chunk_index: int
    score: float


def _source_key(source: str) -> str:
    """Collapse filename variants so duplicate exports do not crowd results."""
    return _SOURCE_KEY_RE.sub("", source.lower())


def _query_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for token in _QUERY_TERM_RE.findall(question.lower()):
        term = _source_key(token)
        if len(term) >= 4 and term not in _QUERY_STOPWORDS:
            terms.add(term)
    return terms


def _text_terms(text: str) -> set[str]:
    return {_source_key(token) for token in _QUERY_TERM_RE.findall(text.lower())}


def _lexical_boost(question_terms: set[str], chunk: RetrievedChunk) -> float:
    """Lift exact project/source matches above generic semantic neighbors."""
    if not question_terms:
        return 0.0

    source_key = _source_key(chunk.source)
    text_terms = _text_terms(chunk.text[:600])
    boost = 0.0
    for term in question_terms:
        if term in source_key:
            boost += 0.6
        elif term in text_terms:
            boost += 0.03
    return min(boost, 1.2)


async def retrieve(
    client: httpx.AsyncClient,
    settings: Settings,
    collection: Collection,
    question: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Embed the retrieval query and return the top_k most similar chunks.

    The collection uses cosine space, so Chroma's distance is 1 - cosine
    similarity; converting back gives a score where 1.0 is identical.
    Chroma's client is synchronous, so the query runs in a threadpool to
    keep the event loop free for concurrent requests. For conversational
    requests, callers may expand the query with assistant-only history so
    pronouns like "them" still retrieve the referenced source documents.
    """
    query_embedding = (await embed_texts(client, settings, [question]))[0]
    candidate_count = max(top_k, top_k * 8, 32)

    results = await run_in_threadpool(
        collection.query,
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    candidates: list[RetrievedChunk] = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        candidates.append(
            RetrievedChunk(
                text=text,
                source=str(meta.get("source", "unknown")),
                chunk_index=int(meta.get("chunk_index", -1)),
                score=round(1.0 - float(distance), 4),
            )
        )

    question_terms = _query_terms(question)
    candidates.extend(
        await _intro_chunks_for_matched_sources(collection, question_terms, candidates)
    )
    ranked_candidates = [
        chunk
        for _, chunk in sorted(
            enumerate(candidates),
            key=lambda item: (
                item[1].score + _lexical_boost(question_terms, item[1]),
                -item[0],
            ),
            reverse=True,
        )
    ]

    selected: list[RetrievedChunk] = []
    seen_sources: set[str] = set()
    for chunk in ranked_candidates:
        key = _source_key(chunk.source)
        if key in seen_sources:
            continue
        selected.append(chunk)
        seen_sources.add(key)
        if len(selected) == top_k:
            return selected

    for chunk in ranked_candidates:
        if chunk not in selected:
            selected.append(chunk)
        if len(selected) == top_k:
            break
    return selected


async def _intro_chunks_for_matched_sources(
    collection: Collection,
    question_terms: set[str],
    candidates: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    if not question_terms:
        return []

    sources: list[str] = []
    seen: set[str] = set()
    for chunk in candidates:
        source = chunk.source
        source_key = _source_key(source)
        if source in seen or not any(term in source_key for term in question_terms):
            continue
        sources.append(source)
        seen.add(source)

    intro_chunks: list[RetrievedChunk] = []
    for source in sources:
        try:
            result = await run_in_threadpool(
                collection.get,
                where={"source": source},
                include=["documents", "metadatas"],
            )
        except Exception:  # noqa: BLE001 - source intro boost is best-effort
            logger.debug("failed to load intro chunk for source=%s", source, exc_info=True)
            continue

        rows = sorted(
            zip(result.get("documents", []), result.get("metadatas", [])),
            key=lambda row: int(row[1].get("chunk_index", 0)),
        )
        for text, meta in rows[:1]:
            intro_chunks.append(
                RetrievedChunk(
                    text=text,
                    source=str(meta.get("source", source)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=1.0,
                )
            )

    return intro_chunks


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble numbered context blocks plus the question.

    Block numbers map one-to-one onto the sources array in the API
    response, so a citation like [2] in the answer is directly checkable
    by the caller.
    """
    blocks = [
        f"[{i + 1}] (source: {chunk.source}, chunk {chunk.chunk_index})\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    ]
    context = "\n\n---\n\n".join(blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


async def generate_answer(
    client: httpx.AsyncClient,
    settings: Settings,
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
) -> tuple[str, dict[str, int]]:
    """Send the grounded prompt to Ollama and return (answer, token meta).

    stream=False keeps the API contract simple: one JSON response with
    token counts. num_ctx is set explicitly because Ollama's default
    silently truncates context that does not fit, which in a RAG system
    means the model never saw the documents you retrieved. Callers pass
    already-filtered, reference-only assistant history when memory is
    enabled; this function always includes supplied history and leaves
    the system prompt to constrain how it is used.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": build_prompt(question, chunks)})

    response = await client.post(
        f"{settings.ollama_host}/api/chat",
        json={
            "model": settings.llm_model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": settings.temperature,
                "num_ctx": settings.num_ctx,
            },
        },
        timeout=settings.generate_timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()

    meta = {
        "prompt_tokens": int(data.get("prompt_eval_count", 0)),
        "completion_tokens": int(data.get("eval_count", 0)),
    }
    return data["message"]["content"], meta
