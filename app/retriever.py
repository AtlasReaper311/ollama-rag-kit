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
from dataclasses import dataclass

import httpx
from chromadb.api.models.Collection import Collection
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.ingest import embed_texts

logger = logging.getLogger(__name__)

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
    "a substitute for the numbered context blocks."
)


@dataclass
class RetrievedChunk:
    """One chunk returned by vector search, with provenance."""

    text: str
    source: str
    chunk_index: int
    score: float


async def retrieve(
    client: httpx.AsyncClient,
    settings: Settings,
    collection: Collection,
    question: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Embed the question and return the top_k most similar chunks.

    The collection uses cosine space, so Chroma's distance is 1 - cosine
    similarity; converting back gives a score where 1.0 is identical.
    Chroma's client is synchronous, so the query runs in a threadpool to
    keep the event loop free for concurrent requests.
    """
    query_embedding = (await embed_texts(client, settings, [question]))[0]

    results = await run_in_threadpool(
        collection.query,
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append(
            RetrievedChunk(
                text=text,
                source=str(meta.get("source", "unknown")),
                chunk_index=int(meta.get("chunk_index", -1)),
                score=round(1.0 - float(distance), 4),
            )
        )
    return chunks


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
    means the model never saw the documents you retrieved.
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
