"""Retrieval and answer generation.

Retrieval is delegated to atlas-corpus over HTTP: the estate keeps one
retrieval brain instead of two. This service used to embed queries and
search its own small ChromaDB collection built from the local docs/
folder (nine files); atlas-corpus indexes every README, pinned doc, and
published case study across the estate, refreshed on push. Ramone
answering from a private nine-document subset while the real index sat
one port away was the drift this rewire removes.

The seams are unchanged on purpose: retrieve() still returns
RetrievedChunk objects, build_prompt() still assembles them into
numbered context blocks, and generate_answer() still asks the LLM to
answer strictly from that context. Citation numbering, the sources
array shape, and the streaming contract ramone-edge depends on are
therefore untouched; only where the chunks come from has changed.

Two contract facts about atlas-corpus are enforced here rather than
discovered as 422s in production: SearchRequest caps the query at 500
characters and top_k at 10 (app/models.py in atlas-corpus). Memory
expanded retrieval questions can exceed the query cap, so _corpus_query
falls back to the primary question plus as much reference context as
fits. The X-Atlas-Internal header marks these calls as machine traffic
from a private address, the same convention specular-sentinel uses, so
Ramone's retrievals never spend the corpus's public rate budget or
pollute its visitor query stats.

Failure mode is explicit: CorpusUnreachable, raised on any transport
error, non-200, or malformed body, with tight timeouts so a sleeping or
stopped corpus fails in seconds instead of hanging a stream. Callers
turn it into a clear message pointing at the estate's status surfaces.
"""

import logging
import re
from dataclasses import dataclass

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

# Mirrors of the atlas-corpus request contract (atlas-corpus
# app/models.py: SearchRequest). Enforced client side so an oversized
# request never leaves this service.
CORPUS_QUERY_MAX_CHARS = 500
CORPUS_TOP_K_MAX = 10

# Presence of this header from a loopback or private source address is
# what atlas-corpus's _is_internal() checks; the value only identifies
# the caller in its logs.
INTERNAL_HEADER = "X-Atlas-Internal"
INTERNAL_CALLER = "ollama-rag-kit"

_RETRIEVAL_MARKER = "Current question for retrieval:"
_PUBLIC_BOUNDARY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(cv|curriculum vitae|resume|cover letter|salary|interview|job application|application material)\b",
            re.IGNORECASE,
        ),
        "That is private application material. I can answer from the public Atlas Systems estate instead.",
    ),
    (
        re.compile(
            r"\b(university notes?|lecture notes?|study notes?|coursework|grades?|marks?|academic drafts?|honours drafts?|abertay)\b",
            re.IGNORECASE,
        ),
        "That is private academic material. I can answer from the public Atlas Systems estate instead.",
    ),
    (
        re.compile(
            r"\b(books?|reading notes?|reference library|private library|licensed third-party text)\b",
            re.IGNORECASE,
        ),
        "That is private reference material. I can answer from the public Atlas Systems estate instead.",
    ),
    (
        re.compile(
            r"\b(soh|employer material|employer code|employer meetings?|employer architecture|work macbook|colleagues?|slack|tickets?)\b",
            re.IGNORECASE,
        ),
        "That is employer material. I can answer from the public Atlas Systems estate instead.",
    ),
    (
        re.compile(
            r"\b(secrets?|tokens?|api keys?|passwords?|credentials?|\.env|webhook values?|trigger_secret|corpus_secret|github_token)\b",
            re.IGNORECASE,
        ),
        "That is secret or credential material. I can answer from the public Atlas Systems estate instead.",
    ),
    (
        re.compile(
            r"\b(private open webui collections?|private openwebui collections?|private collections?)\b",
            re.IGNORECASE,
        ),
        "That is private collection material. I can answer from the public Atlas Systems estate instead.",
    ),
    (
        re.compile(
            r"\b(what did atlas ask you to remember|what did atlas tell you to remember|remember to help|remember this today|private memory|ramone_memory)\b",
            re.IGNORECASE,
        ),
        "That is private memory. I can answer from the public Atlas Systems estate instead.",
    ),
)

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


class CorpusUnreachable(RuntimeError):
    """atlas-corpus did not return a usable search response.

    Raised for transport failures, non-200 statuses, and malformed
    bodies alike, because every one of them means the same thing to a
    caller: retrieval cannot run right now, say so clearly and point at
    the status surfaces instead of hanging or answering from nothing.
    """


def public_boundary_refusal(question: str) -> str | None:
    """Refuse public Ramone requests for obvious private material.

    The corpus API has the same guard for direct public callers, but
    ramone-edge reaches this service first and this service queries the
    corpus as internal machine traffic. Keep the public boundary here
    too so the streaming path never depends on retrieval ranking.
    """
    for pattern, answer in _PUBLIC_BOUNDARY_RULES:
        if pattern.search(question):
            return answer
    return None


@dataclass
class RetrievedChunk:
    """One retrieved context block with provenance.

    source is "repo/path" so the sources ids ramone-edge already emits
    ("{source}#{chunk_index}") stay well formed, and streaming.py's
    subject labelling keeps working on the path stem.
    """

    text: str
    source: str
    chunk_index: int
    score: float


def _primary_query(expanded: str) -> str:
    """Extract the current question from a memory expanded retrieval query.

    app/memory.py's turns_to_retrieval_query() prefixes the live
    question with a marker line and appends recent assistant answers
    for reference resolution. The first non-empty line after the marker
    is the question itself. Text without the marker passes through.
    """
    text = expanded.strip()
    if _RETRIEVAL_MARKER not in text:
        return text
    tail = text.split(_RETRIEVAL_MARKER, 1)[1]
    for line in tail.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def _corpus_query(retrieval_question: str) -> str:
    """Fit a retrieval question inside the corpus's 500 character cap.

    Short questions pass through untouched so behaviour is identical to
    a user typing at the corpus directly. Over the cap, the primary
    question is kept whole (it carries the intent) and the reference
    context fills whatever budget remains, because losing the tail of a
    prior answer is cheaper than losing the question.
    """
    text = retrieval_question.strip()
    if len(text) <= CORPUS_QUERY_MAX_CHARS:
        return text
    primary = _primary_query(text)
    if len(primary) >= CORPUS_QUERY_MAX_CHARS:
        return primary[:CORPUS_QUERY_MAX_CHARS].strip()
    remainder = " ".join(text.replace(primary, " ").split())
    budget = CORPUS_QUERY_MAX_CHARS - len(primary) - 1
    if budget <= 0 or not remainder:
        return primary
    return f"{primary} {remainder[:budget]}".strip()


def _chunk_from_hit(hit: dict, rank: int) -> RetrievedChunk:
    """Map one atlas-corpus hit into the chunk shape this service uses.

    The deployed SearchHit model carries source_repo, file_path, text,
    doc_type, last_updated, and chunk_index. Some estate docs describe
    the same hit as {repo, path, excerpt}; both spellings are accepted
    so a serialization shift in the producer cannot silently break this
    consumer (the API registry panel already taught that lesson).
    """
    repo = str(hit.get("source_repo") or hit.get("repo") or "corpus")
    path = str(hit.get("file_path") or hit.get("path") or "unknown")
    text = str(hit.get("text") or hit.get("excerpt") or "")
    try:
        chunk_index = int(hit.get("chunk_index", rank))
    except (TypeError, ValueError):
        chunk_index = rank
    try:
        score = float(hit.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return RetrievedChunk(
        text=text,
        source=f"{repo}/{path}",
        chunk_index=chunk_index,
        score=score,
    )


async def retrieve(
    client: httpx.AsyncClient,
    settings: Settings,
    question: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Top-k context blocks for a question, from atlas-corpus over HTTP.

    Both stacks run on SPECULAR-CORE; the address in
    settings.atlas_corpus_url stays on the machine (host gateway by
    default) rather than routing back out through the public tunnel.
    The connect timeout is what guarantees "fail clearly, never hang"
    when the corpus container is stopped or the machine is waking up.

    Raises CorpusUnreachable for any outcome that is not a well formed
    200; zero hits is a legitimate empty result, not an error.
    """
    query = _corpus_query(question)
    k = max(1, min(int(top_k), CORPUS_TOP_K_MAX))
    url = settings.atlas_corpus_url.rstrip("/") + "/search"
    timeout = httpx.Timeout(
        settings.atlas_corpus_timeout_seconds,
        connect=settings.atlas_corpus_connect_timeout_seconds,
    )

    try:
        response = await client.post(
            url,
            json={"query": query, "top_k": k},
            headers={INTERNAL_HEADER: INTERNAL_CALLER},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise CorpusUnreachable(
            f"atlas-corpus request failed ({type(exc).__name__}): {exc}"
        ) from exc

    if response.status_code != 200:
        raise CorpusUnreachable(
            f"atlas-corpus answered {response.status_code} at {url}"
        )

    try:
        data = response.json()
        hits = data["hits"]
    except (ValueError, KeyError) as exc:
        raise CorpusUnreachable(
            "atlas-corpus returned a body without a hits field"
        ) from exc
    if not isinstance(hits, list):
        raise CorpusUnreachable("atlas-corpus hits field is not a list")

    chunks = [_chunk_from_hit(hit, rank) for rank, hit in enumerate(hits)]
    logger.info(
        "corpus retrieval: %d hits (asked %d) for %d-char query",
        len(chunks),
        k,
        len(query),
    )
    return chunks


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble numbered context blocks and the question into one prompt.

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
