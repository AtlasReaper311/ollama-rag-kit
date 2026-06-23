"""FastAPI service wiring retrieval, generation, and ingest together.

Startup order matters: wait for Ollama, connect to Chroma, then ingest.
The dependency checks retry rather than relying on compose healthchecks,
because the application owning its own readiness works identically in
compose, plain docker run, and any future orchestrator.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import chromadb
import httpx
from chromadb.api.models.Collection import Collection
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.ingest import IngestStats, run_ingest
from app.notify import send_alert
from app.retriever import generate_answer, retrieve
from app.auth import AtlasSecretMiddleware, load_required_secret
from app.streaming import router as streaming_router

logger = logging.getLogger(__name__)

READINESS_ATTEMPTS = 30
READINESS_DELAY_SECONDS = 2.0


# --------------------------------------------------------------------- #
# Request / response models                                              #
# --------------------------------------------------------------------- #


class AskRequest(BaseModel):
    """A question against the indexed documents."""

    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    """Provenance for one context block used in the answer."""

    source: str
    chunk_index: int
    score: float
    preview: str


class AskResponse(BaseModel):
    """Grounded answer plus the evidence it was built from."""

    answer: str
    sources: list[SourceOut]
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


# --------------------------------------------------------------------- #
# Startup helpers                                                        #
# --------------------------------------------------------------------- #


async def _wait_for_ollama(client: httpx.AsyncClient, settings: Settings) -> None:
    """Block until Ollama answers /api/tags, or raise after the retry budget.

    Containers race their dependencies on boot; retrying here means
    `docker compose up` succeeds regardless of whether Ollama or this
    service wins the race.
    """
    for attempt in range(1, READINESS_ATTEMPTS + 1):
        try:
            response = await client.get(
                f"{settings.ollama_host}/api/tags", timeout=settings.health_timeout_seconds
            )
            response.raise_for_status()
            logger.info("Ollama reachable at %s", settings.ollama_host)
            return
        except Exception:  # noqa: BLE001 - any failure means "not ready yet"
            logger.info(
                "Waiting for Ollama at %s (attempt %d/%d)",
                settings.ollama_host,
                attempt,
                READINESS_ATTEMPTS,
            )
            await asyncio.sleep(READINESS_DELAY_SECONDS)
    raise RuntimeError(
        f"Ollama unreachable at {settings.ollama_host}. On Windows, check that "
        "OLLAMA_HOST=0.0.0.0 is set on the host (see README, Networking)."
    )


def _connect_chroma(settings: Settings) -> Collection:
    """Connect to the Chroma container and return the collection.

    Cosine space is set at creation time: with normalised embeddings it
    makes distance interpretable (score = 1 - distance), and it cannot be
    changed after the collection exists.
    """
    last_error: Exception | None = None
    for attempt in range(1, READINESS_ATTEMPTS + 1):
        try:
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            client.heartbeat()
            return client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.info(
                "Waiting for Chroma at %s:%d (attempt %d/%d)",
                settings.chroma_host,
                settings.chroma_port,
                attempt,
                READINESS_ATTEMPTS,
            )
            time.sleep(READINESS_DELAY_SECONDS)
    raise RuntimeError(f"Chroma unreachable: {last_error}")


async def _check_model_presence(
    client: httpx.AsyncClient, settings: Settings
) -> dict[str, bool]:
    """Report whether the configured models exist in Ollama.

    Missing models are warned about rather than auto-pulled: a multi-GB
    download inside container startup hides cost and fails opaquely.
    Pulling stays an explicit, visible host-side step.
    """
    response = await client.get(
        f"{settings.ollama_host}/api/tags", timeout=settings.health_timeout_seconds
    )
    response.raise_for_status()
    available = {model["name"] for model in response.json().get("models", [])}

    def has(name: str) -> bool:
        return name in available or f"{name}:latest" in available

    presence = {
        "embed_model": has(settings.embed_model),
        "llm_model": has(settings.llm_model),
    }
    for key, model in (("embed_model", settings.embed_model), ("llm_model", settings.llm_model)):
        if not presence[key]:
            logger.warning("Model %s not found in Ollama. Run: ollama pull %s", model, model)
    return presence


# --------------------------------------------------------------------- #
# Application                                                            #
# --------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app.state.settings = settings
    app.state.http = httpx.AsyncClient()
    app.state.ingest_lock = asyncio.Lock()
    app.state.last_ingest: IngestStats | None = None

    try:
        await _wait_for_ollama(app.state.http, settings)
        app.state.collection = _connect_chroma(settings)
        await _check_model_presence(app.state.http, settings)

        stats = await run_ingest(app.state.http, settings, app.state.collection)
        app.state.last_ingest = stats

        if stats.errors:
            await send_alert(
                settings,
                title="RAG ingest completed with errors",
                message="; ".join(stats.errors[:5]),
                level="warning",
            )
        elif settings.notify_on_start:
            await send_alert(
                settings,
                title="ollama-rag-kit online",
                message=(
                    f"{stats.files_indexed} files indexed, "
                    f"{stats.files_skipped} unchanged, "
                    f"{stats.chunks_added} chunks added"
                ),
                level="success",
            )
    except Exception as exc:
        await send_alert(
            settings,
            title="ollama-rag-kit failed to start",
            message=str(exc),
            level="failure",
        )
        raise

    yield

    await app.state.http.aclose()


app = FastAPI(
    title="ollama-rag-kit",
    description="Self-hosted RAG over local documents: Ollama + ChromaDB + FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(AtlasSecretMiddleware, secret=load_required_secret())
app.include_router(streaming_router)

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    # FastAPI ships interactive docs for free; make them discoverable.
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """Dependency-aware health: 200 when fully operational, 503 when degraded.

    Returning 503 on degradation lets compose healthchecks, Uptime Kuma,
    and load balancers all read the same signal without parsing the body.
    """
    settings: Settings = app.state.settings
    status: dict = {"service": "ollama-rag-kit", "status": "ok"}
    degraded = False

    try:
        models = await _check_model_presence(app.state.http, settings)
        status["ollama"] = {"reachable": True, **models}
        if not all(models.values()):
            degraded = True
    except Exception as exc:  # noqa: BLE001
        status["ollama"] = {"reachable": False, "error": str(exc)}
        degraded = True

    try:
        count = app.state.collection.count()
        status["chroma"] = {"reachable": True, "indexed_chunks": count}
    except Exception as exc:  # noqa: BLE001
        status["chroma"] = {"reachable": False, "error": str(exc)}
        degraded = True

    last: IngestStats | None = app.state.last_ingest
    if last is not None:
        status["last_ingest"] = {
            "files_indexed": last.files_indexed,
            "files_skipped": last.files_skipped,
            "chunks_added": last.chunks_added,
            "errors": last.errors,
        }

    if degraded:
        status["status"] = "degraded"
        raise HTTPException(status_code=503, detail=status)
    return status


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    """Answer a question from the indexed documents, with citations."""
    settings: Settings = app.state.settings
    collection = app.state.collection

    indexed = collection.count()
    if indexed == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed. Add files to ./docs and POST /ingest/refresh.",
        )

    # Asking for more results than exist makes Chroma raise; clamp instead.
    top_k = min(body.top_k or settings.top_k, indexed)
    started = time.perf_counter()

    try:
        chunks = await retrieve(app.state.http, settings, collection, body.question, top_k)
        answer, token_meta = await generate_answer(app.state.http, settings, body.question, chunks)
    except httpx.HTTPError as exc:
        # Distinguish "our dependency failed" (502) from "we failed" (500)
        # so the caller knows where to look.
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)

    return AskResponse(
        answer=answer,
        sources=[
            SourceOut(
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                preview=chunk.text[:200],
            )
            for chunk in chunks
        ],
        model=settings.llm_model,
        prompt_tokens=token_meta["prompt_tokens"],
        completion_tokens=token_meta["completion_tokens"],
        latency_ms=latency_ms,
    )


@app.post("/ingest/refresh")
async def ingest_refresh():
    """Re-run ingest over the docs directory without restarting.

    The lock makes concurrent refreshes a 409 rather than a race: two
    ingest runs interleaving deletes and adds on the same collection
    could leave a file half-indexed.
    """
    settings: Settings = app.state.settings

    if app.state.ingest_lock.locked():
        raise HTTPException(status_code=409, detail="An ingest run is already in progress.")

    async with app.state.ingest_lock:
        stats = await run_ingest(app.state.http, settings, app.state.collection)
        app.state.last_ingest = stats

    return {
        "files_seen": stats.files_seen,
        "files_indexed": stats.files_indexed,
        "files_skipped": stats.files_skipped,
        "chunks_added": stats.chunks_added,
        "errors": stats.errors,
    }
