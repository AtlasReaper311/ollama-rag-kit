"""Document ingestion: load, chunk, embed, index.

The pipeline is idempotent by file content. Each file's SHA-256 hash
becomes part of its chunk IDs, so re-running ingest skips unchanged
files, re-indexes changed ones (deleting their stale chunks first), and
never duplicates. That makes "restart the container" a safe operation,
which is the property that matters most for a service that ingests on
startup.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from chromadb.api.models.Collection import Collection
from pypdf import PdfReader

from app.config import Settings

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class IngestStats:
    """Outcome of one ingest run, surfaced via /health and /ingest/refresh."""

    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_added: int = 0
    errors: list[str] = field(default_factory=list)


def load_text(path: Path) -> str:
    """Extract plain text from a supported document.

    PDF extraction quality varies with how the PDF was produced; pages
    that yield no text (scans without OCR) are simply skipped rather than
    failing the file.
    """
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p for p in pages if p.strip())
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks along paragraph boundaries.

    Paragraphs are the natural unit of meaning in prose and Markdown, so
    chunks are built by packing whole paragraphs up to chunk_size. A
    paragraph longer than chunk_size is hard-split. Each chunk carries the
    tail of its predecessor as overlap, so a sentence that straddles a
    boundary remains retrievable from at least one chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        # Oversized paragraph: flush what we have, then hard-split it.
        if len(para) > chunk_size:
            flush()
            start = 0
            while start < len(para):
                chunks.append(para[start : start + chunk_size].strip())
                start += chunk_size - overlap
            continue

        if len(current) + len(para) + 2 > chunk_size:
            flush()
            # Carry trailing context into the next chunk so boundary
            # sentences are never lost to the split.
            if chunks and overlap > 0:
                current = chunks[-1][-overlap:] + "\n\n"

        current += para + "\n\n"

    flush()
    return chunks


async def embed_texts(
    client: httpx.AsyncClient, settings: Settings, texts: list[str]
) -> list[list[float]]:
    """Embed a batch of texts via Ollama's /api/embed endpoint.

    /api/embed (Ollama >= 0.3) accepts a list and returns embeddings in
    order. The count check guards against a partial response being
    silently zipped against the wrong chunks, which would poison the
    index with mismatched vectors.
    """
    response = await client.post(
        f"{settings.ollama_host}/api/embed",
        json={"model": settings.embed_model, "input": texts},
        timeout=settings.embed_timeout_seconds,
    )
    response.raise_for_status()
    embeddings: list[list[float]] = response.json()["embeddings"]
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
        )
    return embeddings


def _file_hash(path: Path) -> str:
    """First 16 hex chars of the file's SHA-256: short, stable, content-keyed."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


async def run_ingest(
    client: httpx.AsyncClient, settings: Settings, collection: Collection
) -> IngestStats:
    """Index every supported document under settings.docs_dir.

    Per-file errors are recorded and skipped rather than aborting the run:
    one corrupt PDF should not stop the other ninety-nine documents from
    being searchable.
    """
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    stats = IngestStats()
    docs_dir = Path(settings.docs_dir)
    if not docs_dir.exists():
        logger.warning("Docs directory %s does not exist; nothing to ingest", docs_dir)
        return stats

    files = sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )

    for path in files:
        stats.files_seen += 1
        relpath = str(path.relative_to(docs_dir))
        try:
            content_hash = _file_hash(path)

            # Idempotence check: if any chunk with this content hash
            # exists, this exact file version is already indexed.
            existing = collection.get(where={"hash": content_hash}, limit=1)
            if existing["ids"]:
                stats.files_skipped += 1
                logger.debug("Skipping unchanged file %s", relpath)
                continue

            # The file is new or changed. Delete chunks from any previous
            # version (matched by source path) before adding, so edits
            # replace rather than accumulate.
            collection.delete(where={"source": relpath})

            text = load_text(path)
            chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                logger.warning("No extractable text in %s", relpath)
                continue

            for batch_start in range(0, len(chunks), settings.embed_batch_size):
                batch = chunks[batch_start : batch_start + settings.embed_batch_size]
                embeddings = await embed_texts(client, settings, batch)
                collection.add(
                    ids=[f"{content_hash}-{batch_start + i}" for i in range(len(batch))],
                    embeddings=embeddings,
                    documents=batch,
                    metadatas=[
                        {
                            "source": relpath,
                            "hash": content_hash,
                            "chunk_index": batch_start + i,
                        }
                        for i in range(len(batch))
                    ],
                )
                stats.chunks_added += len(batch)

            stats.files_indexed += 1
            logger.info("Indexed %s (%d chunks)", relpath, len(chunks))

        except Exception as exc:  # noqa: BLE001 - per-file isolation is the point
            message = f"{relpath}: {exc}"
            stats.errors.append(message)
            logger.error("Failed to ingest %s", message)

    logger.info(
        "Ingest complete: %d indexed, %d skipped, %d chunks added, %d errors",
        stats.files_indexed,
        stats.files_skipped,
        stats.chunks_added,
        len(stats.errors),
    )
    return stats
