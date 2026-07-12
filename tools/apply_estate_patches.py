#!/usr/bin/env python3
"""Apply the corpus-retrieval rewire patches to ollama-rag-kit.

Why a patcher instead of whole-file replacements: app/retriever.py is
replaced wholesale because its entire job changed, but streaming.py and
main.py only change in a few verified regions, and overwriting files
that also carry untouched memory logic would be exactly the kind of
silent guessing this migration forbids. So each edit here is an exact
before/after pair: the old text must appear in the file exactly once or
the patch refuses, loudly, and tells you which region drifted. Same
principle as an exact-match str_replace: match or stop, never fuzz.

Usage, from anywhere:

    python3 tools/apply_estate_patches.py --repo /path/to/ollama-rag-kit
    python3 tools/apply_estate_patches.py --repo /path/to/ollama-rag-kit --apply

Default is a dry run that only reports. --apply writes changes after
saving a one-time <file>.orig-estate-rewire backup next to each touched
file. A file is patched atomically: every one of its patches must match
first, or none of them are applied. CRLF checkouts are handled by
matching in the file's own newline style.

Stdlib only. Python 3.10+.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKUP_SUFFIX = ".orig-estate-rewire"

# --------------------------------------------------------------------- #
# The patch set. Order within a file matters and is preserved.           #
# --------------------------------------------------------------------- #

PATCHES: list[dict] = [
    # ------------------------------------------------------------- #
    # app/streaming.py                                               #
    # ------------------------------------------------------------- #
    {
        "id": "S1-import-corpus-error",
        "file": "app/streaming.py",
        "why": "streaming needs the typed retrieval failure to answer it clearly",
        "old": "from app.retriever import retrieve\n",
        "new": "from app.retriever import CorpusUnreachable, retrieve\n",
    },
    {
        "id": "S2-drop-local-index-guard",
        "file": "app/streaming.py",
        "why": (
            "the empty-index guard and top_k clamp read the deprecated local "
            "collection; atlas-corpus owns corpus size now"
        ),
        "old": (
            "    indexed = collection.count()\n"
            "    if indexed == 0:\n"
            "        yield _sse({\"type\": \"error\", \"reason\": \"no_documents_indexed\"})\n"
            "        yield _sse({\"type\": \"done\"})\n"
            "        return\n"
            "\n"
            "    top_k = min(body.top_k or settings.top_k, indexed)\n"
        ),
        "new": (
            "    # Retrieval lives in atlas-corpus now; corpus size and emptiness\n"
            "    # are its concern. The collection argument stays in the signature\n"
            "    # so the route wiring is untouched, but nothing reads it here any\n"
            "    # more; conversation memory manages its own collection.\n"
            "    top_k = body.top_k or settings.top_k\n"
        ),
    },
    {
        "id": "S3-retrieve-via-corpus",
        "file": "app/streaming.py",
        "why": (
            "point retrieval at atlas-corpus and fail clearly, pointing at the "
            "estate status surfaces, when it is unreachable"
        ),
        "old": (
            "    try:\n"
            "        chunks = await retrieve(http, settings, collection, retrieval_question, top_k)\n"
            "    except httpx.HTTPError as exc:\n"
            "        logger.exception(\"retrieval failed\")\n"
            "        yield _sse({\"type\": \"error\", \"reason\": f\"retrieval_failed:{type(exc).__name__}\"})\n"
            "        yield _sse({\"type\": \"done\"})\n"
            "        return\n"
        ),
        "new": (
            "    try:\n"
            "        chunks = await retrieve(http, settings, retrieval_question, top_k)\n"
            "    except CorpusUnreachable as exc:\n"
            "        logger.error(\"atlas-corpus unreachable: %s\", exc)\n"
            "        yield _sse(\n"
            "            {\n"
            "                \"type\": \"error\",\n"
            "                \"reason\": \"retrieval_failed:CorpusUnreachable\",\n"
            "                \"detail\": (\n"
            "                    \"the corpus index on SPECULAR-CORE is not answering; \"\n"
            "                    \"the Lab status panels at atlas-systems.uk/lab show \"\n"
            "                    \"when it is back\"\n"
            "                ),\n"
            "            }\n"
            "        )\n"
            "        yield _sse({\"type\": \"done\"})\n"
            "        return\n"
            "    except httpx.HTTPError as exc:\n"
            "        logger.exception(\"retrieval failed\")\n"
            "        yield _sse({\"type\": \"error\", \"reason\": f\"retrieval_failed:{type(exc).__name__}\"})\n"
            "        yield _sse({\"type\": \"done\"})\n"
            "        return\n"
        ),
    },
    {
        "id": "S4-source-label-repo-fallback",
        "file": "app/streaming.py",
        "why": (
            "corpus sources are repo/path and most files are README.md; a "
            "generic stem should label as the repo so subject resolution keeps "
            "working in the new source universe"
        ),
        "old": (
            "def _source_label(source: str) -> str:\n"
            "    stem = PurePosixPath(source.replace(\"\\\\\", \"/\")).stem\n"
            "    stem = _SOURCE_PREFIX_RE.sub(\"\", stem).strip(\"-_ \")\n"
        ),
        "new": (
            "def _source_label(source: str) -> str:\n"
            "    normalized = source.replace(\"\\\\\", \"/\")\n"
            "    stem = PurePosixPath(normalized).stem\n"
            "    stem = _SOURCE_PREFIX_RE.sub(\"\", stem).strip(\"-_ \")\n"
            "    if stem.lower() in (\"readme\", \"index\") and \"/\" in normalized:\n"
            "        stem = normalized.split(\"/\", 1)[0]\n"
        ),
    },
    # ------------------------------------------------------------- #
    # app/main.py                                                    #
    # ------------------------------------------------------------- #
    {
        "id": "M1-imports",
        "file": "app/main.py",
        "why": "run_ingest and IngestStats leave with the boot ingest; the typed retrieval failure arrives",
        "old": (
            "from app.ingest import IngestStats, run_ingest\n"
            "from app.notify import send_alert\n"
            "from app.retriever import generate_answer, retrieve\n"
        ),
        "new": (
            "from app.notify import send_alert\n"
            "from app.retriever import CorpusUnreachable, generate_answer, retrieve\n"
        ),
    },
    {
        "id": "M2-drop-ingest-state",
        "file": "app/main.py",
        "why": "no boot ingest and no /ingest/refresh means no lock and no stats to hold",
        "old": (
            "    app.state.ingest_lock = asyncio.Lock()\n"
            "    app.state.last_ingest: IngestStats | None = None\n"
        ),
        "new": (
            "    # Local ingest state removed: atlas-corpus owns document ingest.\n"
        ),
    },
    {
        "id": "M3-drop-boot-ingest",
        "file": "app/main.py",
        "why": (
            "startup no longer ingests docs/; the Chroma connection stays "
            "because conversation memory still lives there"
        ),
        "old": (
            "        stats = await run_ingest(app.state.http, settings, app.state.collection)\n"
            "        app.state.last_ingest = stats\n"
            "\n"
            "        if stats.errors:\n"
            "            await send_alert(\n"
            "                settings,\n"
            "                title=\"RAG ingest completed with errors\",\n"
            "                message=\"; \".join(stats.errors[:5]),\n"
            "                level=\"warning\",\n"
            "            )\n"
            "        elif settings.notify_on_start:\n"
            "            await send_alert(\n"
            "                settings,\n"
            "                title=\"ollama-rag-kit online\",\n"
            "                message=(\n"
            "                    f\"{stats.files_indexed} files indexed, \"\n"
            "                    f\"{stats.files_skipped} unchanged, \"\n"
            "                    f\"{stats.chunks_added} chunks added\"\n"
            "                ),\n"
            "                level=\"success\",\n"
            "            )\n"
        ),
        "new": (
            "        if settings.notify_on_start:\n"
            "            await send_alert(\n"
            "                settings,\n"
            "                title=\"ollama-rag-kit online\",\n"
            "                message=\"retrieval delegated to atlas-corpus; memory and generation stay local\",\n"
            "                level=\"success\",\n"
            "            )\n"
        ),
    },
    {
        "id": "M4-ask-drop-local-guard",
        "file": "app/main.py",
        "why": "same reason as S2, for the non-streaming /ask path",
        "old": (
            "    settings: Settings = app.state.settings\n"
            "    collection = app.state.collection\n"
            "\n"
            "    indexed = collection.count()\n"
            "    if indexed == 0:\n"
            "        raise HTTPException(\n"
            "            status_code=503,\n"
            "            detail=\"No documents indexed. Add files to ./docs and POST /ingest/refresh.\",\n"
            "        )\n"
            "\n"
            "    # Asking for more results than exist makes Chroma raise; clamp instead.\n"
            "    top_k = min(body.top_k or settings.top_k, indexed)\n"
        ),
        "new": (
            "    settings: Settings = app.state.settings\n"
            "\n"
            "    # Retrieval is delegated to atlas-corpus; it owns corpus size and\n"
            "    # emptiness, so the local-collection guard left with the local path.\n"
            "    top_k = body.top_k or settings.top_k\n"
        ),
    },
    {
        "id": "M5-ask-retrieve-via-corpus",
        "file": "app/main.py",
        "why": (
            "retrieval failures become an honest 503 pointing at status; "
            "generation failures keep their 502 so the caller knows which "
            "dependency to look at"
        ),
        "old": (
            "    try:\n"
            "        chunks = await retrieve(app.state.http, settings, collection, body.question, top_k)\n"
            "        answer, token_meta = await generate_answer(app.state.http, settings, body.question, chunks)\n"
            "    except httpx.HTTPError as exc:\n"
            "        # Distinguish \"our dependency failed\" (502) from \"we failed\" (500)\n"
            "        # so the caller knows where to look.\n"
            "        raise HTTPException(status_code=502, detail=f\"Ollama request failed: {exc}\") from exc\n"
        ),
        "new": (
            "    try:\n"
            "        chunks = await retrieve(app.state.http, settings, body.question, top_k)\n"
            "    except CorpusUnreachable as exc:\n"
            "        raise HTTPException(\n"
            "            status_code=503,\n"
            "            detail=(\n"
            "                \"atlas-corpus is unreachable, so retrieval cannot run; \"\n"
            "                \"the Lab status panels at atlas-systems.uk/lab show when \"\n"
            "                \"it is back\"\n"
            "            ),\n"
            "        ) from exc\n"
            "    try:\n"
            "        answer, token_meta = await generate_answer(app.state.http, settings, body.question, chunks)\n"
            "    except httpx.HTTPError as exc:\n"
            "        # Distinguish \"our dependency failed\" (502) from \"we failed\" (500)\n"
            "        # so the caller knows where to look.\n"
            "        raise HTTPException(status_code=502, detail=f\"Ollama request failed: {exc}\") from exc\n"
        ),
    },
    {
        "id": "M6-remove-ingest-refresh",
        "file": "app/main.py",
        "why": (
            "atlas-corpus is the single ingest surface for document retrieval; "
            "docs/ and the atlas_docs collection are retained for development, "
            "see the README"
        ),
        "old": (
            "@app.post(\"/ingest/refresh\")\n"
            "async def ingest_refresh():\n"
            "    \"\"\"Re-run ingest over the docs directory without restarting.\n"
            "\n"
            "    The lock makes concurrent refreshes a 409 rather than a race: two\n"
            "    ingest runs interleaving deletes and adds on the same collection\n"
            "    could leave a file half-indexed.\n"
            "    \"\"\"\n"
            "    settings: Settings = app.state.settings\n"
            "\n"
            "    if app.state.ingest_lock.locked():\n"
            "        raise HTTPException(status_code=409, detail=\"An ingest run is already in progress.\")\n"
            "\n"
            "    async with app.state.ingest_lock:\n"
            "        stats = await run_ingest(app.state.http, settings, app.state.collection)\n"
            "        app.state.last_ingest = stats\n"
            "\n"
            "    return {\n"
            "        \"files_seen\": stats.files_seen,\n"
            "        \"files_indexed\": stats.files_indexed,\n"
            "        \"files_skipped\": stats.files_skipped,\n"
            "        \"chunks_added\": stats.chunks_added,\n"
            "        \"errors\": stats.errors,\n"
            "    }\n"
        ),
        "new": (
            "# /ingest/refresh removed: atlas-corpus is the single ingest surface\n"
            "# for document retrieval across the estate. Local docs/ and the\n"
            "# atlas_docs collection are retained for development only; see README.\n"
        ),
    },
    {
        "id": "M7-health-checks-corpus",
        "file": "app/main.py",
        "why": (
            "atlas-corpus is now a hard dependency of every answer, so /health "
            "must degrade when it is down; ingest stats have nothing to report"
        ),
        "old": (
            "    last: IngestStats | None = app.state.last_ingest\n"
            "    if last is not None:\n"
            "        status[\"last_ingest\"] = {\n"
            "            \"files_indexed\": last.files_indexed,\n"
            "            \"files_skipped\": last.files_skipped,\n"
            "            \"chunks_added\": last.chunks_added,\n"
            "            \"errors\": last.errors,\n"
            "        }\n"
        ),
        "new": (
            "    try:\n"
            "        corpus = await app.state.http.get(\n"
            "            f\"{settings.atlas_corpus_url.rstrip('/')}/health\",\n"
            "            timeout=settings.health_timeout_seconds,\n"
            "        )\n"
            "        status[\"atlas_corpus\"] = {\"reachable\": corpus.status_code == 200}\n"
            "        if corpus.status_code != 200:\n"
            "            degraded = True\n"
            "    except Exception as exc:  # noqa: BLE001\n"
            "        status[\"atlas_corpus\"] = {\"reachable\": False, \"error\": str(exc)}\n"
            "        degraded = True\n"
        ),
    },
    # ------------------------------------------------------------- #
    # app/config.py                                                  #
    # ------------------------------------------------------------- #
    {
        "id": "C1-corpus-settings",
        "file": "app/config.py",
        "why": "the corpus address and timeout budget become first class, env overridable settings",
        "old": (
            "    docs_dir: str = \"/srv/docs\"\n"
            "    chunk_size: int = 1200\n"
            "    chunk_overlap: int = 200\n"
            "    embed_batch_size: int = 32\n"
            "    top_k: int = 4\n"
        ),
        "new": (
            "    docs_dir: str = \"/srv/docs\"\n"
            "    chunk_size: int = 1200\n"
            "    chunk_overlap: int = 200\n"
            "    embed_batch_size: int = 32\n"
            "    top_k: int = 4\n"
            "\n"
            "    # Retrieval source: atlas-corpus over HTTP. Both stacks run on\n"
            "    # SPECULAR-CORE; the corpus publishes 8092 on the host, and this\n"
            "    # container reaches the host through host.docker.internal, the\n"
            "    # exact route OLLAMA_HOST already proves on every platform this\n"
            "    # repo supports. The connect timeout is what guarantees \"fail\n"
            "    # clearly, never hang\" when the corpus is stopped or asleep.\n"
            "    atlas_corpus_url: str = \"http://host.docker.internal:8092\"\n"
            "    atlas_corpus_timeout_seconds: float = 10.0\n"
            "    atlas_corpus_connect_timeout_seconds: float = 3.0\n"
        ),
    },
    # ------------------------------------------------------------- #
    # docker-compose.yml                                             #
    # ------------------------------------------------------------- #
    {
        "id": "D1-corpus-env",
        "file": "docker-compose.yml",
        "why": "surface the new settings in compose like every other knob; extra_hosts already maps the default hostname",
        "old": "      - TOP_K=${TOP_K:-4}\n",
        "new": (
            "      - TOP_K=${TOP_K:-4}\n"
            "      # Retrieval source: atlas-corpus on this host. Same host-gateway\n"
            "      # route as OLLAMA_HOST; override for a different topology.\n"
            "      - ATLAS_CORPUS_URL=${ATLAS_CORPUS_URL:-http://host.docker.internal:8092}\n"
            "      - ATLAS_CORPUS_TIMEOUT_SECONDS=${ATLAS_CORPUS_TIMEOUT_SECONDS:-10}\n"
            "      - ATLAS_CORPUS_CONNECT_TIMEOUT_SECONDS=${ATLAS_CORPUS_CONNECT_TIMEOUT_SECONDS:-3}\n"
        ),
    },
    # ------------------------------------------------------------- #
    # .env.example                                                   #
    # ------------------------------------------------------------- #
    {
        "id": "E1-corpus-env-example",
        "file": ".env.example",
        "why": "every knob and its default is listed here by repo convention",
        "old": (
            "# --- Chunking / retrieval ---------------------------------------------\n"
            "CHUNK_SIZE=1200\n"
            "CHUNK_OVERLAP=200\n"
            "TOP_K=4\n"
        ),
        "new": (
            "# --- Chunking / retrieval ---------------------------------------------\n"
            "CHUNK_SIZE=1200\n"
            "CHUNK_OVERLAP=200\n"
            "TOP_K=4\n"
            "\n"
            "# --- Retrieval source ----------------------------------------------\n"
            "# Document retrieval queries atlas-corpus over HTTP; this service no\n"
            "# longer searches its own local collection. Defaults reach the corpus\n"
            "# published on this host's port 8092 via the host gateway, the same\n"
            "# route OLLAMA_HOST uses. See README: \"Retrieval: atlas-corpus\".\n"
            "ATLAS_CORPUS_URL=http://host.docker.internal:8092\n"
            "ATLAS_CORPUS_TIMEOUT_SECONDS=10\n"
            "ATLAS_CORPUS_CONNECT_TIMEOUT_SECONDS=3\n"
        ),
    },
]


# --------------------------------------------------------------------- #
# Engine                                                                 #
# --------------------------------------------------------------------- #


def _read_raw(path: Path) -> str:
    """Read without newline translation so CRLF checkouts stay CRLF."""
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def _write_raw(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(content)


def _adapt_newlines(text: str, file_content: str) -> str:
    """Match the patch to the file's newline style so CRLF checkouts work."""
    if "\r\n" in file_content and "\r\n" not in text:
        return text.replace("\n", "\r\n")
    return text


def check_file(repo: Path, rel_path: str, patches: list[dict]) -> tuple[bool, list[str]]:
    """Verify every patch for one file. Returns (all_ok, report_lines).

    "Already applied" counts as ok: several patches are append-style
    (their new text contains their old text), so presence of the new
    text, not absence of the old, is the applied signal. That is what
    makes running this tool twice a no-op instead of a duplication.
    """
    lines: list[str] = []
    path = repo / rel_path
    if not path.is_file():
        lines.append(f"  MISSING FILE  {rel_path}")
        return False, lines

    content = _read_raw(path)
    all_ok = True
    for patch in patches:
        old = _adapt_newlines(patch["old"], content)
        new = _adapt_newlines(patch["new"], content)
        if new in content:
            lines.append(f"  already ok    {patch['id']}")
            continue
        count = content.count(old)
        if count == 1:
            lines.append(f"  ok            {patch['id']}")
        elif count == 0:
            all_ok = False
            lines.append(
                f"  NOT FOUND     {patch['id']}  (the file has drifted from the "
                "reviewed version; apply this region by hand using the patch "
                "source in tools/apply_estate_patches.py)"
            )
        else:
            all_ok = False
            lines.append(f"  AMBIGUOUS     {patch['id']}  (matched {count} times)")
    return all_ok, lines


def apply_file(repo: Path, rel_path: str, patches: list[dict]) -> bool:
    """Apply the pending patches to one already-verified file.

    Already-applied patches are skipped, so this is safe to run twice.
    Returns True if the file was modified. The backup is written once,
    before the first modification, and never overwritten after.
    """
    path = repo / rel_path
    content = _read_raw(path)
    original = content

    for patch in patches:
        old = _adapt_newlines(patch["old"], content)
        new = _adapt_newlines(patch["new"], content)
        if new in content:
            continue
        if content.count(old) != 1:
            raise RuntimeError(
                f"{patch['id']} no longer matches exactly once after earlier "
                "patches; file left untouched"
            )
        content = content.replace(old, new, 1)

    if content == original:
        return False

    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        _write_raw(backup, original)
    _write_raw(path, content)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="path to the ollama-rag-kit checkout")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "docker-compose.yml").is_file():
        print(f"refusing: {repo} does not look like ollama-rag-kit (no docker-compose.yml)")
        return 1

    by_file: dict[str, list[dict]] = {}
    for patch in PATCHES:
        by_file.setdefault(patch["file"], []).append(patch)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"estate rewire patches :: {mode} :: {repo}")
    print()

    failures = 0
    for rel_path, patches in by_file.items():
        ok, report = check_file(repo, rel_path, patches)
        print(rel_path)
        for line in report:
            print(line)
        if not ok:
            failures += 1
            print("  file skipped: fix the mismatches above, then re-run")
        elif args.apply:
            if apply_file(repo, rel_path, patches):
                print(f"  applied, backup at {rel_path}{BACKUP_SUFFIX}")
            else:
                print("  nothing to do (all patches already applied)")
        print()

    if failures:
        print(f"{failures} file(s) did not match cleanly; nothing risky was written.")
        return 1
    if not args.apply:
        print("all patches match exactly once; re-run with --apply to write them.")
    else:
        print("all patches applied. rebuild with: docker compose up -d --build rag-api")
    return 0


if __name__ == "__main__":
    sys.exit(main())
