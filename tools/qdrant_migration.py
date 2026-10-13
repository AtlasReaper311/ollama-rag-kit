
"""Guarded Chroma export and disposable Qdrant migration utility.

The default workflow is read-only. Network writes and rollback require
QDRANT_MIGRATION_DISPOSABLE=1 plus --allow-disposable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
from qdrant_client import QdrantClient, models

_POINT_NAMESPACE = uuid.UUID("2f0d2c5e-6f72-4b1a-96d7-bc1c2f8f7f84")
_BATCH_SIZE = 64


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require_disposable(args: argparse.Namespace) -> None:
    if not args.allow_disposable or os.environ.get("QDRANT_MIGRATION_DISPOSABLE") != "1":
        raise RuntimeError(
            "network writes require --allow-disposable and QDRANT_MIGRATION_DISPOSABLE=1"
        )


def _qdrant_client(args: argparse.Namespace) -> QdrantClient:
    _require_disposable(args)
    if not args.qdrant_url.lower().startswith("https://"):
        raise RuntimeError("Qdrant migration requires an HTTPS URL")
    if not args.api_key:
        raise RuntimeError("QDRANT_API_KEY or --api-key is required")
    return QdrantClient(
        url=args.qdrant_url,
        api_key=args.api_key,
        verify=args.ca_cert or True,
        timeout=args.timeout,
    )


def _point_id(collection: str, original_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{collection}:{original_id}"))


def _load_records(
    input_dir: Path, collection_filter: str = ""
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with (input_dir / "records.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            record_collection = str(record.get("collection", ""))
            if collection_filter and record_collection != collection_filter:
                continue
            original_id = str(record["id"])
            if original_id in seen:
                raise ValueError(f"duplicate source id on line {line_number}: {original_id}")
            seen.add(original_id)
            embedding = record["embedding"]
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"invalid embedding on line {line_number}")
            metadata = record.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata must be an object on line {line_number}")
            records.append(
                {
                    "id": original_id,
                    "embedding": [float(value) for value in embedding],
                    "document": record.get("document"),
                    "metadata": metadata,
                }
            )
    collections = {str(record.get("collection", "")) for record in records if record.get("collection")}
    if not collection_filter and len(collections) > 1:
        raise ValueError("export contains multiple collections; pass --collection")
    if collection_filter:
        manifest = {**manifest, "collection": collection_filter}
    elif collections:
        manifest = {**manifest, "collection": next(iter(collections))}
    else:
        manifest_collections = [item.get("name") for item in manifest.get("collections", [])]
        if len(manifest_collections) == 1 and manifest_collections[0]:
            manifest = {**manifest, "collection": manifest_collections[0]}
        elif len(manifest_collections) > 1:
            raise ValueError("manifest contains multiple collections; pass --collection")
    expected = manifest.get("records")
    if expected is not None and int(expected) != len(records):
        raise ValueError(f"manifest records={expected}, export records={len(records)}")
    return manifest, records


def _schema(records: list[dict[str, Any]]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for record in records:
        for key, value in record["metadata"].items():
            current = type(value).__name__
            previous = fields.get(key)
            if previous and previous != current:
                raise ValueError(f"metadata field {key!r} changes type: {previous} vs {current}")
            fields[key] = current
    return dict(sorted(fields.items()))


def _migration_manifest(
    source_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    destination: str,
    validation: str,
) -> dict[str, Any]:
    collection = source_manifest.get("collection", destination)
    mappings = [
        {"original_id": record["id"], "point_id": _point_id(destination, record["id"])}
        for record in records
    ]
    return {
        "schema": "atlas-qdrant-disposable-migration/v2",
        "source_collection": collection,
        "destination_collection": destination,
        "record_count": len(records),
        "embedding_dimension": len(records[0]["embedding"]) if records else 0,
        "distance": "cosine",
        "id_mapping_hash": _sha256(mappings),
        "metadata_schema_hash": _sha256(_schema(records)),
        "validation": validation,
        "rollback_source": str(source_manifest.get("rollback_source", "records.jsonl")),
        "production_cutover": False,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _export(args: argparse.Namespace) -> None:
    _require_disposable(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
    collection = client.get_collection(args.collection)
    rows = collection.get(include=["documents", "metadatas", "embeddings"])
    embeddings = rows.get("embeddings") or []
    documents = rows.get("documents") or [None] * len(rows["ids"])
    metadatas = rows.get("metadatas") or [{}] * len(rows["ids"])
    records = [
        {
            "id": record_id,
            "collection": args.collection,
            "document": documents[index],
            "metadata": metadatas[index] or {},
            "embedding": list(embeddings[index]),
        }
        for index, record_id in enumerate(rows["ids"])
    ]
    manifest = {
        "schema": "atlas-chroma-export/v1",
        "collection": args.collection,
        "records": len(records),
        "embedding_dimension": len(records[0]["embedding"]) if records else 0,
        "distance": "cosine",
        "rollback_source": str(args.output_dir),
    }
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    _write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"action": "export", "collection": args.collection, "records": len(records)}))


def _points(records: list[dict[str, Any]], collection: str) -> list[models.PointStruct]:
    return [
        models.PointStruct(
            id=_point_id(collection, record["id"]),
            vector=record["embedding"],
            payload={
                "original_id": record["id"],
                "document": record["document"],
                "metadata": record["metadata"],
                "metadata_json": _canonical(record["metadata"]),
            },
        )
        for record in records
    ]


def _verify(client: QdrantClient, records: list[dict[str, Any]], destination: str) -> dict[str, Any]:
    if not client.collection_exists(destination):
        raise RuntimeError(f"destination collection does not exist: {destination}")
    info = client.get_collection(destination)
    size = info.config.params.vectors.size
    expected_size = len(records[0]["embedding"]) if records else size
    if size != expected_size:
        raise RuntimeError(f"dimension mismatch: destination={size}, source={expected_size}")
    count = int(client.count(collection_name=destination, exact=True).count)
    if count != len(records):
        raise RuntimeError(f"record count mismatch: destination={count}, source={len(records)}")
    expected = {record["id"]: record for record in records}
    actual: dict[str, Any] = {}
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=destination,
            limit=_BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in page:
            payload = point.payload or {}
            actual[str(payload.get("original_id", ""))] = point
        if offset is None:
            break
    if set(actual) != set(expected):
        raise RuntimeError("ID set mismatch")
    max_drift = 0.0
    for original_id, record in expected.items():
        point = actual[original_id]
        payload = point.payload or {}
        if payload.get("metadata") != record["metadata"]:
            raise RuntimeError(f"metadata mismatch for {original_id}")
        if payload.get("document") != record["document"]:
            raise RuntimeError(f"document mismatch for {original_id}")
        vector = list(point.vector or [])
        if len(vector) != len(record["embedding"]):
            raise RuntimeError(f"embedding dimension mismatch for {original_id}")
        max_drift = max(
            max_drift,
            max(abs(float(left) - float(right)) for left, right in zip(vector, record["embedding"])),
        )
    return {
        "records": len(actual),
        "ids_exact": True,
        "metadata_exact": True,
        "documents_exact": True,
        "embedding_dimension": expected_size,
        "max_embedding_abs_error": max_drift,
    }


def _apply(args: argparse.Namespace, rollback: bool = False) -> None:
    _require_disposable(args)
    source_manifest, records = _load_records(args.input_dir, args.collection)
    destination = args.destination or source_manifest.get("collection")
    if not destination:
        raise RuntimeError("--destination is required when the source has no collection")
    client = _qdrant_client(args)
    if rollback:
        if client.collection_exists(destination):
            client.delete_collection(destination)
    elif client.collection_exists(destination):
        raise RuntimeError(
            f"refusing to overwrite existing collection {destination}; "
            "use rollback only in a disposable environment"
        )
    dimension = len(records[0]["embedding"]) if records else args.embedding_dimension
    client.create_collection(
        collection_name=destination,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    points = _points(records, destination)
    for start in range(0, len(points), _BATCH_SIZE):
        client.upsert(collection_name=destination, points=points[start : start + _BATCH_SIZE], wait=True)
    validation = _verify(client, records, destination)
    output = _migration_manifest(source_manifest, records, destination, "passed")
    output["verification"] = validation
    output["rollback_action"] = "delete_destination_and_reindex_from_verified_export"
    _write_json(args.output_manifest, output)
    print(json.dumps({"action": "rollback" if rollback else "apply", **validation}))


def _plan(args: argparse.Namespace) -> None:
    source_manifest, records = _load_records(args.input_dir, args.collection)
    destination = args.destination or source_manifest.get("collection")
    plan = _migration_manifest(source_manifest, records, destination, "not-run-dry-run")
    plan["action"] = "plan"
    plan["writes"] = False
    print(json.dumps(plan, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "export", "apply", "verify", "rollback"))
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-manifest", type=Path, default=Path("qdrant-migration-manifest.json"))
    parser.add_argument("--collection", default="")
    parser.add_argument("--destination", default="")
    parser.add_argument("--chroma-host", default="chroma")
    parser.add_argument("--chroma-port", type=int, default=8000)
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY", ""))
    parser.add_argument("--ca-cert", default=os.environ.get("QDRANT_CA_CERT", ""))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--embedding-dimension", type=int, default=768)
    parser.add_argument("--allow-disposable", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "plan":
            _plan(args)
        elif args.action == "export":
            _export(args)
        elif args.action == "apply":
            _apply(args)
        elif args.action == "verify":
            source_manifest, records = _load_records(args.input_dir, args.collection)
            destination = args.destination or source_manifest.get("collection")
            print(json.dumps({"action": "verify", **_verify(_qdrant_client(args), records, destination)}))
        else:
            _apply(args, rollback=True)
    except (OSError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
