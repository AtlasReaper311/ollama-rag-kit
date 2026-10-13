"""Selectable Chroma/Qdrant collection adapter.

Chroma remains the default. Qdrant is opt-in and requires HTTPS plus an
API key. Original IDs remain in payload metadata while Qdrant receives
deterministic UUID point IDs.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

import chromadb
from qdrant_client import QdrantClient, models

from app.config import Settings

_POINT_NAMESPACE = uuid.UUID("2f0d2c5e-6f72-4b1a-96d7-bc1c2f8f7f84")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _point_id(collection: str, original_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{collection}:{original_id}"))


def _metadata_filter(where: dict[str, Any] | None) -> models.Filter | None:
    if not where:
        return None
    scalar_conditions: list[models.FieldCondition] = []
    logical_filters: list[models.Filter] = []
    for key, value in where.items():
        if key == "$and":
            logical_filters.extend(_metadata_filter(item) for item in value)
            continue
        if key == "$or":
            logical_filters.append(models.Filter(should=[_metadata_filter(item) for item in value]))
            continue
        if key.startswith("$"):
            raise ValueError(f"unsupported metadata operator: {key}")
        field = f"metadata.{key}"
        if isinstance(value, dict):
            ranges = {}
            for operator, bound in value.items():
                if operator == "$gte":
                    ranges["gte"] = bound
                elif operator == "$gt":
                    ranges["gt"] = bound
                elif operator == "$lte":
                    ranges["lte"] = bound
                elif operator == "$lt":
                    ranges["lt"] = bound
                else:
                    raise ValueError(f"unsupported metadata operator: {operator}")
            scalar_conditions.append(models.FieldCondition(key=field, range=models.Range(**ranges)))
        else:
            scalar_conditions.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    if scalar_conditions and logical_filters:
        return models.Filter(must=[*scalar_conditions, *logical_filters])
    if logical_filters:
        if len(logical_filters) == 1:
            return logical_filters[0]
        return models.Filter(must=logical_filters)
    return models.Filter(must=scalar_conditions)


class QdrantCollection:
    """A Chroma-shaped view over one private Qdrant collection."""

    backend = "qdrant"

    def __init__(self, client: QdrantClient, settings: Settings) -> None:
        self.client = client
        self.name = settings.collection_name
        self.vector_size = settings.qdrant_vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(collection_name=self.name):
            self.client.create_collection(
                collection_name=self.name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
            )
            return
        info = self.client.get_collection(collection_name=self.name)
        size = info.config.params.vectors.size
        if size != self.vector_size:
            raise RuntimeError(f"Qdrant collection {self.name} has dimension {size}; expected {self.vector_size}")

    def count(self) -> int:
        return int(self.client.count(collection_name=self.name, exact=True).count)

    def _payload(self, document: str | None, metadata: dict[str, Any] | None) -> dict[str, Any]:
        metadata = metadata or {}
        schema = {key: type(value).__name__ for key, value in sorted(metadata.items())}
        return {
            "document": document,
            "metadata": metadata,
            "metadata_json": _canonical(metadata),
            "metadata_schema_hash": hashlib.sha256(_canonical(schema).encode()).hexdigest(),
        }

    def _point(self, original_id: str, embedding: list[float], document: str | None, metadata: dict[str, Any] | None) -> models.PointStruct:
        return models.PointStruct(
            id=_point_id(self.name, original_id),
            vector=embedding,
            payload={"original_id": original_id, **self._payload(document, metadata)},
        )

    def _rows(self, points: list[Any], include_vectors: bool = False) -> dict[str, list[Any]]:
        rows: dict[str, list[Any]] = {"ids": [], "documents": [], "metadatas": []}
        if include_vectors:
            rows["embeddings"] = []
        for point in points:
            payload = point.payload or {}
            rows["ids"].append(str(payload.get("original_id", point.id)))
            rows["documents"].append(payload.get("document"))
            rows["metadatas"].append(
                json.loads(payload["metadata_json"]) if payload.get("metadata_json") else payload.get("metadata") or {}
            )
            if include_vectors:
                rows["embeddings"].append(list(point.vector or []))
        return rows

    def _all(self, where: dict[str, Any] | None = None, include_vectors: bool = False) -> list[Any]:
        points: list[Any] = []
        offset = None
        query_filter = _metadata_filter(where)
        while True:
            page, offset = self.client.scroll(
                collection_name=self.name,
                scroll_filter=query_filter,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=include_vectors,
            )
            points.extend(page)
            if offset is None:
                return points

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, list[Any]]:
        include_vectors = bool(include and "embeddings" in include)
        if ids is not None:
            points = self.client.retrieve(
                collection_name=self.name,
                ids=[_point_id(self.name, value) for value in ids],
                with_payload=True,
                with_vectors=include_vectors,
            )
        else:
            points = self._all(where, include_vectors)
        start = offset or 0
        return self._rows(points[start : start + limit if limit is not None else None], include_vectors)

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[list[Any]]]:
        if n_results < 1:
            raise ValueError("n_results must be positive")
        query_filter = _metadata_filter(where)
        result: dict[str, list[list[Any]]] = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        for vector in query_embeddings:
            points = self.client.query_points(
                collection_name=self.name,
                query=vector,
                query_filter=query_filter,
                limit=n_results,
                with_payload=True,
            ).points
            rows = self._rows(points)
            result["ids"].append(rows["ids"])
            result["documents"].append(rows["documents"])
            result["metadatas"].append(rows["metadatas"])
            result["distances"].append([1.0 - float(point.score) for point in points])
        return result

    def _write(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str] | None,
        metadatas: list[dict[str, Any]] | None,
        allow_existing: bool,
    ) -> None:
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings must have equal length")
        if documents is not None and len(documents) != len(ids):
            raise ValueError("documents and ids must have equal length")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas and ids must have equal length")
        if any(len(vector) != self.vector_size for vector in embeddings):
            raise ValueError(f"embeddings must have dimension {self.vector_size}")
        if not allow_existing:
            existing = self.get(ids=ids)["ids"]
            if existing:
                raise ValueError(f"duplicate IDs: {existing[:3]}")
        self.client.upsert(
            collection_name=self.name,
            wait=True,
            points=[
                self._point(
                    original_id=doc_id,
                    embedding=vector,
                    document=documents[index] if documents else None,
                    metadata=metadatas[index] if metadatas else None,
                )
                for index, (doc_id, vector) in enumerate(zip(ids, embeddings))
            ],
        )

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str] | None = None, metadatas: list[dict[str, Any]] | None = None) -> None:
        self._write(ids, embeddings, documents, metadatas, allow_existing=False)

    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str] | None = None, metadatas: list[dict[str, Any]] | None = None) -> None:
        self._write(ids, embeddings, documents, metadatas, allow_existing=True)

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        if ids is not None:
            self.client.delete(
                collection_name=self.name,
                wait=True,
                points_selector=models.PointIdsList(points=[_point_id(self.name, value) for value in ids]),
            )
        elif where:
            self.client.delete(
                collection_name=self.name,
                wait=True,
                points_selector=models.FilterSelector(filter=_metadata_filter(where)),
            )


def connect_collection(settings: Settings):
    """Connect to the configured backend; Chroma remains the safe default."""
    backend = settings.vector_backend.strip().lower()
    if backend == "chroma":
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
        return client.get_or_create_collection(
            name=settings.collection_name, metadata={"hnsw:space": "cosine"}
        )
    if backend != "qdrant":
        raise ValueError(f"unsupported VECTOR_BACKEND: {settings.vector_backend}")
    if not settings.qdrant_url.lower().startswith("https://"):
        raise ValueError("Qdrant backend requires an HTTPS QDRANT_URL")
    if not settings.qdrant_api_key:
        raise ValueError("Qdrant backend requires QDRANT_API_KEY")
    verify: bool | str = settings.qdrant_verify_tls
    if settings.qdrant_ca_cert:
        verify = settings.qdrant_ca_cert
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        verify=verify,
        timeout=settings.qdrant_timeout_seconds,
    )
    client.get_collections()
    return QdrantCollection(client, settings)
