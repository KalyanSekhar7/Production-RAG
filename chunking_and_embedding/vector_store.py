"""
Qdrant vector store layer.

Handles collection creation, upserting chunks with embeddings, and querying.
Credentials are loaded from environment variables or passed directly.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from .config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT, EMBEDDING_DIM


def get_client(
    host: str | None = None,
    port: int | None = None,
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    """
    Create a Qdrant client. Priority:
    1. Explicit url + api_key (for Qdrant Cloud)
    2. Environment variables QDRANT_URL / QDRANT_API_KEY
    3. Fallback to localhost
    """
    url = url or os.environ.get("QDRANT_URL")
    api_key = api_key or os.environ.get("QDRANT_API_KEY")

    if url:
        return QdrantClient(url=url, api_key=api_key)

    return QdrantClient(
        host=host or QDRANT_HOST,
        port=port or QDRANT_PORT,
    )


def ensure_collection(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION,
    embedding_dim: int = EMBEDDING_DIM,
):
    """Create the collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection '{collection_name}' (dim={embedding_dim})")
    else:
        print(f"Collection '{collection_name}' already exists")


def upsert_chunks(
    client: QdrantClient,
    texts: list[str],
    embeddings: np.ndarray,
    metadatas: list[dict],
    collection_name: str = QDRANT_COLLECTION,
    batch_size: int = 100,
) -> int:
    """
    Upsert chunks with embeddings and metadata into Qdrant.

    Args:
        texts: Chunk text content.
        embeddings: numpy array of shape (n, embedding_dim).
        metadatas: List of metadata dicts (one per chunk).
        collection_name: Target collection.
        batch_size: Points per upsert batch.

    Returns:
        Number of points upserted.
    """
    points = []
    for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadatas)):
        # Build payload: metadata + full text
        payload = {**meta, "text": text}
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meta.get('doc_id', '')}_{meta.get('chunk_index', i)}"))
        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload,
        ))

    # Batch upsert
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        total += len(batch)

    return total


def search(
    client: QdrantClient,
    query_embedding: np.ndarray,
    collection_name: str = QDRANT_COLLECTION,
    top_k: int = 10,
    source: str | None = None,
    space: str | None = None,
    meeting_type: str | None = None,
) -> list[dict]:
    """
    Search for similar chunks.

    Args:
        query_embedding: Query vector.
        top_k: Number of results.
        source: Filter by source type (e.g., "confluence", "fireflies").
        space: Filter by space (e.g., "eng-sre").
        meeting_type: Filter by meeting type (fireflies-specific).

    Returns:
        List of result dicts with score, text, and metadata.
    """
    filters = []
    if source:
        filters.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if space:
        filters.append(FieldCondition(key="space", match=MatchValue(value=space)))
    if meeting_type:
        filters.append(FieldCondition(key="meeting_type", match=MatchValue(value=meeting_type)))

    search_filter = Filter(must=filters) if filters else None

    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "score": r.score,
            "text": r.payload.get("text", ""),
            "metadata": {k: v for k, v in r.payload.items() if k != "text"},
        }
        for r in results.points
    ]
