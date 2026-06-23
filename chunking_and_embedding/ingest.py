"""
Unified ingestion pipeline: chunk all sources → embed → upsert into Qdrant.

Usage:
    # Full run (all sources):
    python -m chunking_and_embedding.ingest

    # Single source:
    python -m chunking_and_embedding.ingest --sources slack

    # Multiple sources:
    python -m chunking_and_embedding.ingest --sources confluence,fireflies,github

    # Dry run (chunk only, no embedding/upsert):
    python -m chunking_and_embedding.ingest --dry-run --sources gmail

    # Limit files per source (for testing):
    python -m chunking_and_embedding.ingest --limit 10 --sources hubspot

    # Skip Qdrant upsert (embed + save locally only):
    python -m chunking_and_embedding.ingest --no-upsert
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import (
    CONFLUENCE_DIR, FIREFLIES_DIR, GITHUB_DIR, GMAIL_DIR,
    GOOGLE_DRIVE_DIR, HUBSPOT_DIR, LINEAR_DIR, SLACK_DIR, JIRA_DIR,
    OUTPUT_DIR, LONG_DOC_THRESHOLD, BATCH_SIZE, QDRANT_COLLECTION,
)
from .late_chunking import LateChunkingEncoder, count_tokens
from .long_doc_handler import process_document
from .vector_store import get_client, ensure_collection, upsert_chunks

# Source-specific chunkers
from .confluence_chunker import chunk_confluence_document
from .fireflies_chunker import chunk_fireflies_document
from .github_chunker import chunk_github_document
from .gmail_chunker import chunk_gmail_document
from .google_drive_chunker import chunk_google_drive_document
from .hubspot_chunker import chunk_hubspot_document
from .linear_chunker import chunk_linear_document
from .slack_chunker import chunk_slack_document
from .jira_chunker import chunk_jira_document


# ---------------------------------------------------------------------------
# Metadata normalization — each chunker has its own dataclass; we flatten
# them into a uniform dict for Qdrant payloads
# ---------------------------------------------------------------------------

def _metadata_to_dict(meta, source: str) -> dict:
    """Convert any source-specific metadata dataclass to a flat dict."""
    if hasattr(meta, '__dataclass_fields__'):
        d = asdict(meta)
    else:
        d = dict(meta) if isinstance(meta, dict) else {"raw": str(meta)}
    d["source"] = source
    return d


# ---------------------------------------------------------------------------
# Per-source chunk+embed logic
# ---------------------------------------------------------------------------

@dataclass
class PreparedChunk:
    """Uniform representation across all sources before embedding."""
    text: str
    metadata: dict  # flat dict with "source" key
    chunk_index: int


def _prepare_confluence(files: list[Path], encoder: LateChunkingEncoder | None) -> list[PreparedChunk]:
    """Confluence uses late chunking (section-based + long doc handler)."""
    prepared = []
    for fp in tqdm(files, desc="Chunking confluence"):
        try:
            if encoder:
                chunk_embeddings = process_document(fp, encoder)
                for ce in chunk_embeddings:
                    c = ce.chunk
                    meta = _metadata_to_dict(c.metadata, "confluence")
                    meta["section_header"] = c.section_header
                    meta["chunk_index"] = c.chunk_index
                    prepared.append(PreparedChunk(
                        text=c.text,
                        metadata=meta,
                        chunk_index=c.chunk_index,
                    ))
                    # Store pre-computed embedding in metadata temporarily
                    meta["_embedding"] = ce.embedding
            else:
                chunks = chunk_confluence_document(fp)
                for c in chunks:
                    meta = _metadata_to_dict(c.metadata, "confluence")
                    meta["section_header"] = c.section_header
                    meta["chunk_index"] = c.chunk_index
                    prepared.append(PreparedChunk(
                        text=c.text, metadata=meta, chunk_index=c.chunk_index,
                    ))
        except Exception as e:
            print(f"  WARN: {fp.name}: {e}")
    return prepared


def _prepare_fireflies(files: list[Path], embed_fn) -> list[PreparedChunk]:
    """Fireflies uses semantic topic detection via embed_fn."""
    prepared = []
    for fp in tqdm(files, desc="Chunking fireflies"):
        try:
            chunks = chunk_fireflies_document(fp, embed_fn=embed_fn)
            for c in chunks:
                meta = _metadata_to_dict(c.metadata, "fireflies")
                meta["chunk_type"] = c.chunk_type
                meta["section_label"] = c.section_label
                meta["chunk_index"] = c.chunk_index
                prepared.append(PreparedChunk(
                    text=c.text, metadata=meta, chunk_index=c.chunk_index,
                ))
        except Exception as e:
            print(f"  WARN: {fp.name}: {e}")
    return prepared


def _prepare_simple_source(
    source_name: str,
    chunk_fn,
    files: list[Path],
    embed_fn=None,
) -> list[PreparedChunk]:
    """Generic handler for sources that use whole-doc-as-chunk (github, gmail, etc.)."""
    prepared = []
    for fp in tqdm(files, desc=f"Chunking {source_name}"):
        try:
            # Slack takes embed_fn
            if source_name == "slack" and embed_fn is not None:
                chunks = chunk_fn(fp, embed_fn=embed_fn)
            else:
                chunks = chunk_fn(fp)
            for c in chunks:
                meta = _metadata_to_dict(c.metadata, source_name)
                meta["chunk_index"] = c.chunk_index
                prepared.append(PreparedChunk(
                    text=c.text, metadata=meta, chunk_index=c.chunk_index,
                ))
        except Exception as e:
            print(f"  WARN: {fp.name}: {e}")
    return prepared


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_chunks(
    prepared: list[PreparedChunk],
    encoder: LateChunkingEncoder,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed all chunks. Skips any that already have pre-computed embeddings (confluence)."""
    embeddings = []
    to_embed_indices = []
    to_embed_texts = []

    for i, pc in enumerate(prepared):
        pre = pc.metadata.pop("_embedding", None)
        if pre is not None:
            embeddings.append((i, pre))
        else:
            to_embed_indices.append(i)
            to_embed_texts.append(pc.text)

    # Batch embed remaining
    if to_embed_texts:
        print(f"  Embedding {len(to_embed_texts)} chunks...")
        for start in tqdm(range(0, len(to_embed_texts), batch_size), desc="  Embedding"):
            batch = to_embed_texts[start : start + batch_size]
            batch_embs = encoder.embed_texts(batch)
            for j, emb in enumerate(batch_embs):
                embeddings.append((to_embed_indices[start + j], emb))

    # Sort by original index and stack
    embeddings.sort(key=lambda x: x[0])
    return np.stack([e for _, e in embeddings])


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SOURCES = {
    "confluence": {
        "dir": CONFLUENCE_DIR,
        "pattern": "**/*.txt",
    },
    "fireflies": {
        "dir": FIREFLIES_DIR,
        "pattern": "**/*.txt",
    },
    "github": {
        "dir": GITHUB_DIR,
        "chunk_fn": chunk_github_document,
        "pattern": "**/*.txt",
    },
    "gmail": {
        "dir": GMAIL_DIR,
        "chunk_fn": chunk_gmail_document,
        "pattern": "**/*.txt",
    },
    "google_drive": {
        "dir": GOOGLE_DRIVE_DIR,
        "chunk_fn": chunk_google_drive_document,
        "pattern": "**/*.txt",
    },
    "hubspot": {
        "dir": HUBSPOT_DIR,
        "chunk_fn": chunk_hubspot_document,
        "pattern": "**/*.txt",
    },
    "linear": {
        "dir": LINEAR_DIR,
        "chunk_fn": chunk_linear_document,
        "pattern": "**/*.txt",
    },
    "slack": {
        "dir": SLACK_DIR,
        "chunk_fn": chunk_slack_document,
        "pattern": "**/*.txt",
    },
    "jira": {
        "dir": JIRA_DIR,
        "chunk_fn": chunk_jira_document,
        "pattern": "**/*.txt",
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest all sources into Qdrant")
    parser.add_argument(
        "--sources", type=str, default=None,
        help="Comma-separated source names (default: all). E.g. --sources confluence,slack",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit files per source (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chunk only — no embedding or Qdrant upsert",
    )
    parser.add_argument(
        "--no-upsert", action="store_true",
        help="Embed but skip Qdrant upsert (save locally only)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--collection", type=str, default=QDRANT_COLLECTION,
        help="Qdrant collection name",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # .env is loaded automatically by config.py (imported above)

    # Determine which sources to process
    if args.sources:
        source_names = [s.strip() for s in args.sources.split(",")]
        invalid = [s for s in source_names if s not in SOURCES]
        if invalid:
            print(f"Unknown sources: {invalid}. Available: {list(SOURCES.keys())}")
            return
    else:
        source_names = list(SOURCES.keys())

    print(f"Sources to process: {source_names}")

    # Initialize encoder (skip for dry run)
    encoder = None
    embed_fn = None
    if not args.dry_run:
        encoder = LateChunkingEncoder()
        embed_fn = encoder.embed_texts

    # Connect to Qdrant (skip for dry run and no-upsert)
    qdrant_client = None
    if not args.dry_run and not args.no_upsert:
        qdrant_client = get_client()
        ensure_collection(qdrant_client, collection_name=args.collection)
        print(f"Connected to Qdrant (collection: {args.collection})")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    grand_total = 0

    for source_name in source_names:
        src = SOURCES[source_name]
        src_dir = src["dir"]

        if not src_dir.exists():
            print(f"\n[{source_name}] Directory not found: {src_dir}, skipping")
            continue

        files = sorted(src_dir.rglob(src["pattern"]))
        if args.limit:
            files = files[: args.limit]

        print(f"\n{'='*60}")
        print(f"[{source_name}] {len(files)} files from {src_dir}")
        print(f"{'='*60}")

        if not files:
            continue

        # --- Chunk ---
        if source_name == "confluence":
            prepared = _prepare_confluence(files, encoder)
        elif source_name == "fireflies":
            prepared = _prepare_fireflies(files, embed_fn)
        else:
            prepared = _prepare_simple_source(
                source_name, src["chunk_fn"], files, embed_fn=embed_fn,
            )

        print(f"  Chunked: {len(prepared)} chunks from {len(files)} files")

        if not prepared:
            continue

        # --- Dry run: save chunks and move on ---
        if args.dry_run:
            _save_dry_run(prepared, source_name, output_dir)
            grand_total += len(prepared)
            continue

        # --- Embed ---
        embeddings = _embed_chunks(prepared, encoder, batch_size=args.batch_size)
        print(f"  Embeddings: {embeddings.shape}")

        # --- Save locally ---
        _save_chunks_and_embeddings(prepared, embeddings, source_name, output_dir)

        # --- Upsert to Qdrant ---
        if qdrant_client is not None:
            texts = [pc.text for pc in prepared]
            metadatas = [pc.metadata for pc in prepared]
            n = upsert_chunks(
                qdrant_client, texts, embeddings, metadatas,
                collection_name=args.collection,
            )
            print(f"  Upserted {n} points to Qdrant")

        grand_total += len(prepared)

    print(f"\n{'='*60}")
    print(f"Done! Total chunks processed: {grand_total}")
    if qdrant_client:
        info = qdrant_client.get_collection(args.collection)
        print(f"Qdrant collection '{args.collection}': {info.points_count} points")


def _save_dry_run(prepared: list[PreparedChunk], source_name: str, output_dir: Path):
    """Save chunk texts and metadata for inspection (no embeddings)."""
    out_file = output_dir / f"{source_name}_chunks_dryrun.jsonl"
    with open(out_file, "w") as f:
        for pc in prepared:
            row = {**pc.metadata, "text": pc.text, "chunk_index": pc.chunk_index}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved dry-run chunks to {out_file}")


def _save_chunks_and_embeddings(
    prepared: list[PreparedChunk],
    embeddings: np.ndarray,
    source_name: str,
    output_dir: Path,
):
    """Save chunks as JSONL and embeddings as .npy (local backup)."""
    chunks_file = output_dir / f"{source_name}_chunks.jsonl"
    with open(chunks_file, "w") as f:
        for pc in prepared:
            row = {**pc.metadata, "text": pc.text}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    emb_file = output_dir / f"{source_name}_embeddings.npy"
    np.save(emb_file, embeddings)
    print(f"  Saved: {chunks_file} + {emb_file}")


if __name__ == "__main__":
    main()
