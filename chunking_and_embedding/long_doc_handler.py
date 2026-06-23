"""
Handles documents that exceed the embedding model's context window.

Strategy: segment the document into overlapping windows that each fit within
the model context, then apply late chunking within each segment. The document
title + summary is prepended to every segment so cross-segment chunks still
carry document-level context.
"""

from __future__ import annotations

from pathlib import Path

from .confluence_chunker import Chunk, DocumentMetadata, chunk_confluence_document, extract_metadata
from .late_chunking import LateChunkingEncoder, ChunkEmbedding, count_tokens, _strip_overlap_prefix
from .config import LONG_DOC_THRESHOLD, MAX_CONTEXT_TOKENS


def _build_context_prefix(metadata: DocumentMetadata) -> str:
    """Build a short prefix with document title and summary for context."""
    parts = [f"Document: {metadata.title}"]
    if metadata.summary:
        parts.append(f"Summary: {metadata.summary}")
    parts.append(f"Source: {metadata.space}/{metadata.section}")
    return "\n".join(parts) + "\n\n---\n\n"


def process_long_document(
    file_path: Path,
    encoder: LateChunkingEncoder,
    **chunk_kwargs,
) -> list[ChunkEmbedding]:
    """
    Process a document that exceeds the model context window.

    1. Chunk the document normally (section-based).
    2. Group consecutive chunks into segments that fit within the context window,
       with a document-context prefix prepended to each segment.
    3. Apply late chunking within each segment.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    metadata = extract_metadata(file_path)
    chunks = chunk_confluence_document(file_path, **chunk_kwargs)

    total_tokens = count_tokens(text, encoder.tokenizer)

    if total_tokens <= LONG_DOC_THRESHOLD:
        # Fits in context — standard late chunking
        return encoder.encode_document_chunks(text, chunks)

    # Build context prefix
    prefix = _build_context_prefix(metadata)
    prefix_tokens = count_tokens(prefix, encoder.tokenizer)
    budget = MAX_CONTEXT_TOKENS - prefix_tokens - 50  # small margin

    # Group chunks into segments that fit within the token budget
    segments: list[list[Chunk]] = []
    current_segment: list[Chunk] = []
    current_tokens = 0

    for chunk in chunks:
        raw_text = _strip_overlap_prefix(chunk.text)
        chunk_tokens = count_tokens(raw_text, encoder.tokenizer)

        if current_tokens + chunk_tokens > budget and current_segment:
            segments.append(current_segment)
            # Start new segment with overlap: include last chunk from previous
            current_segment = [current_segment[-1]]
            current_tokens = count_tokens(
                _strip_overlap_prefix(current_segment[0].text), encoder.tokenizer
            )

        current_segment.append(chunk)
        current_tokens += chunk_tokens

    if current_segment:
        segments.append(current_segment)

    # Process each segment with late chunking
    all_embeddings: list[ChunkEmbedding] = []
    seen_chunk_indices: set[int] = set()

    for segment_chunks in segments:
        # Build the full text for this segment (prefix + chunk texts)
        segment_texts = []
        for c in segment_chunks:
            segment_texts.append(_strip_overlap_prefix(c.text))
        segment_body = "\n\n".join(segment_texts)
        segment_full = prefix + segment_body

        segment_embeddings = encoder.encode_document_chunks(segment_full, segment_chunks)

        # Deduplicate: if a chunk appeared in the overlap of a previous segment, keep
        # the first embedding (it had more preceding context)
        for emb in segment_embeddings:
            if emb.chunk.chunk_index not in seen_chunk_indices:
                seen_chunk_indices.add(emb.chunk.chunk_index)
                all_embeddings.append(emb)

    return all_embeddings


def process_document(
    file_path: Path,
    encoder: LateChunkingEncoder,
    **chunk_kwargs,
) -> list[ChunkEmbedding]:
    """
    Unified entry point: routes to standard late chunking or long-doc handler.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    total_tokens = count_tokens(text, encoder.tokenizer)

    if total_tokens <= LONG_DOC_THRESHOLD:
        chunks = chunk_confluence_document(file_path, **chunk_kwargs)
        return encoder.encode_document_chunks(text, chunks)
    else:
        return process_long_document(file_path, encoder, **chunk_kwargs)
