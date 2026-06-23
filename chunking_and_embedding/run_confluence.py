"""
Main runner: processes all Confluence documents with late chunking and
saves chunk texts + embeddings to disk.

Usage:
    python -m chunking_and_embedding.run_confluence
    python -m chunking_and_embedding.run_confluence --limit 10  # process first 10 files
    python -m chunking_and_embedding.run_confluence --dry-run    # just chunk, no embeddings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import CONFLUENCE_DIR, OUTPUT_DIR, BATCH_SIZE
from .confluence_chunker import chunk_confluence_document
from .late_chunking import LateChunkingEncoder, count_tokens
from .long_doc_handler import process_document
from .config import LONG_DOC_THRESHOLD


def parse_args():
    parser = argparse.ArgumentParser(description="Late-chunk Confluence documents")
    parser.add_argument(
        "--confluence-dir", type=Path, default=CONFLUENCE_DIR,
        help="Path to confluence documents directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for chunks and embeddings",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N files (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only chunk documents, skip embedding generation",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Batch size for progress reporting",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    confluence_dir = args.confluence_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover files
    files = sorted(confluence_dir.rglob("*.txt"))
    if args.limit:
        files = files[: args.limit]

    print(f"Found {len(files)} Confluence documents in {confluence_dir}")

    # --- Dry run: just chunk and report stats ---
    if args.dry_run:
        _dry_run(files, output_dir)
        return

    # --- Full run: chunk + embed ---
    encoder = LateChunkingEncoder()

    all_results = []
    all_embeddings = []
    total_chunks = 0
    total_long_docs = 0
    errors = []

    for file_path in tqdm(files, desc="Processing documents"):
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            token_count = count_tokens(text, encoder.tokenizer)
            is_long = token_count > LONG_DOC_THRESHOLD

            if is_long:
                total_long_docs += 1

            chunk_embeddings = process_document(file_path, encoder)
            total_chunks += len(chunk_embeddings)

            for ce in chunk_embeddings:
                all_results.append({
                    "doc_id": ce.chunk.metadata.doc_id,
                    "title": ce.chunk.metadata.title,
                    "space": ce.chunk.metadata.space,
                    "section": ce.chunk.metadata.section,
                    "section_header": ce.chunk.section_header,
                    "chunk_index": ce.chunk.chunk_index,
                    "start_line": ce.chunk.start_line,
                    "end_line": ce.chunk.end_line,
                    "text": ce.chunk.text,
                    "file_path": ce.chunk.metadata.file_path,
                    "is_long_doc": is_long,
                })
                all_embeddings.append(ce.embedding)

        except Exception as e:
            errors.append({"file": str(file_path), "error": str(e)})
            print(f"\nERROR processing {file_path.name}: {e}")

    # --- Save outputs ---
    # 1. Chunk metadata + text as JSONL (line N corresponds to embedding row N)
    chunks_file = output_dir / "confluence_chunks.jsonl"
    with open(chunks_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_results)} chunks to {chunks_file}")

    # 2. Embeddings as numpy matrix (row index matches JSONL line index)
    if all_embeddings:
        embeddings_matrix = np.stack(all_embeddings)
        embeddings_file = output_dir / "confluence_embeddings.npy"
        np.save(embeddings_file, embeddings_matrix)
        print(f"Saved embeddings {embeddings_matrix.shape} to {embeddings_file}")

    print(f"\nStats:")
    print(f"  Documents processed: {len(files)}")
    print(f"  Long documents (segmented): {total_long_docs}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Embedding dim: {all_embeddings[0].shape[0] if all_embeddings else 'N/A'}")
    print(f"  Errors: {len(errors)}")

    if errors:
        errors_file = output_dir / "errors.json"
        with open(errors_file, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"  Error details saved to {errors_file}")


def _dry_run(files: list[Path], output_dir: Path):
    """Chunk documents without generating embeddings. Useful for inspecting chunks."""
    all_chunks = []
    stats = {"total_files": len(files), "total_chunks": 0, "chunk_sizes": []}

    for file_path in tqdm(files, desc="Chunking (dry run)"):
        try:
            chunks = chunk_confluence_document(file_path)
            stats["total_chunks"] += len(chunks)

            for c in chunks:
                line_count = c.end_line - c.start_line + 1
                stats["chunk_sizes"].append(line_count)
                all_chunks.append({
                    "doc_id": c.metadata.doc_id,
                    "title": c.metadata.title,
                    "space": c.metadata.space,
                    "section": c.metadata.section,
                    "section_header": c.section_header,
                    "chunk_index": c.chunk_index,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "num_lines": line_count,
                    "text": c.text,
                })
        except Exception as e:
            print(f"\nERROR chunking {file_path.name}: {e}")

    # Save dry-run chunks
    chunks_file = output_dir / "confluence_chunks_dryrun.jsonl"
    with open(chunks_file, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = stats["chunk_sizes"]
    if sizes:
        sizes_arr = np.array(sizes)
        print(f"\nDry run stats:")
        print(f"  Documents: {stats['total_files']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Avg chunks/doc: {stats['total_chunks'] / stats['total_files']:.1f}")
        print(f"  Chunk sizes (lines): min={sizes_arr.min()}, median={int(np.median(sizes_arr))}, max={sizes_arr.max()}, mean={sizes_arr.mean():.1f}")
        print(f"  Output: {chunks_file}")


if __name__ == "__main__":
    main()
