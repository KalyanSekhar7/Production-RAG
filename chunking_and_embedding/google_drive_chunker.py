"""
Google Drive document chunker.

Google Drive files live under two top-level paths:
  - shared_drives/<team>/<subfolder>/  (org-wide shared docs)
  - users/                              (personal docs/scratchpads)

100% of files fit within the 8K context window (max ~4300 tokens),
so the default is whole-document-as-one-chunk. Oversized fallback
splits at paragraph boundaries with overlap.

Metadata extracted:
  - drive_type: "shared" or "personal"
  - team: e.g. "engineering", "finance-and-legal", "product"
  - subfolder: e.g. "serving-runtime", "applied-ml"
  - title: first line of the document
  - summary: first substantive paragraph
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoogleDriveMetadata:
    doc_id: str
    title: str
    drive_type: str  # "shared" or "personal"
    team: str  # e.g. "engineering", "finance-and-legal"
    subfolder: str  # e.g. "serving-runtime", "applied-ml"
    summary: str
    file_path: str


@dataclass
class GoogleDriveChunk:
    text: str
    metadata: GoogleDriveMetadata
    chunk_index: int
    start_line: int
    end_line: int


def _extract_metadata(file_path: Path, lines: list[str]) -> GoogleDriveMetadata:
    """Extract metadata from path structure and content."""
    fname = file_path.stem
    doc_id = ""
    if fname.startswith("dsid_"):
        id_part, _, _ = fname.partition("__")
        doc_id = id_part

    title = lines[0].strip() if lines else ""

    # Parse path: .../google_drive/shared_drives/<team>/<subfolder>/file.txt
    #         or: .../google_drive/users/file.txt
    parts = file_path.parts
    try:
        gd_idx = parts.index("google_drive")
    except ValueError:
        gd_idx = -3

    drive_type = "unknown"
    team = ""
    subfolder = ""

    if gd_idx + 1 < len(parts):
        next_part = parts[gd_idx + 1]
        if next_part == "shared_drives":
            drive_type = "shared"
            if gd_idx + 2 < len(parts):
                team = parts[gd_idx + 2]
            if gd_idx + 3 < len(parts) and not parts[gd_idx + 3].endswith(".txt"):
                subfolder = parts[gd_idx + 3]
        elif next_part == "users":
            drive_type = "personal"

    summary = _extract_summary(lines)

    return GoogleDriveMetadata(
        doc_id=doc_id,
        title=title,
        drive_type=drive_type,
        team=team,
        subfolder=subfolder,
        summary=summary,
        file_path=str(file_path),
    )


_HEADER_LIKE = re.compile(
    r"^(Purpose|Summary|Background|Scope|Overview|Context|Goal|TL;DR|High.level)[:\s]",
    re.IGNORECASE,
)


def _extract_summary(lines: list[str]) -> str:
    """Extract a short summary from the document."""
    # Try labeled sections first
    for i, line in enumerate(lines[:20]):
        if _HEADER_LIKE.match(line.strip()):
            content = line.split(":", 1)[1].strip() if ":" in line else ""
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    break
                content += " " + next_line
            if len(content) > 20:
                return content[:400]

    # Fallback: first substantial line after title
    for line in lines[1:]:
        stripped = line.strip()
        if len(stripped) > 40:
            return stripped[:400]

    return ""


def chunk_google_drive_document(
    file_path: Path,
    max_tokens: int = 8192,
    overlap_lines: int = 5,
    tokenizer=None,
) -> list[GoogleDriveChunk]:
    """
    Chunk a Google Drive document. Whole doc = one chunk in virtually all cases.
    """
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    # Handle literal \n like in some Drive exports
    literal_count = raw_text.count("\\n")
    real_count = raw_text.count("\n")
    if literal_count > real_count * 2 and literal_count > 10:
        raw_text = raw_text.replace("\\n", "\n")

    lines = raw_text.split("\n")
    metadata = _extract_metadata(file_path, lines)

    token_count = _count_tokens(raw_text, tokenizer)

    if token_count <= max_tokens:
        return [GoogleDriveChunk(
            text=raw_text,
            metadata=metadata,
            chunk_index=0,
            start_line=0,
            end_line=len(lines) - 1,
        )]

    # Oversized fallback: split at paragraph boundaries
    return _split_oversized(lines, raw_text, metadata, max_tokens, overlap_lines, tokenizer)


def _count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(text.split()) * 1.3)


def _build_context_prefix(metadata: GoogleDriveMetadata) -> str:
    """Context prefix prepended to chunks after the first so each chunk
    retains awareness of the full document."""
    parts = [metadata.title]
    if metadata.drive_type == "shared" and metadata.team:
        parts.append(f"Team: {metadata.team}")
    if metadata.summary:
        parts.append(f"Summary: {metadata.summary[:200]}")
    return "\n".join(parts) + "\n---\n"


def _split_oversized(
    lines: list[str],
    full_text: str,
    metadata: GoogleDriveMetadata,
    max_tokens: int,
    overlap_lines: int,
    tokenizer,
) -> list[GoogleDriveChunk]:
    """Split at paragraph boundaries with overlap and context prefix."""
    context_prefix = _build_context_prefix(metadata)
    prefix_tokens = _count_tokens(context_prefix, tokenizer)

    para_breaks = [i for i, line in enumerate(lines) if not line.strip()]

    if not para_breaks:
        est_lines = max(10, int(max_tokens / 1.3 / 10))
        para_breaks = list(range(est_lines, len(lines), est_lines))

    chunks: list[GoogleDriveChunk] = []
    chunk_start = 0
    chunk_idx = 0

    for bp in para_breaks:
        segment_text = "\n".join(lines[chunk_start : bp + 1])
        budget = max_tokens if chunk_idx == 0 else max_tokens - prefix_tokens
        if _count_tokens(segment_text, tokenizer) >= budget and bp > chunk_start:
            chunk_text = "\n".join(lines[chunk_start:bp])
            if chunk_idx > 0:
                chunk_text = context_prefix + chunk_text
            chunks.append(GoogleDriveChunk(
                text=chunk_text,
                metadata=metadata,
                chunk_index=chunk_idx,
                start_line=chunk_start,
                end_line=bp - 1,
            ))
            chunk_idx += 1
            chunk_start = max(0, bp - overlap_lines)

    if chunk_start < len(lines):
        chunk_text = "\n".join(lines[chunk_start:])
        if chunk_idx > 0:
            chunk_text = context_prefix + chunk_text
        chunks.append(GoogleDriveChunk(
            text=chunk_text,
            metadata=metadata,
            chunk_index=chunk_idx,
            start_line=chunk_start,
            end_line=len(lines) - 1,
        ))

    return chunks if chunks else [GoogleDriveChunk(
        text=full_text,
        metadata=metadata,
        chunk_index=0,
        start_line=0,
        end_line=len(lines) - 1,
    )]


def chunk_all_google_drive(drive_dir: Path, **kwargs) -> list[GoogleDriveChunk]:
    """Walk the google_drive directory and chunk every .txt file."""
    all_chunks: list[GoogleDriveChunk] = []
    files = sorted(drive_dir.rglob("*.txt"))
    for fp in files:
        try:
            doc_chunks = chunk_google_drive_document(fp, **kwargs)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"WARN: failed to chunk {fp}: {e}")
    return all_chunks
