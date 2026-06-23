"""
Linear issue/ticket chunker.

Linear items are engineering, product, design, and business-ops tickets.
Structure: title, problem/goal description, acceptance criteria,
implementation plan, timeline comments with attributions.

35,308 files across 5 team folders (business-ops, design, engineering,
misc-chores, product-management). Median ~876 tokens, p99 ~1,663 tokens.
99.97% fit within the 8K context window, so default is whole-doc-as-one-chunk.

Metadata extracted:
  - ticket_id: e.g. "ENG-3927", "PM-31827"
  - team: folder name (engineering, product-management, etc.)
  - title: first line of the document
  - summary: problem/goal paragraph
  - contributors: people mentioned in timeline comments
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LinearMetadata:
    doc_id: str
    ticket_id: str  # e.g. "ENG-3927", "PM-31827"
    team: str  # folder: engineering, product-management, etc.
    title: str
    summary: str
    contributors: list[str]  # people mentioned in comments
    file_path: str


@dataclass
class LinearChunk:
    text: str
    metadata: LinearMetadata
    chunk_index: int
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

# Extract ticket ID from filename slug: ENG-3927, PM-31827, DESIGN-100, etc.
_RE_TICKET_ID = re.compile(r"((?:ENG|PM|DESIGN|OPS|MISC|BIZ|SUP)-\d+)", re.IGNORECASE)

# Timeline comment patterns:
# "2025-03-13 Maya Patel:" or "2025-03-13 - Maya Patel:"
_RE_TIMELINE_FULL_NAME = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s*[-–—]?\s*([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][\w'-]+)?)"
)

# "2025-03-16 - Jenna Morales (Security):" — name with role
_RE_NAME_WITH_ROLE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s*[-–—]?\s*"
    r"([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][\w'-]+)?)"
    r"\s*\(([^)]+)\)"
)

# "2026-02-24 - Aisha:" or "2026-03-03 - Design review (Sophie):"
_RE_TIMELINE_SINGLE_NAME = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s*[-–—]?\s*"
    r"(?:[A-Za-z ]+\s*\()?([A-Z][a-z]+)\)?:"
)


def _extract_ticket_id(filename_stem: str) -> str:
    """Extract the ticket ID (e.g. ENG-3927) from the filename."""
    # Filename format: dsid_xxx__ENG-3927-some-slug
    _, _, slug = filename_stem.partition("__")
    m = _RE_TICKET_ID.match(slug)
    return m.group(1) if m else ""


def _extract_contributors(lines: list[str]) -> list[str]:
    """Extract unique contributors from timeline comments."""
    contributors: list[str] = []
    seen: set[str] = set()

    for line in lines:
        stripped = line.strip()

        # Try name with role first: "2025-03-16 - Jenna Morales (Security):"
        m = _RE_NAME_WITH_ROLE.match(stripped)
        if m:
            name = m.group(1).strip()
            role = m.group(2).strip()
            key = name.lower()
            if key not in seen:
                seen.add(key)
                contributors.append(f"{name} ({role})")
            continue

        # Full name: "2025-03-13 Maya Patel:"
        m = _RE_TIMELINE_FULL_NAME.match(stripped)
        if m:
            name = m.group(1).strip()
            if len(name) > 3:
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    contributors.append(name)
            continue

        # Single name or role-prefixed: "2026-02-24 - Aisha:" or "Design review (Sophie):"
        m = _RE_TIMELINE_SINGLE_NAME.match(stripped)
        if m:
            name = m.group(1).strip()
            if len(name) > 2:
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    contributors.append(name)

    return contributors


def _extract_summary(lines: list[str]) -> str:
    """
    Extract summary from the ticket body.
    Looks for Problem/Goal/Overview sections, falls back to first substantial paragraph.
    """
    for i, line in enumerate(lines[:20]):
        low = line.strip().lower()
        if low.startswith(("problem:", "goal:", "overview:", "summary:", "context:", "motivation:")):
            content = line.split(":", 1)[1].strip()
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    break
                content += " " + next_line
            if len(content) > 20:
                return content[:500]

    # Fallback: first substantial line after title
    for line in lines[1:]:
        stripped = line.strip()
        if len(stripped) > 50:
            return stripped[:500]

    return ""


def _extract_metadata(file_path: Path, lines: list[str]) -> LinearMetadata:
    """Extract metadata from filename and content."""
    fname = file_path.stem
    doc_id = ""
    if fname.startswith("dsid_"):
        id_part, _, _ = fname.partition("__")
        doc_id = id_part

    ticket_id = _extract_ticket_id(fname)
    team = file_path.parent.name
    title = lines[0].strip() if lines else ""
    summary = _extract_summary(lines)
    contributors = _extract_contributors(lines)

    return LinearMetadata(
        doc_id=doc_id,
        ticket_id=ticket_id,
        team=team,
        title=title,
        summary=summary,
        contributors=contributors,
        file_path=str(file_path),
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_linear_document(
    file_path: Path,
    max_tokens: int = 8192,
    overlap_lines: int = 5,
    tokenizer=None,
) -> list[LinearChunk]:
    """
    Chunk a Linear ticket. Whole doc = one chunk in virtually all cases.
    """
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")

    # Normalize literal \n if needed
    literal_count = raw_text.count("\\n")
    real_count = raw_text.count("\n")
    if literal_count > real_count * 2 and literal_count > 10:
        raw_text = raw_text.replace("\\n", "\n")

    lines = raw_text.split("\n")
    metadata = _extract_metadata(file_path, lines)

    token_count = _count_tokens(raw_text, tokenizer)

    if token_count <= max_tokens:
        return [LinearChunk(
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


def _build_context_prefix(metadata: LinearMetadata) -> str:
    """Context prefix prepended to chunks after the first so each chunk
    retains awareness of the full ticket."""
    parts = [f"[{metadata.ticket_id}] {metadata.title}"]
    if metadata.summary:
        parts.append(f"Summary: {metadata.summary[:200]}")
    if metadata.team:
        parts.append(f"Team: {metadata.team}")
    return "\n".join(parts) + "\n---\n"


def _split_oversized(
    lines: list[str],
    full_text: str,
    metadata: LinearMetadata,
    max_tokens: int,
    overlap_lines: int,
    tokenizer,
) -> list[LinearChunk]:
    """Split at paragraph boundaries with overlap and context prefix."""
    context_prefix = _build_context_prefix(metadata)
    prefix_tokens = _count_tokens(context_prefix, tokenizer)

    para_breaks = [i for i, line in enumerate(lines) if not line.strip()]

    if not para_breaks:
        est_lines = max(10, int(max_tokens / 1.3 / 10))
        para_breaks = list(range(est_lines, len(lines), est_lines))

    chunks: list[LinearChunk] = []
    chunk_start = 0
    chunk_idx = 0

    for bp in para_breaks:
        segment_text = "\n".join(lines[chunk_start : bp + 1])
        budget = max_tokens if chunk_idx == 0 else max_tokens - prefix_tokens
        if _count_tokens(segment_text, tokenizer) >= budget and bp > chunk_start:
            chunk_text = "\n".join(lines[chunk_start:bp])
            if chunk_idx > 0:
                chunk_text = context_prefix + chunk_text
            chunks.append(LinearChunk(
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
        chunks.append(LinearChunk(
            text=chunk_text,
            metadata=metadata,
            chunk_index=chunk_idx,
            start_line=chunk_start,
            end_line=len(lines) - 1,
        ))

    return chunks if chunks else [LinearChunk(
        text=full_text,
        metadata=metadata,
        chunk_index=0,
        start_line=0,
        end_line=len(lines) - 1,
    )]


def chunk_all_linear(linear_dir: Path, **kwargs) -> list[LinearChunk]:
    """Walk the linear directory and chunk every .txt file."""
    all_chunks: list[LinearChunk] = []
    files = sorted(linear_dir.rglob("*.txt"))
    for fp in files:
        try:
            doc_chunks = chunk_linear_document(fp, **kwargs)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"WARN: failed to chunk {fp}: {e}")
    return all_chunks
