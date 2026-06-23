"""
Jira ticket chunker.

Jira files are support/ops tickets with structured sections:
  - Title (line 1)
  - Issue summary, impact, environment
  - Steps to reproduce, logs, analysis
  - Resolution/mitigation, action items
  - Chronological comments with attributions

6,120 files across 4 queues (customer-support, internal-support,
misc-requests, ops-requests). Median ~976 tokens, max ~1,904 tokens.
100% fit within the 8K context window → whole-doc-as-one-chunk.

Oversized fallback (for future data): splits at section/paragraph
boundaries and prepends a context summary (title + issue summary +
resolution) to each subsequent chunk so it retains awareness of the
full ticket narrative.

Metadata extracted:
  - ticket_id: e.g. "SUP-482158"
  - queue: folder name (customer-support, ops-requests, etc.)
  - title: first line
  - summary: issue summary section
  - resolution: mitigation/fix description
  - participants: people from timeline comments
  - customer: affected customer name if mentioned
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JiraMetadata:
    doc_id: str
    ticket_id: str  # e.g. "SUP-482158"
    queue: str  # folder: customer-support, ops-requests, etc.
    title: str
    summary: str
    resolution: str
    participants: list[str]
    customer: str  # affected customer if mentioned
    file_path: str


@dataclass
class JiraChunk:
    text: str
    metadata: JiraMetadata
    chunk_index: int
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Ticket ID from filename: SUP-482158, INT-842, OPS-1234, etc.
_RE_TICKET_ID = re.compile(r"((?:SUP|INT|OPS|MISC|REQ)-\d+)", re.IGNORECASE)

# Format A: "2026-03-10 09:33 — Name:" or "2026-03-10 Name:"
_RE_TIMELINE_A = re.compile(
    r"^\d{4}-\d{2}-\d{2}[\sT][\d:]*\s*[-–—]?\s*"
    r"([A-Z][a-z]+ [A-Z][\w'-]+(?:\s[A-Z][\w'-]+)?)"
)

# Format A with role: "2026-03-10 — Name (Role):"
_RE_TIMELINE_A_ROLE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[\sT][\d:]*\s*[-–—]?\s*"
    r"([A-Z][a-z]+ [A-Z][\w'-]+(?:\s[A-Z][\w'-]+)?)"
    r"\s*\(([^)]+)\)"
)

# Format B: "Name (Role) 2026-03-10:" — name before date
_RE_TIMELINE_B = re.compile(
    r"^([A-Z][a-z]+ [A-Z][\w'-]+(?:\s[A-Z][\w'-]+)?)"
    r"\s*\(([^)]+)\)\s*\d{4}-\d{2}-\d{2}"
)

# Customer name patterns: "Customer XYZ reported" or "tenant XYZ"
# Captures capitalized company names (1+ words) before action verbs
_RE_CUSTOMER_LINE = re.compile(
    r"(?:customer|tenant|client)\s+([A-Z][\w-]+(?:\s+[A-Z][\w-]+)*)(?:\s+(?:reported|observed|experienced|confirmed|asked|contacted|reports))",
)

# Section headers for splitting
_RE_SECTION = re.compile(
    r"^(Issue summary|Impact|Environment|Steps to reproduce|"
    r"Key logs|Observed telemetry|Notes|Mitigations? applied|"
    r"Fixes delivered|Follow-ups?|Action items|Resolution|Workaround|"
    r"Immediate customer communication|Root cause)[:\s]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_ticket_id(filename_stem: str) -> str:
    _, _, slug = filename_stem.partition("__")
    m = _RE_TICKET_ID.match(slug)
    return m.group(1) if m else ""


def _extract_participants(lines: list[str]) -> list[str]:
    participants: list[str] = []
    seen: set[str] = set()

    for line in lines:
        stripped = line.strip()

        # Format A with role: "2026-03-10 — Liam O'Connor (SRE):"
        m = _RE_TIMELINE_A_ROLE.match(stripped)
        if m:
            name = m.group(1).strip()
            role = m.group(2).strip()
            key = name.lower()
            if key not in seen:
                seen.add(key)
                participants.append(f"{name} ({role})")
            continue

        # Format A plain: "2026-03-10 Aisha Patel:"
        m = _RE_TIMELINE_A.match(stripped)
        if m:
            name = m.group(1).strip()
            key = name.lower()
            if key not in seen:
                seen.add(key)
                participants.append(name)
            continue

        # Format B: "Laura Chen (Support) 2026-03-10:"
        m = _RE_TIMELINE_B.match(stripped)
        if m:
            name = m.group(1).strip()
            role = m.group(2).strip()
            key = name.lower()
            if key not in seen:
                seen.add(key)
                participants.append(f"{name} ({role})")

    return participants


def _extract_customer(lines: list[str]) -> str:
    """Try to find the affected customer name."""
    for line in lines[:20]:
        m = _RE_CUSTOMER_LINE.search(line)
        if m:
            return m.group(1).strip()
    # Also check for "Customer: Name" in timeline comments
    for line in lines:
        stripped = line.strip()
        if "Customer" in stripped and "(" in stripped:
            # "Maple Health AI (Customer - tech@...)"
            idx = stripped.find("(Customer")
            if idx > 0:
                name = stripped[:idx].split("—")[-1].strip().split(":")[-1].strip()
                if len(name) > 2:
                    return name
    return ""


def _extract_section_text(lines: list[str], section_name: str) -> str:
    """Extract text from a named section."""
    collecting = False
    parts: list[str] = []

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()

        if low.startswith(section_name.lower()):
            # Start collecting after the header
            after = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if after:
                parts.append(after)
            collecting = True
            continue

        if collecting:
            # Stop at next section header or timeline entry
            if _RE_SECTION.match(stripped) or _RE_TIMELINE_A.match(stripped) or _RE_TIMELINE_B.match(stripped):
                break
            if stripped:
                parts.append(stripped)
            elif parts:
                break  # blank line after content

    return " ".join(parts)[:500]


def _extract_summary(lines: list[str]) -> str:
    """Extract the issue summary section."""
    # Try "Issue summary:" section first
    summary = _extract_section_text(lines, "issue summary")
    if summary:
        return summary

    # Fallback: first substantial line after title
    for line in lines[1:]:
        stripped = line.strip()
        if len(stripped) > 50 and not _RE_TIMELINE_A.match(stripped):
            return stripped[:500]
    return ""


def _extract_resolution(lines: list[str]) -> str:
    """Extract the resolution/mitigation/fix section."""
    for section in ("fixes delivered", "mitigations applied", "mitigation applied",
                    "resolution", "workaround"):
        text = _extract_section_text(lines, section)
        if text:
            return text
    return ""


def _extract_metadata(file_path: Path, lines: list[str]) -> JiraMetadata:
    fname = file_path.stem
    doc_id = ""
    if fname.startswith("dsid_"):
        id_part, _, _ = fname.partition("__")
        doc_id = id_part

    ticket_id = _extract_ticket_id(fname)
    queue = file_path.parent.name
    title = lines[0].strip() if lines else ""
    summary = _extract_summary(lines)
    resolution = _extract_resolution(lines)
    participants = _extract_participants(lines)
    customer = _extract_customer(lines)

    return JiraMetadata(
        doc_id=doc_id,
        ticket_id=ticket_id,
        queue=queue,
        title=title,
        summary=summary,
        resolution=resolution,
        participants=participants,
        customer=customer,
        file_path=str(file_path),
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_jira_document(
    file_path: Path,
    max_tokens: int = 8192,
    overlap_lines: int = 5,
    tokenizer=None,
) -> list[JiraChunk]:
    """
    Chunk a Jira ticket. Whole doc = one chunk in virtually all cases.

    Oversized fallback: split at section/paragraph boundaries, prepending
    a context prefix (title + summary + resolution) to each subsequent chunk
    so it retains awareness of the full ticket narrative.
    """
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")

    # Normalize literal \n
    literal_count = raw_text.count("\\n")
    real_count = raw_text.count("\n")
    if literal_count > real_count * 2 and literal_count > 10:
        raw_text = raw_text.replace("\\n", "\n")

    lines = raw_text.split("\n")
    metadata = _extract_metadata(file_path, lines)

    token_count = _count_tokens(raw_text, tokenizer)

    if token_count <= max_tokens:
        return [JiraChunk(
            text=raw_text,
            metadata=metadata,
            chunk_index=0,
            start_line=0,
            end_line=len(lines) - 1,
        )]

    # Oversized fallback with context prefix
    return _split_with_context(lines, raw_text, metadata, max_tokens, overlap_lines, tokenizer)


def _count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(text.split()) * 1.3)


def _build_context_prefix(metadata: JiraMetadata) -> str:
    """
    Build a short context prefix that summarizes what the ticket is about
    and how it was resolved. Prepended to every chunk after the first so
    each chunk carries awareness of the full ticket narrative.
    """
    parts = [f"[{metadata.ticket_id}] {metadata.title}"]
    if metadata.summary:
        parts.append(f"Summary: {metadata.summary[:200]}")
    if metadata.resolution:
        parts.append(f"Resolution: {metadata.resolution[:200]}")
    if metadata.customer:
        parts.append(f"Customer: {metadata.customer}")
    return "\n".join(parts) + "\n---\n"


def _split_with_context(
    lines: list[str],
    full_text: str,
    metadata: JiraMetadata,
    max_tokens: int,
    overlap_lines: int,
    tokenizer,
) -> list[JiraChunk]:
    """
    Split at section/paragraph boundaries, prepending context prefix
    to each chunk after the first.
    """
    context_prefix = _build_context_prefix(metadata)
    prefix_tokens = _count_tokens(context_prefix, tokenizer)
    # Budget for content per chunk (reserve space for prefix)
    content_budget = max_tokens - prefix_tokens

    # Find split points: section headers and paragraph breaks
    split_points = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _RE_SECTION.match(stripped):
            split_points.append(i)
        elif not stripped and i > 0:
            split_points.append(i)

    if not split_points:
        est_lines = max(10, int(content_budget / 1.3 / 10))
        split_points = list(range(est_lines, len(lines), est_lines))

    chunks: list[JiraChunk] = []
    chunk_start = 0
    chunk_idx = 0

    for sp in split_points:
        segment_text = "\n".join(lines[chunk_start : sp + 1])
        budget = max_tokens if chunk_idx == 0 else content_budget
        if _count_tokens(segment_text, tokenizer) >= budget and sp > chunk_start:
            chunk_text = "\n".join(lines[chunk_start:sp])
            if chunk_idx > 0:
                chunk_text = context_prefix + chunk_text
            chunks.append(JiraChunk(
                text=chunk_text,
                metadata=metadata,
                chunk_index=chunk_idx,
                start_line=chunk_start,
                end_line=sp - 1,
            ))
            chunk_idx += 1
            chunk_start = max(0, sp - overlap_lines)

    # Final segment
    if chunk_start < len(lines):
        chunk_text = "\n".join(lines[chunk_start:])
        if chunk_idx > 0:
            chunk_text = context_prefix + chunk_text
        chunks.append(JiraChunk(
            text=chunk_text,
            metadata=metadata,
            chunk_index=chunk_idx,
            start_line=chunk_start,
            end_line=len(lines) - 1,
        ))

    return chunks if chunks else [JiraChunk(
        text=full_text,
        metadata=metadata,
        chunk_index=0,
        start_line=0,
        end_line=len(lines) - 1,
    )]


def chunk_all_jira(jira_dir: Path, **kwargs) -> list[JiraChunk]:
    """Walk the jira directory and chunk every .txt file."""
    all_chunks: list[JiraChunk] = []
    files = sorted(jira_dir.rglob("*.txt"))
    for fp in files:
        try:
            doc_chunks = chunk_jira_document(fp, **kwargs)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"WARN: failed to chunk {fp}: {e}")
    return all_chunks
