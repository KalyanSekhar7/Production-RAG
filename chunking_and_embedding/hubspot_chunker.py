"""
HubSpot CRM record chunker.

HubSpot files are company/deal records with semi-structured data:
  - Narrative description paragraph (company overview, product, pain points)
  - Chronological timeline entries (dated CRM activity)
  - Key-value metrics (latency targets, cost, SLA requirements)
  - Action items, blockers, next steps
  - Direct quotes from stakeholders

14,567 files (+ 450 exports), median ~555 tokens, p99 ~1,039 tokens.
100% fit within the 8K context window, so default is whole-doc-as-one-chunk.
Oversized fallback splits at paragraph boundaries with overlap.

Metadata extracted:
  - company_name: first line of the document
  - doc_id: from filename (dsid_...)
  - contacts: key people mentioned (names with roles/titles)
  - timeline_count: number of dated timeline entries
  - summary: first substantive paragraph describing the company/deal
  - blockers: extracted blocker/risk items
  - is_export: whether the file is from the exports subfolder
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HubSpotMetadata:
    doc_id: str
    company_name: str
    contacts: list[str]  # key people mentioned with context
    timeline_count: int  # number of dated activity entries
    summary: str  # first substantive paragraph
    blockers: list[str]  # risk/blocker items
    is_export: bool  # from exports/ subfolder
    file_path: str


@dataclass
class HubSpotChunk:
    text: str
    metadata: HubSpotMetadata
    chunk_index: int
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

# Matches dated timeline entries like "2026-02-10: ..." or "2024-11-12 - ..."
_RE_TIMELINE = re.compile(r"^\d{4}-\d{2}-\d{2}\s*[:\-–—]")

# Pattern 1: "Role: Name (email/context)" — e.g. "CTO: Lena Ortiz (lena@...)"
_RE_ROLE_NAME = re.compile(
    r"(?:^[-\s]*)"
    r"((?:Founder|Co-[Ff]ounder|CTO|CEO|CFO|COO|VP|Head of \w+|Director|Lead|"
    r"Sr\.?\s+\w+\s+Engineer|Senior\s+\w+\s+Engineer|"
    r"AE|SE|CSM|Champion|Sponsor|Product|Infra|Security|Legal|Procurement)"
    r"(?:\s+\w+)*)"
    r"[:\s]+([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
)

# Pattern 2: "Name (role)" — e.g. "Jonah Park (solo dev, ex-SaaS engineer)"
_RE_NAME_ROLE = re.compile(
    r"(?:^|\b)([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"
    r"\s*\(([^)]+)\)",
)

# Pattern 3: "Role (Name)" — e.g. "Head of ML ops (Laura Hendricks)"
_RE_ROLE_PARENS_NAME = re.compile(
    r"((?:Head of \w+(?:\s\w+)?|Senior\s+\w+\s+Engineer|Sr\.?\s+\w+\s+Engineer|"
    r"CTO|CEO|CFO|COO|VP\s+\w+|Director\s+\w+|Lead\s+\w+|Founder|Co-[Ff]ounder))"
    r"\s*\(([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\)",
)

_ROLE_KEYWORDS = {
    "cto", "ceo", "cfo", "coo", "vp", "head", "director", "lead",
    "manager", "engineer", "founder", "co-founder", "architect",
    "security", "legal", "procurement", "ops", "infra", "product",
    "ae", "se", "csm", "champion", "sponsor", "dev", "ml",
}

# Blocker/risk indicators
_RE_BLOCKER = re.compile(
    r"^[-\s]*(?:blocker|risk|blocked|concern|issue|flag|warning)[:\s]",
    re.IGNORECASE,
)


def _extract_contacts(lines: list[str]) -> list[str]:
    """Extract key contacts mentioned with roles/titles."""
    contacts: list[str] = []
    seen: set[str] = set()

    for line in lines:
        # Pattern 1: "Role: Name ..."
        for m in _RE_ROLE_NAME.finditer(line):
            role = m.group(1).strip()
            name = m.group(2).strip()
            key = name.lower()
            if key not in seen:
                seen.add(key)
                contacts.append(f"{name} ({role})")

        # Pattern 2: "Name (role)"
        for m in _RE_NAME_ROLE.finditer(line):
            name = m.group(1).strip()
            role_text = m.group(2).strip()
            role_lower = role_text.lower()
            if any(kw in role_lower for kw in _ROLE_KEYWORDS):
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    contacts.append(f"{name} ({role_text})")

        # Pattern 3: "Role (Name)" — inverse pattern
        for m in _RE_ROLE_PARENS_NAME.finditer(line):
            role_text = m.group(1).strip()
            name = m.group(2).strip()
            key = name.lower()
            if key not in seen:
                seen.add(key)
                contacts.append(f"{name} ({role_text})")

    return contacts


def _count_timeline_entries(lines: list[str]) -> int:
    """Count dated timeline/activity entries."""
    return sum(1 for line in lines if _RE_TIMELINE.match(line.strip()))


def _extract_blockers(lines: list[str]) -> list[str]:
    """Extract blocker/risk items."""
    blockers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _RE_BLOCKER.match(stripped):
            blockers.append(stripped[:300])
        # Also catch lines that are clearly blockers by content
        elif stripped.startswith("- ") and any(
            kw in stripped.lower()
            for kw in ("blocker", "risk", "blocked", "concerned", "flagged")
        ):
            blockers.append(stripped[:300])
    return blockers


def _extract_summary(lines: list[str]) -> str:
    """
    Extract a summary from the document.
    Tries the first substantive paragraph after the company name (line 0).
    """
    # Skip line 0 (company name) and any blank lines
    for line in lines[1:]:
        stripped = line.strip()
        # Skip blank lines and short bullet points
        if not stripped:
            continue
        # Skip very short lines (likely bullets or labels)
        if len(stripped) < 30:
            continue
        # Skip timeline entries
        if _RE_TIMELINE.match(stripped):
            continue
        return stripped[:500]

    return ""


def _extract_metadata(file_path: Path, lines: list[str]) -> HubSpotMetadata:
    """Extract metadata from filename and content."""
    fname = file_path.stem
    doc_id = ""
    if fname.startswith("dsid_"):
        id_part, _, _ = fname.partition("__")
        doc_id = id_part

    company_name = lines[0].strip() if lines else ""

    # Check if this is from exports subfolder
    is_export = "exports" in file_path.parts

    contacts = _extract_contacts(lines)
    timeline_count = _count_timeline_entries(lines)
    summary = _extract_summary(lines)
    blockers = _extract_blockers(lines)

    return HubSpotMetadata(
        doc_id=doc_id,
        company_name=company_name,
        contacts=contacts,
        timeline_count=timeline_count,
        summary=summary,
        blockers=blockers,
        is_export=is_export,
        file_path=str(file_path),
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_hubspot_document(
    file_path: Path,
    max_tokens: int = 8192,
    overlap_lines: int = 5,
    tokenizer=None,
) -> list[HubSpotChunk]:
    """
    Chunk a HubSpot CRM record. Whole doc = one chunk in virtually all cases.
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
        return [HubSpotChunk(
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


def _build_context_prefix(metadata: HubSpotMetadata) -> str:
    """Context prefix prepended to chunks after the first so each chunk
    retains awareness of the full CRM record."""
    parts = [metadata.company_name]
    if metadata.summary:
        parts.append(f"Summary: {metadata.summary[:200]}")
    if metadata.contacts:
        parts.append(f"Contacts: {', '.join(metadata.contacts[:3])}")
    return "\n".join(parts) + "\n---\n"


def _split_oversized(
    lines: list[str],
    full_text: str,
    metadata: HubSpotMetadata,
    max_tokens: int,
    overlap_lines: int,
    tokenizer,
) -> list[HubSpotChunk]:
    """Split at paragraph boundaries with overlap and context prefix."""
    context_prefix = _build_context_prefix(metadata)
    prefix_tokens = _count_tokens(context_prefix, tokenizer)

    para_breaks = [i for i, line in enumerate(lines) if not line.strip()]

    if not para_breaks:
        est_lines = max(10, int(max_tokens / 1.3 / 10))
        para_breaks = list(range(est_lines, len(lines), est_lines))

    chunks: list[HubSpotChunk] = []
    chunk_start = 0
    chunk_idx = 0

    for bp in para_breaks:
        segment_text = "\n".join(lines[chunk_start : bp + 1])
        budget = max_tokens if chunk_idx == 0 else max_tokens - prefix_tokens
        if _count_tokens(segment_text, tokenizer) >= budget and bp > chunk_start:
            chunk_text = "\n".join(lines[chunk_start:bp])
            if chunk_idx > 0:
                chunk_text = context_prefix + chunk_text
            chunks.append(HubSpotChunk(
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
        chunks.append(HubSpotChunk(
            text=chunk_text,
            metadata=metadata,
            chunk_index=chunk_idx,
            start_line=chunk_start,
            end_line=len(lines) - 1,
        ))

    return chunks if chunks else [HubSpotChunk(
        text=full_text,
        metadata=metadata,
        chunk_index=0,
        start_line=0,
        end_line=len(lines) - 1,
    )]


def chunk_all_hubspot(hubspot_dir: Path, **kwargs) -> list[HubSpotChunk]:
    """Walk the hubspot directory (including exports/) and chunk every .txt file."""
    all_chunks: list[HubSpotChunk] = []
    files = sorted(hubspot_dir.rglob("*.txt"))
    for fp in files:
        try:
            doc_chunks = chunk_hubspot_document(fp, **kwargs)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"WARN: failed to chunk {fp}: {e}")
    return all_chunks
