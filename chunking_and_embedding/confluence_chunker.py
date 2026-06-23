"""
Confluence document section-boundary detector and chunker.

Detects section headers, identifies atomic units (tables, code blocks,
numbered procedures, checklists), and produces chunk boundaries suitable
for late chunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentMetadata:
    doc_id: str
    title: str
    summary: str
    space: str  # e.g. "customer-success-and-support"
    section: str  # e.g. "enterprise-onboarding"
    file_path: str


@dataclass
class Chunk:
    text: str
    metadata: DocumentMetadata
    section_header: str
    chunk_index: int
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Header detection patterns
# ---------------------------------------------------------------------------

# Markdown-style: ## Header
_RE_MD_HEADER = re.compile(r"^#{1,4}\s+(.+)$")

# Underline-style: Header\n===== or Header\n-----
_RE_UNDERLINE = re.compile(r"^[=\-]{3,}\s*$")

# Colon-terminated header (short line ending with colon, no leading whitespace)
# e.g.  "Summary:", "Goals:", "Rollback criteria (automatic and manual):"
_RE_COLON_HEADER = re.compile(
    r"^([A-Z][A-Za-z0-9 ,/&\-\(\)]+):[ \t]*$"
)

def _is_header_line(line: str, next_line: str | None) -> tuple[bool, str]:
    """Return (is_header, header_text) for a given line."""
    # Markdown header
    m = _RE_MD_HEADER.match(line)
    if m:
        return True, m.group(1).strip()

    # Underline on next line → current line is a header
    if next_line is not None and _RE_UNDERLINE.match(next_line) and line.strip():
        return True, line.strip()

    # Colon-terminated header (must be relatively short)
    m = _RE_COLON_HEADER.match(line)
    if m and len(line) < 120:
        return True, m.group(1).strip()

    return False, ""


# ---------------------------------------------------------------------------
# Atomic unit detection
# ---------------------------------------------------------------------------

def _detect_atomic_spans(lines: list[str]) -> list[tuple[int, int]]:
    """
    Return list of (start, end) line indices for atomic units that should
    never be split: tables, code blocks, numbered step sequences, checklists.
    Spans may overlap; caller should merge.
    """
    spans: list[tuple[int, int]] = []
    n = len(lines)
    i = 0

    while i < n:
        line = lines[i]

        # --- pipe-delimited table ---
        if "|" in line and line.strip().startswith("|"):
            start = i
            while i < n and "|" in lines[i] and lines[i].strip().startswith("|"):
                i += 1
            if i - start >= 2:  # at least header + separator
                spans.append((start, i - 1))
            continue

        # --- fenced code block (``` or ~~~) ---
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            fence = line.strip()[:3]
            start = i
            i += 1
            while i < n and not lines[i].strip().startswith(fence):
                i += 1
            spans.append((start, min(i, n - 1)))
            i += 1
            continue

        # --- shebang script block ---
        if line.strip().startswith("#!"):
            start = i
            i += 1
            while i < n and lines[i].strip() and not _is_header_line(lines[i], lines[i + 1] if i + 1 < n else None)[0]:
                i += 1
            spans.append((start, i - 1))
            continue

        # --- checklist block (- [ ] items) ---
        if re.match(r"^\s*-\s*\[[ x]\]", line):
            start = i
            while i < n and re.match(r"^\s*-\s*\[[ x]\]", lines[i]):
                i += 1
            spans.append((start, i - 1))
            continue

        # --- consecutive numbered steps (1. 2. 3. or 1) 2) 3)) ---
        if re.match(r"^\s*\d+[.)]\s", line):
            start = i
            while i < n and (
                re.match(r"^\s*\d+[.)]\s", lines[i])
                or (lines[i].strip() and not _is_header_line(lines[i], lines[i + 1] if i + 1 < n else None)[0] and not re.match(r"^\s*$", lines[i]))
            ):
                i += 1
            # Only mark as atomic if there were at least 2 numbered items
            num_items = sum(1 for l in lines[start:i] if re.match(r"^\s*\d+[.)]\s", l))
            if num_items >= 2:
                spans.append((start, i - 1))
            continue

        i += 1

    return _merge_spans(spans)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping spans."""
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(file_path: Path) -> DocumentMetadata:
    """Extract document metadata from file path and first few lines."""
    # Parse path:  .../confluence/<space>/<section>/dsid_xxx__title.txt
    parts = file_path.parts
    try:
        conf_idx = parts.index("confluence")
    except ValueError:
        conf_idx = -3  # fallback

    space = parts[conf_idx + 1] if conf_idx + 1 < len(parts) else "unknown"

    # Section could be missing if file is directly under the space
    if conf_idx + 3 < len(parts):
        # There's at least space/section/file
        section = parts[conf_idx + 2]
    else:
        section = ""

    # Doc ID from filename
    fname = file_path.stem
    doc_id = ""
    title_from_fname = fname
    if fname.startswith("dsid_"):
        id_part, _, slug = fname.partition("__")
        doc_id = id_part
        title_from_fname = slug.replace("-", " ").replace("_", " ").strip()

    # Read first lines for actual title and summary
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    title = lines[0].strip() if lines else title_from_fname
    # Summary: grab text between the title and the first real section header
    summary_lines = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            if summary_lines:
                break
            continue
        is_hdr, _ = _is_header_line(line, None)
        if is_hdr and summary_lines:
            break
        summary_lines.append(stripped)
    summary = " ".join(summary_lines[:5])  # cap at 5 lines

    return DocumentMetadata(
        doc_id=doc_id,
        title=title,
        summary=summary,
        space=space,
        section=section,
        file_path=str(file_path),
    )


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    header: str
    start_line: int
    end_line: int  # inclusive
    lines: list[str] = field(default_factory=list)


def _split_into_sections(lines: list[str]) -> list[_Section]:
    """Split document lines into sections based on detected headers."""
    sections: list[_Section] = []
    current_header = "Introduction"
    current_start = 0

    i = 0
    while i < len(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        is_hdr, hdr_text = _is_header_line(lines[i], next_line)

        if is_hdr and i > 0:
            # Close previous section
            sections.append(_Section(
                header=current_header,
                start_line=current_start,
                end_line=i - 1,
                lines=lines[current_start:i],
            ))
            current_header = hdr_text
            current_start = i
            # Skip underline if present
            if next_line is not None and _RE_UNDERLINE.match(next_line):
                i += 1

        i += 1

    # Final section
    if current_start < len(lines):
        sections.append(_Section(
            header=current_header,
            start_line=current_start,
            end_line=len(lines) - 1,
            lines=lines[current_start:],
        ))

    return sections


# ---------------------------------------------------------------------------
# Main chunking logic
# ---------------------------------------------------------------------------

def chunk_confluence_document(
    file_path: Path,
    min_chunk_lines: int = 4,
    max_chunk_lines: int = 60,
    merge_threshold_lines: int = 6,
    overlap_lines: int = 2,
) -> list[Chunk]:
    """
    Chunk a single Confluence document into sections, respecting atomic
    units and size constraints.

    Returns a list of Chunk objects with text and metadata. Chunk boundaries
    (as line ranges) are suitable for late chunking token-position mapping.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    metadata = extract_metadata(file_path)
    atomic_spans = _detect_atomic_spans(lines)
    sections = _split_into_sections(lines)

    # Merge tiny sections with their successor
    merged_sections: list[_Section] = []
    for sec in sections:
        if (
            merged_sections
            and len(merged_sections[-1].lines) < merge_threshold_lines
        ):
            # Merge previous tiny section into this one
            prev = merged_sections[-1]
            prev.end_line = sec.end_line
            prev.lines = lines[prev.start_line : sec.end_line + 1]
            # Keep the header of whichever is more descriptive
            if len(sec.header) > len(prev.header):
                prev.header = sec.header
        else:
            merged_sections.append(sec)

    # If the last section is tiny, merge it into the previous one
    if (
        len(merged_sections) > 1
        and len(merged_sections[-1].lines) < merge_threshold_lines
    ):
        prev = merged_sections[-2]
        last = merged_sections[-1]
        prev.end_line = last.end_line
        prev.lines = lines[prev.start_line : last.end_line + 1]
        merged_sections.pop()

    # Sub-split large sections on paragraph boundaries, respecting atomic units
    chunks: list[Chunk] = []
    chunk_idx = 0

    for sec in merged_sections:
        sec_lines = sec.lines
        if len(sec_lines) <= max_chunk_lines:
            chunks.append(Chunk(
                text="\n".join(sec_lines),
                metadata=metadata,
                section_header=sec.header,
                chunk_index=chunk_idx,
                start_line=sec.start_line,
                end_line=sec.end_line,
            ))
            chunk_idx += 1
        else:
            # Sub-split on blank-line paragraph boundaries
            sub_chunks = _subsplit_section(
                sec, lines, atomic_spans, max_chunk_lines, min_chunk_lines
            )
            for sc_start, sc_end in sub_chunks:
                chunk_text = "\n".join(lines[sc_start : sc_end + 1])
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=metadata,
                    section_header=sec.header,
                    chunk_index=chunk_idx,
                    start_line=sc_start,
                    end_line=sc_end,
                ))
                chunk_idx += 1

    # Add overlap context from previous chunk
    if overlap_lines > 0:
        for i in range(1, len(chunks)):
            prev_lines = chunks[i - 1].text.split("\n")
            overlap = "\n".join(prev_lines[-overlap_lines:])
            chunks[i].text = f"[...] {overlap}\n\n{chunks[i].text}"

    return chunks


def _subsplit_section(
    sec: _Section,
    all_lines: list[str],
    atomic_spans: list[tuple[int, int]],
    max_lines: int,
    min_lines: int,
) -> list[tuple[int, int]]:
    """
    Sub-split a large section into smaller chunks at paragraph boundaries
    (blank lines), avoiding splits inside atomic spans.
    Returns list of (start_line, end_line) tuples in document-level indices.
    """
    abs_start = sec.start_line
    abs_end = sec.end_line

    # Find paragraph break points (blank lines) within this section
    break_points = []
    for i in range(abs_start, abs_end + 1):
        if not all_lines[i].strip():
            # Check this break isn't inside an atomic span
            inside_atomic = any(s <= i <= e for s, e in atomic_spans)
            if not inside_atomic:
                break_points.append(i)

    if not break_points:
        # No safe break points — return the whole section as one chunk
        return [(abs_start, abs_end)]

    # Greedily build sub-chunks up to max_lines
    sub_chunks: list[tuple[int, int]] = []
    current_start = abs_start

    for bp in break_points:
        chunk_len = bp - current_start
        if chunk_len >= max_lines:
            # Close this sub-chunk just before the break
            sub_chunks.append((current_start, bp - 1))
            current_start = bp + 1

    # Final sub-chunk
    if current_start <= abs_end:
        if sub_chunks and (abs_end - current_start + 1) < min_lines:
            # Too small — merge into previous
            prev_start, _ = sub_chunks[-1]
            sub_chunks[-1] = (prev_start, abs_end)
        else:
            sub_chunks.append((current_start, abs_end))

    return sub_chunks if sub_chunks else [(abs_start, abs_end)]


# ---------------------------------------------------------------------------
# Convenience: chunk all confluence files
# ---------------------------------------------------------------------------

def chunk_all_confluence(confluence_dir: Path, **kwargs) -> list[Chunk]:
    """Walk a confluence directory tree and chunk every .txt file."""
    all_chunks: list[Chunk] = []
    files = sorted(confluence_dir.rglob("*.txt"))
    for fp in files:
        try:
            doc_chunks = chunk_confluence_document(fp, **kwargs)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"WARN: failed to chunk {fp}: {e}")
    return all_chunks
