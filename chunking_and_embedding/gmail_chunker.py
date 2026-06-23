"""
Gmail email thread chunker.

Gmail documents are email threads (one or more replies in a single file).
Nearly all fit within the model context window (~100% under 4K tokens),
so the default strategy is whole-thread-as-one-chunk.

Metadata extracted per thread:
  - Original sender (From) and recipients (To/Cc)
  - Date of first email
  - Subject line
  - Number of replies in the thread
  - All participants across the thread
  - Mailbox owner (from directory name)
  - Auto-generated summary (first ~300 chars of substantive content)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GmailMetadata:
    doc_id: str
    subject: str
    original_from: str
    original_to: str
    date: str
    mailbox: str  # directory name — whose mailbox this is from
    reply_count: int  # number of messages in the thread
    participants: list[str]  # all unique people in the thread
    summary: str  # short auto-summary of the thread
    file_path: str


@dataclass
class GmailChunk:
    text: str
    metadata: GmailMetadata
    chunk_index: int
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Email header parsing
# ---------------------------------------------------------------------------

_RE_FROM = re.compile(r"^From:\s*(.+)", re.IGNORECASE)
_RE_TO = re.compile(r"^To:\s*(.+)", re.IGNORECASE)
_RE_CC = re.compile(r"^Cc:\s*(.+)", re.IGNORECASE)
_RE_DATE = re.compile(r"^Date:\s*(.+)", re.IGNORECASE)
_RE_SUBJECT = re.compile(r"^Subject:\s*(.+)", re.IGNORECASE)

# Extract just the name from "Name <email>" or "email"
_RE_NAME_EMAIL = re.compile(r"([^<,]+?)(?:\s*<[^>]+>)?(?:\s*,\s*|$)")


def _extract_name(addr: str) -> str:
    """Extract the human name from 'Name <email>' or return the email."""
    addr = addr.strip()
    m = re.match(r"(.+?)\s*<(.+?)>", addr)
    if m:
        name = m.group(1).strip().strip('"').strip("'")
        return name if name else m.group(2).strip()
    return addr


def _extract_all_names(field: str) -> list[str]:
    """Extract all names from a To/Cc field with multiple addresses."""
    names = []
    for m in _RE_NAME_EMAIL.finditer(field):
        name = m.group(1).strip().strip('"').strip("'")
        if name and not name.startswith("<"):
            names.append(name)
    return names


@dataclass
class _EmailHeader:
    from_field: str
    to_field: str
    cc_field: str
    date: str
    subject: str
    from_name: str
    line_idx: int


def _parse_email_headers(lines: list[str]) -> list[_EmailHeader]:
    """
    Find all email headers in the thread. Each 'From:' line starts a new
    message in the thread.
    """
    headers: list[_EmailHeader] = []
    current: dict = {}
    current_start = -1

    for i, line in enumerate(lines):
        m_from = _RE_FROM.match(line)
        if m_from:
            # Save previous header if exists
            if current:
                headers.append(_make_header(current, current_start))
            current = {"from": m_from.group(1).strip(), "to": "", "cc": "", "date": "", "subject": ""}
            current_start = i
            continue

        if not current:
            continue

        m_to = _RE_TO.match(line)
        if m_to:
            current["to"] = m_to.group(1).strip()
            continue

        m_cc = _RE_CC.match(line)
        if m_cc:
            current["cc"] = m_cc.group(1).strip()
            continue

        m_date = _RE_DATE.match(line)
        if m_date:
            current["date"] = m_date.group(1).strip()
            continue

        m_subj = _RE_SUBJECT.match(line)
        if m_subj:
            current["subject"] = m_subj.group(1).strip()
            continue

    # Final header
    if current:
        headers.append(_make_header(current, current_start))

    return headers


def _make_header(fields: dict, line_idx: int) -> _EmailHeader:
    return _EmailHeader(
        from_field=fields.get("from", ""),
        to_field=fields.get("to", ""),
        cc_field=fields.get("cc", ""),
        date=fields.get("date", ""),
        subject=fields.get("subject", ""),
        from_name=_extract_name(fields.get("from", "")),
        line_idx=line_idx,
    )


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------

_HEADER_PREFIXES = ("From:", "To:", "Cc:", "Date:", "Subject:", "Attachments:", "Attachment:")
_SIGNATURE_MARKERS = {"--", "---", "Best,", "Thanks,", "Regards,", "Cheers,"}


def _extract_summary(lines: list[str], headers: list[_EmailHeader]) -> str:
    """
    Build a short summary from the first email's body content.
    Skips headers, signatures, and attachment lines.
    """
    if not headers:
        # No headers parsed — fall back to first substantial line after title
        for line in lines[1:]:
            stripped = line.strip()
            if len(stripped) > 30 and not any(stripped.startswith(p) for p in _HEADER_PREFIXES):
                return stripped[:400]
        return ""

    # Find where the first email's headers end
    first_header_line = headers[0].line_idx
    # The second email starts at headers[1].line_idx if it exists
    first_email_end = headers[1].line_idx if len(headers) > 1 else len(lines)

    # Scan for body content: skip header lines, blanks, and short greeting lines
    summary_parts = []
    in_body = False

    for i in range(first_header_line, first_email_end):
        line = lines[i].strip()

        # Skip header-like lines
        if any(line.startswith(p) for p in _HEADER_PREFIXES):
            continue

        # Skip blanks before body starts
        if not line:
            if in_body and summary_parts:
                break  # end of first paragraph
            continue

        # Skip signature markers
        if line in _SIGNATURE_MARKERS:
            break

        # Skip short greeting-only lines like "Hi Team," or "Team,"
        if len(line) < 25 and (line.endswith(",") or line.endswith(" —")):
            continue

        # This is body content
        in_body = True
        summary_parts.append(line)

    summary = " ".join(summary_parts)
    return summary[:400]


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(file_path: Path, lines: list[str]) -> GmailMetadata:
    """Extract rich metadata from the email thread."""
    fname = file_path.stem
    doc_id = ""
    if fname.startswith("dsid_"):
        id_part, _, _ = fname.partition("__")
        doc_id = id_part

    mailbox = file_path.parent.name

    headers = _parse_email_headers(lines)
    reply_count = len(headers)

    # First email's fields
    original_from = headers[0].from_name if headers else ""
    original_to = headers[0].to_field if headers else ""
    date = headers[0].date if headers else ""
    subject = headers[0].subject if headers else ""

    # If subject is empty, use the title (line 1)
    if not subject and lines:
        subject = lines[0].strip()

    # Collect all unique participants
    seen: set[str] = set()
    participants: list[str] = []
    for h in headers:
        for name in [h.from_name] + _extract_all_names(h.to_field) + _extract_all_names(h.cc_field):
            if name and name.lower() not in seen:
                seen.add(name.lower())
                participants.append(name)

    summary = _extract_summary(lines, headers)

    return GmailMetadata(
        doc_id=doc_id,
        subject=subject,
        original_from=original_from,
        original_to=original_to,
        date=date,
        mailbox=mailbox,
        reply_count=reply_count,
        participants=participants,
        summary=summary,
        file_path=str(file_path),
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_gmail_document(
    file_path: Path,
    max_tokens: int = 8192,
    overlap_lines: int = 5,
    tokenizer=None,
) -> list[GmailChunk]:
    """
    Chunk a Gmail email thread. Whole thread = one chunk in almost all cases.
    Oversized fallback splits at email boundaries (From: lines) with overlap.
    """
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    # Some emails have literal \n instead of real newlines — normalize
    # Detect: if there are many more literal \n than real newlines, replace them
    literal_count = raw_text.count("\\n")
    real_count = raw_text.count("\n")
    if literal_count > real_count * 2 and literal_count > 10:
        raw_text = raw_text.replace("\\n", "\n")

    lines = raw_text.split("\n")
    metadata = _extract_metadata(file_path, lines)

    token_count = _count_tokens(raw_text, tokenizer)

    # Common case: whole thread as one chunk
    if token_count <= max_tokens:
        return [GmailChunk(
            text=raw_text,
            metadata=metadata,
            chunk_index=0,
            start_line=0,
            end_line=len(lines) - 1,
        )]

    # Oversized fallback: split at email boundaries (From: lines)
    return _split_at_email_boundaries(lines, raw_text, metadata, max_tokens, overlap_lines, tokenizer)


def _count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(text.split()) * 1.3)


def _build_context_prefix(metadata: GmailMetadata) -> str:
    """Context prefix prepended to chunks after the first so each chunk
    retains awareness of the full email thread."""
    parts = [f"Subject: {metadata.subject}"]
    if metadata.original_from:
        parts.append(f"From: {metadata.original_from}")
    if metadata.date:
        parts.append(f"Date: {metadata.date}")
    if metadata.summary:
        parts.append(f"Thread summary: {metadata.summary[:200]}")
    return "\n".join(parts) + "\n---\n"


def _split_at_email_boundaries(
    lines: list[str],
    full_text: str,
    metadata: GmailMetadata,
    max_tokens: int,
    overlap_lines: int,
    tokenizer,
) -> list[GmailChunk]:
    """Split at From: boundaries with context prefix on subsequent chunks."""
    context_prefix = _build_context_prefix(metadata)
    prefix_tokens = _count_tokens(context_prefix, tokenizer)

    # Find email boundaries
    email_starts = [i for i, line in enumerate(lines) if _RE_FROM.match(line)]
    if not email_starts:
        email_starts = [0]

    chunks: list[GmailChunk] = []
    chunk_idx = 0
    current_start = 0

    for j in range(1, len(email_starts)):
        segment = "\n".join(lines[current_start:email_starts[j]])
        budget = max_tokens if chunk_idx == 0 else max_tokens - prefix_tokens
        if _count_tokens(segment, tokenizer) >= budget:
            chunk_text = "\n".join(lines[current_start:email_starts[j]])
            if chunk_idx > 0:
                chunk_text = context_prefix + chunk_text
            chunks.append(GmailChunk(
                text=chunk_text,
                metadata=metadata,
                chunk_index=chunk_idx,
                start_line=current_start,
                end_line=email_starts[j] - 1,
            ))
            chunk_idx += 1
            current_start = max(0, email_starts[j] - overlap_lines)

    # Final segment
    if current_start < len(lines):
        chunk_text = "\n".join(lines[current_start:])
        if chunk_idx > 0:
            chunk_text = context_prefix + chunk_text
        chunks.append(GmailChunk(
            text=chunk_text,
            metadata=metadata,
            chunk_index=chunk_idx,
            start_line=current_start,
            end_line=len(lines) - 1,
        ))

    return chunks if chunks else [GmailChunk(
        text=full_text,
        metadata=metadata,
        chunk_index=0,
        start_line=0,
        end_line=len(lines) - 1,
    )]


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def chunk_all_gmail(gmail_dir: Path, **kwargs) -> list[GmailChunk]:
    """Walk the gmail directory and chunk every .txt file."""
    all_chunks: list[GmailChunk] = []
    files = sorted(gmail_dir.rglob("*.txt"))
    for fp in files:
        try:
            doc_chunks = chunk_gmail_document(fp, **kwargs)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"WARN: failed to chunk {fp}: {e}")
    return all_chunks
