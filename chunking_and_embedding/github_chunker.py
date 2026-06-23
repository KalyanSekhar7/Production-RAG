"""
GitHub document chunker.

GitHub items (PRs, issues) are short enough to embed as whole documents
in nearly all cases (median ~800 tokens, max ~2500). Strategy:

  - Default: entire document = one chunk, embedded with full context.
  - Fallback: if a document exceeds the model context window, split into
    overlapping segments at paragraph boundaries.

Metadata is extracted from the filename (repo, PR number, doc_id) and
from the content (title, author, reviewers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitHubMetadata:
    doc_id: str
    title: str
    repo: str
    pr_number: str
    summary: str  # motivation/summary extracted from content
    author: str  # PR author if detectable
    reviewers: list[str]  # reviewers who commented/approved
    file_path: str


@dataclass
class GitHubChunk:
    text: str
    metadata: GitHubMetadata
    chunk_index: int
    start_line: int
    end_line: int


_RE_PR_NUMBER = re.compile(r"pr-(\d+)")


_RE_AUTHOR = re.compile(r"^(?:\d{4}-\d{2}-\d{2}\s+)?(.+?)\s*\(author\)", re.IGNORECASE)
_RE_REVIEWER = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}\s+)?(.+?):\s*(?:approved|lgtm|looks good|approving)",
    re.IGNORECASE,
)


def _extract_metadata(file_path: Path, lines: list[str]) -> GitHubMetadata:
    """Extract metadata from filename and content."""
    fname = file_path.stem
    doc_id = ""
    pr_number = ""

    if fname.startswith("dsid_"):
        id_part, _, slug = fname.partition("__")
        doc_id = id_part
        m = _RE_PR_NUMBER.search(slug)
        if m:
            pr_number = m.group(1)

    repo = file_path.parent.name
    title = lines[0].strip() if lines else ""

    # Extract summary: first paragraph that looks like motivation/summary/context
    summary = _extract_summary(lines)

    # Extract author and reviewers from review comments
    author = ""
    reviewers: list[str] = []
    seen_reviewers: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not author:
            m = _RE_AUTHOR.match(stripped)
            if m:
                author = m.group(1).strip()
        m = _RE_REVIEWER.match(stripped)
        if m:
            name = m.group(1).strip()
            if name not in seen_reviewers and name != author:
                reviewers.append(name)
                seen_reviewers.add(name)

    return GitHubMetadata(
        doc_id=doc_id,
        title=title,
        repo=repo,
        pr_number=pr_number,
        summary=summary,
        author=author,
        reviewers=reviewers,
        file_path=str(file_path),
    )


def _extract_summary(lines: list[str]) -> str:
    """
    Extract a concise summary from the PR body. Looks for:
    - A 'Summary:' or 'Motivation:' or 'Context:' labeled section
    - Falling back to the first substantial paragraph after the title
    """
    # Try to find a labeled summary/motivation
    for i, line in enumerate(lines[:20]):
        low = line.strip().lower()
        if low.startswith(("summary:", "motivation:", "context:")):
            # Grab the rest of this line + continuation lines
            content = line.split(":", 1)[1].strip()
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    break
                content += " " + next_line
            if len(content) > 20:
                return content[:500]

    # Fallback: first substantial paragraph after title
    for i, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if len(stripped) > 50:
            return stripped[:500]

    return ""


def chunk_github_document(
    file_path: Path,
    max_tokens: int = 8192,
    overlap_lines: int = 5,
    tokenizer=None,
) -> list[GitHubChunk]:
    """
    Chunk a single GitHub document.

    Nearly all GitHub items fit as a single chunk. For the rare oversized
    doc, split at paragraph boundaries with overlap.

    Args:
        file_path: Path to the .txt file.
        max_tokens: Model context window size.
        overlap_lines: Lines of overlap between segments for oversized docs.
        tokenizer: Optional tokenizer for accurate token counting.
                   If None, uses a rough word-based estimate.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    metadata = _extract_metadata(file_path, lines)

    token_count = _count_tokens(text, tokenizer)

    # Common case: whole document as one chunk
    if token_count <= max_tokens:
        return [GitHubChunk(
            text=text,
            metadata=metadata,
            chunk_index=0,
            start_line=0,
            end_line=len(lines) - 1,
        )]

    # Rare fallback: split at paragraph boundaries with overlap
    return _split_oversized(lines, metadata, max_tokens, overlap_lines, tokenizer)


def _count_tokens(text: str, tokenizer=None) -> int:
    """Token count — exact if tokenizer provided, rough estimate otherwise."""
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(text.split()) * 1.3)


def _build_context_prefix(metadata: GitHubMetadata) -> str:
    """Context prefix prepended to chunks after the first so each chunk
    retains awareness of the full document."""
    parts = [f"[{metadata.repo} PR #{metadata.pr_number}] {metadata.title}"]
    if metadata.summary:
        parts.append(f"Summary: {metadata.summary[:200]}")
    if metadata.author:
        parts.append(f"Author: {metadata.author}")
    return "\n".join(parts) + "\n---\n"


def _split_oversized(
    lines: list[str],
    metadata: GitHubMetadata,
    max_tokens: int,
    overlap_lines: int,
    tokenizer,
) -> list[GitHubChunk]:
    """Split an oversized document at paragraph breaks with overlap.
    Prepends a context prefix to each chunk after the first."""
    context_prefix = _build_context_prefix(metadata)
    prefix_tokens = _count_tokens(context_prefix, tokenizer)

    # Find paragraph boundaries (blank lines)
    para_breaks = [i for i, line in enumerate(lines) if not line.strip()]

    if not para_breaks:
        # No paragraph breaks — hard split by line count
        est_lines_per_chunk = max(10, int(max_tokens / 1.3 / 10))
        para_breaks = list(range(est_lines_per_chunk, len(lines), est_lines_per_chunk))

    # Greedily accumulate paragraphs up to the token budget
    chunks: list[GitHubChunk] = []
    chunk_start = 0
    chunk_idx = 0

    for bp in para_breaks:
        segment_text = "\n".join(lines[chunk_start : bp + 1])
        budget = max_tokens if chunk_idx == 0 else max_tokens - prefix_tokens
        if _count_tokens(segment_text, tokenizer) >= budget and bp > chunk_start:
            # Close current chunk at previous break
            chunk_text = "\n".join(lines[chunk_start:bp])
            if chunk_idx > 0:
                chunk_text = context_prefix + chunk_text
            chunks.append(GitHubChunk(
                text=chunk_text,
                metadata=metadata,
                chunk_index=chunk_idx,
                start_line=chunk_start,
                end_line=bp - 1,
            ))
            chunk_idx += 1
            chunk_start = max(0, bp - overlap_lines)

    # Final segment
    if chunk_start < len(lines):
        chunk_text = "\n".join(lines[chunk_start:])
        if chunk_idx > 0:
            chunk_text = context_prefix + chunk_text
        chunks.append(GitHubChunk(
            text=chunk_text,
            metadata=metadata,
            chunk_index=chunk_idx,
            start_line=chunk_start,
            end_line=len(lines) - 1,
        ))

    return chunks if chunks else [GitHubChunk(
        text="\n".join(lines),
        metadata=metadata,
        chunk_index=0,
        start_line=0,
        end_line=len(lines) - 1,
    )]


def chunk_all_github(github_dir: Path, **kwargs) -> list[GitHubChunk]:
    """Walk the github directory and chunk every .txt file."""
    all_chunks: list[GitHubChunk] = []
    files = sorted(github_dir.rglob("*.txt"))
    for fp in files:
        try:
            doc_chunks = chunk_github_document(fp, **kwargs)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"WARN: failed to chunk {fp}: {e}")
    return all_chunks
