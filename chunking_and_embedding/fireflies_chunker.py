"""
Fireflies meeting transcript chunker.

Fireflies documents have a consistent structure:
  - Title (line 1)
  - Summary paragraph
  - Meeting header (date, time, duration, attendees)
  - Optional: topics, action items
  - Transcript body with timestamped speaker turns: [MM:SS] Speaker: text

Chunking strategy:
  1. Pre-transcript metadata (title, summary, header, action items) → one chunk
  2. Transcript → split into topic-coherent segments using SEMANTIC similarity
     between sliding windows of turns. When cosine similarity between consecutive
     windows drops below a threshold, that's a topic boundary.
  3. Each transcript chunk carries the meeting metadata as context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MeetingMetadata:
    doc_id: str
    title: str
    summary: str
    date: str
    duration: str
    attendees: str
    meeting_type: str  # sales-calls, customer-success, all-hands, etc.
    file_path: str


@dataclass
class MeetingChunk:
    text: str
    metadata: MeetingMetadata
    chunk_type: str  # "metadata" or "transcript"
    section_label: str  # e.g. "Meeting overview", "Transcript [00:00-03:30]"
    chunk_index: int
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_RE_TIMESTAMP_TURN = re.compile(
    r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s+(.+?):\s+(.*)"
)

_RE_MEETING_FIELD = re.compile(
    r"^(Date|Time|Duration|Start|Start time|Attendees|Call type|Customer|"
    r"Redwood attendees|Customer attendees|Meeting type)[:\s]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(file_path: Path, lines: list[str]) -> MeetingMetadata:
    """Extract meeting metadata from filename, path, and content."""
    # Doc ID + title from filename
    fname = file_path.stem
    doc_id = ""
    if fname.startswith("dsid_"):
        id_part, _, _ = fname.partition("__")
        doc_id = id_part

    # Meeting type from parent directory
    meeting_type = file_path.parent.name
    if meeting_type == "fireflies":
        meeting_type = "misc"

    # Title from first line
    title = lines[0].strip() if lines else ""

    # Parse structured fields from the header area (first ~35 lines)
    header_area = lines[:35]
    date = ""
    duration = ""
    attendees_parts = []

    for line in header_area:
        low = line.strip().lower()
        if low.startswith("date:"):
            date = line.split(":", 1)[1].strip()
        elif low.startswith("duration:"):
            duration = line.split(":", 1)[1].strip()
        elif "attendees" in low and ":" in line:
            attendees_parts.append(line.split(":", 1)[1].strip())

    attendees = "; ".join(attendees_parts)

    # Summary: text between title and meeting header/transcript
    summary_lines = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            if summary_lines:
                break
            continue
        if _RE_MEETING_FIELD.match(stripped):
            break
        if _RE_TIMESTAMP_TURN.match(stripped):
            break
        if stripped.lower().startswith(("meeting header", "transcript", "auto-summary")):
            break
        summary_lines.append(stripped)

    summary = " ".join(summary_lines[:5])

    return MeetingMetadata(
        doc_id=doc_id,
        title=title,
        summary=summary,
        date=date,
        duration=duration,
        attendees=attendees,
        meeting_type=meeting_type,
        file_path=str(file_path),
    )


# ---------------------------------------------------------------------------
# Transcript and pre-transcript splitting
# ---------------------------------------------------------------------------

def _find_transcript_start(lines: list[str]) -> int:
    """Find the line index where the transcript begins."""
    for i, line in enumerate(lines):
        if _RE_TIMESTAMP_TURN.match(line.strip()):
            return i
    return len(lines)  # no transcript found


def _build_metadata_chunk(
    lines: list[str], transcript_start: int, metadata: MeetingMetadata
) -> MeetingChunk:
    """Build a single chunk from all pre-transcript content."""
    pre_transcript = "\n".join(lines[:transcript_start]).strip()
    return MeetingChunk(
        text=pre_transcript,
        metadata=metadata,
        chunk_type="metadata",
        section_label="Meeting overview",
        chunk_index=0,
        start_line=0,
        end_line=max(0, transcript_start - 1),
    )


# ---------------------------------------------------------------------------
# Topic-based transcript chunking
# ---------------------------------------------------------------------------

@dataclass
class _Turn:
    timestamp: str
    speaker: str
    text: str
    line_idx: int


def _parse_turns(lines: list[str], start: int) -> list[_Turn]:
    """Parse timestamped speaker turns from transcript lines."""
    turns: list[_Turn] = []
    for i in range(start, len(lines)):
        m = _RE_TIMESTAMP_TURN.match(lines[i].strip())
        if m:
            turns.append(_Turn(
                timestamp=m.group(1),
                speaker=m.group(2).strip(),
                text=m.group(3).strip(),
                line_idx=i,
            ))
        elif turns and lines[i].strip():
            # Continuation line (no timestamp) — append to previous turn
            turns[-1].text += " " + lines[i].strip()
    return turns


def _timestamp_to_seconds(ts: str) -> int:
    """Convert MM:SS or HH:MM:SS to seconds."""
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_window_texts(turns: list[_Turn], window_size: int = 3) -> list[str]:
    """
    Build sliding window texts from turns. Each window concatenates
    `window_size` consecutive turns into a single string for embedding.
    This smooths out noise from individual short turns.
    """
    texts = []
    for i in range(len(turns)):
        window_turns = turns[max(0, i - window_size + 1) : i + 1]
        combined = " ".join(t.text for t in window_turns)
        texts.append(combined)
    return texts


def _find_semantic_breakpoints(
    similarities: list[float],
    similarity_threshold: float,
    min_turns_per_chunk: int,
) -> list[int]:
    """
    Find indices where similarity drops below threshold (topic shifts).
    Returns turn indices where a new chunk should start.

    Uses percentile-based adaptive thresholding: if the provided threshold
    finds no breaks, it falls back to splitting at the deepest valleys.
    """
    if not similarities:
        return []

    breakpoints = []
    since_last_break = 0

    for i, sim in enumerate(similarities):
        since_last_break += 1
        if sim < similarity_threshold and since_last_break >= min_turns_per_chunk:
            breakpoints.append(i + 1)  # split AFTER this position
            since_last_break = 0

    # Fallback: if no breaks found and transcript is long, use percentile valleys
    if not breakpoints and len(similarities) > min_turns_per_chunk * 2:
        # Find the lowest 20th percentile of similarities as candidate breaks
        p20 = float(np.percentile(similarities, 20))
        for i, sim in enumerate(similarities):
            since_last = i - breakpoints[-1] if breakpoints else i
            if sim <= p20 and since_last >= min_turns_per_chunk:
                breakpoints.append(i + 1)

    return breakpoints


def _chunk_transcript_turns(
    turns: list[_Turn],
    embed_fn,
    max_turns_per_chunk: int = 15,
    min_turns_per_chunk: int = 4,
    similarity_threshold: float = 0.5,
    window_size: int = 3,
    time_gap_threshold_sec: int = 45,
) -> list[list[_Turn]]:
    """
    Group turns into topic-coherent segments using semantic similarity.

    How it works:
    1. Build sliding-window text for each turn (concatenates `window_size`
       consecutive turns for smoother embeddings).
    2. Embed all windows using `embed_fn`.
    3. Compute cosine similarity between consecutive windows.
    4. Split where similarity drops below threshold (semantic topic shift)
       OR where there's a time gap > threshold (structural pause).
    5. Enforce max/min turns per chunk constraints.

    Args:
        turns: Parsed speaker turns.
        embed_fn: Callable that takes list[str] and returns np.ndarray of shape
                  (n, embedding_dim). This is the model that decides topic shifts.
        max_turns_per_chunk: Hard cap on turns per chunk.
        min_turns_per_chunk: Minimum turns before allowing a split.
        similarity_threshold: Cosine sim below this = topic shift.
        window_size: Number of turns in the sliding window for embedding.
        time_gap_threshold_sec: Time gap that forces a split regardless of similarity.
    """
    if not turns:
        return []

    if len(turns) <= min_turns_per_chunk:
        return [turns]

    # Step 1: Build sliding window texts and embed them
    window_texts = _build_window_texts(turns, window_size=window_size)
    embeddings = embed_fn(window_texts)  # (n_turns, dim)

    # Step 2: Compute consecutive cosine similarities
    similarities = []
    for i in range(1, len(embeddings)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        similarities.append(sim)

    # Step 3: Find semantic breakpoints
    semantic_breaks = set(_find_semantic_breakpoints(
        similarities, similarity_threshold, min_turns_per_chunk
    ))

    # Step 4: Also find time-gap breakpoints
    time_breaks = set()
    for i in range(1, len(turns)):
        gap = _timestamp_to_seconds(turns[i].timestamp) - _timestamp_to_seconds(turns[i - 1].timestamp)
        if gap >= time_gap_threshold_sec:
            time_breaks.add(i)

    # Combine all breakpoints
    all_breaks = sorted(semantic_breaks | time_breaks)

    # Step 5: Build groups from breakpoints, enforcing max/min constraints
    groups: list[list[_Turn]] = []
    current_start = 0

    for bp in all_breaks:
        if bp - current_start >= min_turns_per_chunk:
            groups.append(turns[current_start:bp])
            current_start = bp

    # Final group
    if current_start < len(turns):
        remaining = turns[current_start:]
        if len(remaining) < min_turns_per_chunk and groups:
            groups[-1].extend(remaining)
        else:
            groups.append(remaining)

    # Enforce max_turns_per_chunk by sub-splitting large groups at lowest similarity
    final_groups: list[list[_Turn]] = []
    for group in groups:
        if len(group) <= max_turns_per_chunk:
            final_groups.append(group)
        else:
            final_groups.extend(
                _subsplit_large_group(group, turns, similarities, max_turns_per_chunk, min_turns_per_chunk)
            )

    return final_groups


def _subsplit_large_group(
    group: list[_Turn],
    all_turns: list[_Turn],
    all_similarities: list[float],
    max_size: int,
    min_size: int,
) -> list[list[_Turn]]:
    """Split a group that exceeds max_size at the lowest-similarity point."""
    if len(group) <= max_size:
        return [group]

    # Find the index of the first turn in this group within all_turns
    first_idx = next(
        i for i, t in enumerate(all_turns)
        if t.line_idx == group[0].line_idx
    )

    # Get similarities within this group
    group_sims = []
    for i in range(1, len(group)):
        sim_idx = first_idx + i - 1
        if sim_idx < len(all_similarities):
            group_sims.append((i, all_similarities[sim_idx]))
        else:
            group_sims.append((i, 1.0))

    # Find valid split points (respecting min_size)
    valid_splits = [
        (idx, sim) for idx, sim in group_sims
        if min_size <= idx <= len(group) - min_size
    ]

    if not valid_splits:
        # No valid split — just hard split at midpoint
        mid = len(group) // 2
        return [group[:mid], group[mid:]]

    # Split at lowest similarity
    best_idx = min(valid_splits, key=lambda x: x[1])[0]
    left = group[:best_idx]
    right = group[best_idx:]

    # Recursively split if still too large
    result = []
    result.extend(_subsplit_large_group(left, all_turns, all_similarities, max_size, min_size))
    result.extend(_subsplit_large_group(right, all_turns, all_similarities, max_size, min_size))
    return result


def _chunk_even(
    turns: list[_Turn],
    max_turns: int,
    min_turns: int,
) -> list[list[_Turn]]:
    """Simple fallback: split into even-sized groups when no embed_fn is available."""
    if len(turns) <= max_turns:
        return [turns]
    groups = []
    for i in range(0, len(turns), max_turns):
        group = turns[i : i + max_turns]
        if len(group) < min_turns and groups:
            groups[-1].extend(group)
        else:
            groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------

def chunk_fireflies_document(
    file_path: Path,
    embed_fn=None,
    max_turns_per_chunk: int = 15,
    min_turns_per_chunk: int = 4,
    similarity_threshold: float = 0.5,
    window_size: int = 3,
    overlap_turns: int = 1,
) -> list[MeetingChunk]:
    """
    Chunk a Fireflies meeting transcript.

    Args:
        file_path: Path to the .txt file.
        embed_fn: Callable that takes list[str] → np.ndarray (n, dim).
                  Used for semantic topic detection. If None, falls back to
                  even-sized chunking (no semantic detection).
        max_turns_per_chunk: Hard cap on turns per chunk.
        min_turns_per_chunk: Minimum turns before allowing a split.
        similarity_threshold: Cosine sim below this = topic shift.
        window_size: Sliding window size for turn embeddings.
        overlap_turns: Number of turns to overlap between chunks.

    Returns:
        List of MeetingChunk objects. First chunk is always the meeting
        metadata/overview. Subsequent chunks are transcript segments.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    metadata = _extract_metadata(file_path, lines)

    transcript_start = _find_transcript_start(lines)

    chunks: list[MeetingChunk] = []

    # Chunk 0: pre-transcript metadata
    if transcript_start > 0:
        meta_chunk = _build_metadata_chunk(lines, transcript_start, metadata)
        if meta_chunk.text.strip():
            chunks.append(meta_chunk)

    # Parse and group transcript turns
    turns = _parse_turns(lines, transcript_start)
    if not turns:
        # No transcript — return just the metadata chunk
        return chunks

    if embed_fn is None:
        # Fallback: even-sized chunks (no semantic detection)
        turn_groups = _chunk_even(turns, max_turns_per_chunk, min_turns_per_chunk)
    else:
        turn_groups = _chunk_transcript_turns(
            turns,
            embed_fn=embed_fn,
            max_turns_per_chunk=max_turns_per_chunk,
            min_turns_per_chunk=min_turns_per_chunk,
            similarity_threshold=similarity_threshold,
            window_size=window_size,
        )

    # Build context prefix for transcript chunks so each carries meeting awareness
    context_prefix = metadata.title
    if metadata.summary:
        context_prefix += f"\nSummary: {metadata.summary[:200]}"
    if metadata.attendees:
        context_prefix += f"\nAttendees: {metadata.attendees[:150]}"
    context_prefix += "\n---\n"

    chunk_idx = len(chunks)
    for group in turn_groups:
        # Build chunk text from turns
        turn_texts = [f"[{t.timestamp}] {t.speaker}: {t.text}" for t in group]
        chunk_text = context_prefix + "\n".join(turn_texts)

        # Time range label
        t_start = group[0].timestamp
        t_end = group[-1].timestamp
        section_label = f"Transcript [{t_start}-{t_end}]"

        chunks.append(MeetingChunk(
            text=chunk_text,
            metadata=metadata,
            chunk_type="transcript",
            section_label=section_label,
            chunk_index=chunk_idx,
            start_line=group[0].line_idx,
            end_line=group[-1].line_idx,
        ))
        chunk_idx += 1

    # Add overlap: prepend last N turns from previous chunk
    if overlap_turns > 0:
        for i in range(1, len(chunks)):
            if chunks[i].chunk_type != "transcript":
                continue
            prev = chunks[i - 1] if chunks[i - 1].chunk_type == "transcript" else None
            if prev is None:
                continue
            prev_lines = prev.text.split("\n")
            overlap = "\n".join(prev_lines[-overlap_turns:])
            # Insert overlap after the context prefix
            chunks[i].text = chunks[i].text.replace(
                "---\n", "---\n" + f"[...prev] {overlap}\n\n", 1
            )

    return chunks


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def chunk_all_fireflies(fireflies_dir: Path, **kwargs) -> list[MeetingChunk]:
    """Walk the fireflies directory and chunk every .txt file."""
    all_chunks: list[MeetingChunk] = []
    files = sorted(fireflies_dir.rglob("*.txt"))
    for fp in files:
        try:
            doc_chunks = chunk_fireflies_document(fp, **kwargs)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"WARN: failed to chunk {fp}: {e}")
    return all_chunks
