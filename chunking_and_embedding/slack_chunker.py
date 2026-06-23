"""
Slack conversation chunker.

Slack files are channel threads/conversations with multi-person discussions.
Structure:
  - Line 1: channel name
  - Messages: "name: message text" format, may include code blocks, links, bot msgs

285,605 files across 37 channels. Median ~616 tokens, p99 ~1,276 tokens.
Most are short and fit in 8K as one chunk.

Chunking strategy:
  - Short threads (fits in 8K): whole thread = one chunk.
  - Longer threads: semantic similarity-based splitting. Embed sliding windows
    of consecutive messages, split where cosine similarity drops (topic shift).
    Same approach as fireflies transcripts but adapted for Slack's format.

Metadata extracted:
  - channel: channel name (first line or parent directory)
  - participants: unique speakers in the thread
  - message_count: total messages in the thread
  - summary: first substantive message
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SlackMetadata:
    doc_id: str
    channel: str
    participants: list[str]
    message_count: int
    summary: str
    file_path: str


@dataclass
class SlackChunk:
    text: str
    metadata: SlackMetadata
    chunk_index: int
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------

# Matches "name: message" — name is typically lowercase, may have hyphens/underscores
_RE_MESSAGE = re.compile(r"^([a-zA-Z][\w._-]*)\s*:\s+(.*)")

# Bot messages: "merge-bot:", "infra-bot:", "security-bot:", etc.
_RE_BOT = re.compile(r".*-bot$", re.IGNORECASE)


@dataclass
class _Message:
    speaker: str
    text: str
    line_idx: int
    is_bot: bool


def _parse_messages(lines: list[str], start: int) -> list[_Message]:
    """Parse speaker messages from conversation lines."""
    messages: list[_Message] = []
    for i in range(start, len(lines)):
        m = _RE_MESSAGE.match(lines[i])
        if m:
            speaker = m.group(1).strip()
            text = m.group(2).strip()
            is_bot = bool(_RE_BOT.match(speaker))
            messages.append(_Message(
                speaker=speaker,
                text=text,
                line_idx=i,
                is_bot=is_bot,
            ))
        elif messages and lines[i].strip():
            # Continuation line (code block, multi-line message)
            messages[-1].text += "\n" + lines[i]
    return messages


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(file_path: Path, lines: list[str], messages: list[_Message]) -> SlackMetadata:
    """Extract metadata from filename, path, and content."""
    fname = file_path.stem
    doc_id = ""
    if fname.startswith("dsid_"):
        id_part, _, _ = fname.partition("__")
        doc_id = id_part

    # Channel: first line if it looks like a channel name, else parent dir
    channel = ""
    if lines and not _RE_MESSAGE.match(lines[0]):
        channel = lines[0].strip()
    if not channel:
        channel = file_path.parent.name

    # Participants: unique non-bot speakers
    seen: set[str] = set()
    participants: list[str] = []
    for msg in messages:
        if not msg.is_bot and msg.speaker.lower() not in seen:
            seen.add(msg.speaker.lower())
            participants.append(msg.speaker)

    # Summary: first substantive human message
    summary = ""
    for msg in messages:
        if not msg.is_bot and len(msg.text) > 30:
            summary = msg.text[:500]
            break

    return SlackMetadata(
        doc_id=doc_id,
        channel=channel,
        participants=participants,
        message_count=len(messages),
        summary=summary,
        file_path=str(file_path),
    )


# ---------------------------------------------------------------------------
# Semantic similarity utilities (shared approach with fireflies)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_window_texts(messages: list[_Message], window_size: int = 3) -> list[str]:
    """
    Build sliding window texts from messages. Each window concatenates
    `window_size` consecutive messages for smoother embeddings.
    """
    texts = []
    for i in range(len(messages)):
        window = messages[max(0, i - window_size + 1) : i + 1]
        combined = " ".join(f"{m.speaker}: {m.text}" for m in window)
        texts.append(combined)
    return texts


def _find_semantic_breakpoints(
    similarities: list[float],
    threshold: float,
    min_msgs: int,
) -> list[int]:
    """Find indices where similarity drops below threshold (topic shifts)."""
    if not similarities:
        return []

    breakpoints = []
    since_last = 0

    for i, sim in enumerate(similarities):
        since_last += 1
        if sim < threshold and since_last >= min_msgs:
            breakpoints.append(i + 1)
            since_last = 0

    # Adaptive fallback: if no breaks found, use percentile valleys
    if not breakpoints and len(similarities) > min_msgs * 2:
        p20 = float(np.percentile(similarities, 20))
        for i, sim in enumerate(similarities):
            since_last_bp = i - breakpoints[-1] if breakpoints else i
            if sim <= p20 and since_last_bp >= min_msgs:
                breakpoints.append(i + 1)

    return breakpoints


def _subsplit_large_group(
    group: list[_Message],
    all_messages: list[_Message],
    all_similarities: list[float],
    max_size: int,
    min_size: int,
) -> list[list[_Message]]:
    """Split a group that exceeds max_size at the lowest-similarity point."""
    if len(group) <= max_size:
        return [group]

    first_idx = next(
        i for i, m in enumerate(all_messages)
        if m.line_idx == group[0].line_idx
    )

    group_sims = []
    for i in range(1, len(group)):
        sim_idx = first_idx + i - 1
        if sim_idx < len(all_similarities):
            group_sims.append((i, all_similarities[sim_idx]))
        else:
            group_sims.append((i, 1.0))

    valid_splits = [
        (idx, sim) for idx, sim in group_sims
        if min_size <= idx <= len(group) - min_size
    ]

    if not valid_splits:
        mid = len(group) // 2
        return [group[:mid], group[mid:]]

    best_idx = min(valid_splits, key=lambda x: x[1])[0]
    left = group[:best_idx]
    right = group[best_idx:]

    result = []
    result.extend(_subsplit_large_group(left, all_messages, all_similarities, max_size, min_size))
    result.extend(_subsplit_large_group(right, all_messages, all_similarities, max_size, min_size))
    return result


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_messages_semantic(
    messages: list[_Message],
    embed_fn,
    max_msgs_per_chunk: int = 20,
    min_msgs_per_chunk: int = 4,
    similarity_threshold: float = 0.5,
    window_size: int = 3,
) -> list[list[_Message]]:
    """Group messages into topic-coherent segments using semantic similarity."""
    if not messages:
        return []

    if len(messages) <= min_msgs_per_chunk:
        return [messages]

    # Embed sliding windows
    window_texts = _build_window_texts(messages, window_size=window_size)
    embeddings = embed_fn(window_texts)

    # Consecutive cosine similarities
    similarities = []
    for i in range(1, len(embeddings)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        similarities.append(sim)

    # Find breakpoints
    semantic_breaks = set(_find_semantic_breakpoints(
        similarities, similarity_threshold, min_msgs_per_chunk
    ))

    all_breaks = sorted(semantic_breaks)

    # Build groups
    groups: list[list[_Message]] = []
    current_start = 0

    for bp in all_breaks:
        if bp - current_start >= min_msgs_per_chunk:
            groups.append(messages[current_start:bp])
            current_start = bp

    if current_start < len(messages):
        remaining = messages[current_start:]
        if len(remaining) < min_msgs_per_chunk and groups:
            groups[-1].extend(remaining)
        else:
            groups.append(remaining)

    # Enforce max constraint
    final_groups: list[list[_Message]] = []
    for group in groups:
        if len(group) <= max_msgs_per_chunk:
            final_groups.append(group)
        else:
            final_groups.extend(
                _subsplit_large_group(group, messages, similarities, max_msgs_per_chunk, min_msgs_per_chunk)
            )

    return final_groups


def _chunk_messages_even(
    messages: list[_Message],
    max_msgs: int,
    min_msgs: int,
) -> list[list[_Message]]:
    """Fallback: split into even-sized groups when no embed_fn is available."""
    if len(messages) <= max_msgs:
        return [messages]
    groups = []
    for i in range(0, len(messages), max_msgs):
        group = messages[i : i + max_msgs]
        if len(group) < min_msgs and groups:
            groups[-1].extend(group)
        else:
            groups.append(group)
    return groups


def chunk_slack_document(
    file_path: Path,
    embed_fn=None,
    max_tokens: int = 8192,
    max_msgs_per_chunk: int = 20,
    min_msgs_per_chunk: int = 4,
    similarity_threshold: float = 0.5,
    window_size: int = 3,
    overlap_msgs: int = 1,
    tokenizer=None,
) -> list[SlackChunk]:
    """
    Chunk a Slack conversation thread.

    Short threads: whole thread = one chunk.
    Long threads: semantic similarity-based topic splitting.

    Args:
        file_path: Path to the .txt file.
        embed_fn: Callable that takes list[str] → np.ndarray (n, dim).
                  Used for semantic topic detection. If None, falls back to
                  even-sized chunking.
        max_tokens: Token budget for whole-thread-as-one-chunk check.
        max_msgs_per_chunk: Hard cap on messages per chunk.
        min_msgs_per_chunk: Minimum messages before allowing a split.
        similarity_threshold: Cosine sim below this = topic shift.
        window_size: Sliding window size for message embeddings.
        overlap_msgs: Number of messages to overlap between chunks.
        tokenizer: Optional tokenizer for accurate token counting.
    """
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")

    # Normalize literal \n if needed
    literal_count = raw_text.count("\\n")
    real_count = raw_text.count("\n")
    if literal_count > real_count * 2 and literal_count > 10:
        raw_text = raw_text.replace("\\n", "\n")

    lines = raw_text.split("\n")

    # Determine where messages start (skip channel name header)
    msg_start = 0
    if lines and not _RE_MESSAGE.match(lines[0]):
        msg_start = 1
        # Skip blank lines after channel name
        while msg_start < len(lines) and not lines[msg_start].strip():
            msg_start += 1

    messages = _parse_messages(lines, msg_start)
    metadata = _extract_metadata(file_path, lines, messages)

    # Short thread: whole thing as one chunk
    token_count = _count_tokens(raw_text, tokenizer)
    if token_count <= max_tokens:
        return [SlackChunk(
            text=raw_text,
            metadata=metadata,
            chunk_index=0,
            start_line=0,
            end_line=len(lines) - 1,
        )]

    # Long thread: semantic splitting
    if not messages:
        return [SlackChunk(
            text=raw_text,
            metadata=metadata,
            chunk_index=0,
            start_line=0,
            end_line=len(lines) - 1,
        )]

    if embed_fn is not None:
        msg_groups = _chunk_messages_semantic(
            messages,
            embed_fn=embed_fn,
            max_msgs_per_chunk=max_msgs_per_chunk,
            min_msgs_per_chunk=min_msgs_per_chunk,
            similarity_threshold=similarity_threshold,
            window_size=window_size,
        )
    else:
        msg_groups = _chunk_messages_even(messages, max_msgs_per_chunk, min_msgs_per_chunk)

    # Build context prefix for chunks after the first
    context_prefix = f"#{metadata.channel}\n"
    if metadata.summary:
        context_prefix += f"Thread context: {metadata.summary[:200]}\n"
    if metadata.participants:
        context_prefix += f"Participants: {', '.join(metadata.participants[:5])}\n"
    context_prefix += "---\n"

    channel_header = f"#{metadata.channel}\n\n"
    chunks: list[SlackChunk] = []

    for idx, group in enumerate(msg_groups):
        msg_texts = [f"{m.speaker}: {m.text}" for m in group]
        if idx == 0:
            chunk_text = channel_header + "\n".join(msg_texts)
        else:
            chunk_text = context_prefix + "\n".join(msg_texts)

        chunks.append(SlackChunk(
            text=chunk_text,
            metadata=metadata,
            chunk_index=idx,
            start_line=group[0].line_idx,
            end_line=group[-1].line_idx,
        ))

    # Add overlap: prepend last N messages from previous chunk
    if overlap_msgs > 0:
        for i in range(1, len(chunks)):
            prev_lines = chunks[i - 1].text.split("\n")
            overlap = "\n".join(prev_lines[-overlap_msgs:])
            # Insert overlap after the context prefix
            sep = "---\n"
            if sep in chunks[i].text:
                chunks[i].text = chunks[i].text.replace(
                    sep, sep + f"[...prev] {overlap}\n\n", 1
                )

    return chunks


def _count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(text.split()) * 1.3)


def chunk_all_slack(slack_dir: Path, **kwargs) -> list[SlackChunk]:
    """Walk the slack directory and chunk every .txt file."""
    all_chunks: list[SlackChunk] = []
    files = sorted(slack_dir.rglob("*.txt"))
    for fp in files:
        try:
            doc_chunks = chunk_slack_document(fp, **kwargs)
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"WARN: failed to chunk {fp}: {e}")
    return all_chunks
