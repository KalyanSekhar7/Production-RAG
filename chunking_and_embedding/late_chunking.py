"""
Late chunking pipeline.

Encodes the full document through a long-context embedding model to get
token-level embeddings, then mean-pools within each chunk's token span.
This gives each chunk embedding contextual awareness of the entire document.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer

from .confluence_chunker import Chunk
from .config import EMBEDDING_MODEL, MAX_CONTEXT_TOKENS


@dataclass
class ChunkEmbedding:
    chunk: Chunk
    embedding: np.ndarray  # shape: (hidden_dim,)


class LateChunkingEncoder:
    """
    Encodes documents using late chunking:
    1. Tokenize the full document
    2. Run through the model to get token-level embeddings (with full attention)
    3. Map chunk text boundaries to token positions
    4. Mean-pool token embeddings within each chunk span
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "mps"
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded.")

    @torch.no_grad()
    def encode_document_chunks(self, full_text: str, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        """
        Late-chunk a single document.

        Args:
            full_text: The complete document text.
            chunks: Pre-computed chunks with text that are substrings of full_text
                    (possibly with overlap prefix stripped for boundary mapping).

        Returns:
            List of ChunkEmbedding with contextual embeddings.
        """
        # Step 1: Tokenize full document
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=MAX_CONTEXT_TOKENS,
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        offset_mapping = encoded["offset_mapping"][0].tolist()  # list of (start_char, end_char)

        # Step 2: Get token-level embeddings from model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state[0].cpu().numpy()  # (seq_len, hidden_dim)

        # Step 3: Map each chunk to token positions using character offsets
        chunk_embeddings = []
        for chunk in chunks:
            # Find the chunk's raw text (strip any overlap prefix we added)
            raw_text = _strip_overlap_prefix(chunk.text)

            # Find the character span of this chunk in the full document
            char_start = full_text.find(raw_text)
            if char_start == -1:
                # Fallback: use the first sentence to locate
                first_line = raw_text.split("\n")[0]
                char_start = full_text.find(first_line)
            if char_start == -1:
                # Last resort: encode chunk independently
                emb = self._encode_standalone(raw_text)
                chunk_embeddings.append(ChunkEmbedding(chunk=chunk, embedding=emb))
                continue

            char_end = char_start + len(raw_text)

            # Map character span to token positions
            tok_start, tok_end = _char_span_to_token_span(
                offset_mapping, char_start, char_end
            )

            if tok_start is None or tok_end is None or tok_start >= tok_end:
                emb = self._encode_standalone(raw_text)
                chunk_embeddings.append(ChunkEmbedding(chunk=chunk, embedding=emb))
                continue

            # Step 4: Mean-pool token embeddings in this span
            span_embeddings = token_embeddings[tok_start:tok_end]
            pooled = span_embeddings.mean(axis=0)
            # L2 normalize
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm

            chunk_embeddings.append(ChunkEmbedding(chunk=chunk, embedding=pooled))

        return chunk_embeddings

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts independently (standard mean-pooling).

        This is used by the fireflies chunker for semantic topic detection:
        each turn's sliding-window text is embedded, and cosine similarity
        between consecutive windows detects topic shifts.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), hidden_dim), L2-normalized.
        """
        all_embeddings = []
        for text in texts:
            emb = self._encode_standalone(text)
            all_embeddings.append(emb)
        return np.stack(all_embeddings)

    @torch.no_grad()
    def _encode_standalone(self, text: str) -> np.ndarray:
        """Fallback: encode a chunk independently (standard embedding)."""
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_CONTEXT_TOKENS,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over all tokens
        token_embs = outputs.last_hidden_state[0].cpu().numpy()
        mask = attention_mask[0].cpu().numpy()
        pooled = (token_embs * mask[:, None]).sum(axis=0) / mask.sum()
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled


def _strip_overlap_prefix(text: str) -> str:
    """Remove the '[...] ...' overlap prefix added by the chunker."""
    if text.startswith("[...]"):
        # Find the end of the overlap block (double newline)
        idx = text.find("\n\n")
        if idx != -1:
            return text[idx + 2:]
    return text


def _char_span_to_token_span(
    offset_mapping: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int | None, int | None]:
    """
    Given a character-level span and the tokenizer's offset mapping,
    return the corresponding token-level (start, end) span.
    """
    tok_start = None
    tok_end = None

    for i, (cs, ce) in enumerate(offset_mapping):
        if cs == 0 and ce == 0:
            # Special token
            continue
        if tok_start is None and ce > char_start:
            tok_start = i
        if cs < char_end:
            tok_end = i + 1

    return tok_start, tok_end


def count_tokens(text: str, tokenizer) -> int:
    """Quick token count for a text string."""
    return len(tokenizer.encode(text, add_special_tokens=False))
