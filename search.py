from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

_EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_embedder = SentenceTransformer(_EMBED_MODEL_NAME)

_DOCUMENT_CHUNKS: List[str] = []
_DOCUMENT_EMBEDDINGS: Optional[NDArray[np.floating]] = None


def _split_into_chunks(text: str, max_words: int = 220, overlap: int = 60) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    step = max(1, max_words - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        part = words[start : start + max_words]
        if part:
            chunks.append(" ".join(part))
    return chunks


def build_index_from_text(text: str) -> None:
    global _DOCUMENT_CHUNKS, _DOCUMENT_EMBEDDINGS

    _DOCUMENT_CHUNKS = _split_into_chunks(text)
    if not _DOCUMENT_CHUNKS:
        _DOCUMENT_EMBEDDINGS = None
        return

    emb = _embedder.encode(
        _DOCUMENT_CHUNKS,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _DOCUMENT_EMBEDDINGS = emb


def has_index() -> bool:
    return _DOCUMENT_EMBEDDINGS is not None and len(_DOCUMENT_CHUNKS) > 0


def search_documents(query: str, top: int = 6) -> List[Tuple[str, float]]:
    if not has_index():
        return []
    query = (query or "").strip()
    if not query:
        return []

    assert _DOCUMENT_EMBEDDINGS is not None

    query_emb = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores: NDArray[np.floating] = np.dot(_DOCUMENT_EMBEDDINGS, query_emb)

    best_idx = np.argsort(scores)[::-1][:top].tolist()
    out: List[Tuple[str, float]] = []
    for i in best_idx:
        ii = int(i)
        out.append((_DOCUMENT_CHUNKS[ii], float(scores[ii])))
    return out
