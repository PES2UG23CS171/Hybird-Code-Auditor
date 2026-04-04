from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import GuidelineChunk


@dataclass
class GuidelineStore:
    chunks: list[GuidelineChunk]
    vectorizer: TfidfVectorizer
    index: faiss.Index
    matrix: np.ndarray

    def search(self, query: str, top_k: int = 3) -> list[GuidelineChunk]:
        query_vector = self.vectorizer.transform([query]).astype(np.float32).toarray()
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        results: list[GuidelineChunk] = []
        for score, index in zip(distances[0], indices[0]):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            results.append(GuidelineChunk(text=chunk.text, source=chunk.source, score=float(score)))
        return results


def build_default_guideline_store() -> GuidelineStore:
    base_path = Path(__file__).resolve().parent.parent / "data" / "guidelines"
    texts: list[GuidelineChunk] = []
    for file_path in sorted(base_path.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        for chunk in _chunk_text(content, source=file_path.name):
            texts.append(chunk)

    documents = [chunk.text for chunk in texts]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents).astype(np.float32).toarray()
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return GuidelineStore(chunks=texts, vectorizer=vectorizer, index=index, matrix=matrix)


def _chunk_text(text: str, source: str, chunk_size: int = 700, overlap: int = 120) -> list[GuidelineChunk]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[GuidelineChunk] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk_text = normalized[start:end].strip()
        if chunk_text:
            chunks.append(GuidelineChunk(text=chunk_text, source=source))
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks
