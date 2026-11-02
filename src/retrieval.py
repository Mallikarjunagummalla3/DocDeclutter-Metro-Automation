"""
DocDeclutter — Hybrid Retrieval Engine
Combines Dense Semantic Search and Keyword BM25 Search with Reciprocal Rank Fusion (Novelty 2).
Designed for precision over transit acronyms, equipment codes, and operational concepts.
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.chunking import Chunk

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


def tokenize_transit_text(text: str) -> List[str]:
    """
    Tokenizes text while preserving hyphenated codes, model numbers, 
    and station names (e.g., 'TM-104', 'VCB-02', 'KMRL-INC-2024-089').
    """
    # Match alphanumeric words, or hyphenated codes like TM-104, AP-ALV-03
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", text.lower())
    return [t for t in tokens if len(t) > 1]


class DenseSemanticIndex:
    """
    Dense semantic search index using normalized sublinear TF-IDF / LSA vectors
    with cosine dot-product matrix multiplication.
    """
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.chunk_matrix: Optional[np.ndarray] = None
        self.chunks: List[Chunk] = []

    def fit_and_index(self, chunks: List[Chunk]):
        self.chunks = chunks
        if not chunks:
            return

        # 1. Build vocabulary and document frequencies
        doc_freq = {}
        tokenized_docs = []
        for ch in chunks:
            text = f"{ch.doc_title} {ch.section_title} {ch.text}"
            tokens = set(tokenize_transit_text(text))
            tokenized_docs.append(tokens)
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        n_docs = len(chunks)
        # Filter terms
        self.vocabulary = {t: idx for idx, (t, df) in enumerate(sorted(doc_freq.items()))}
        vocab_size = len(self.vocabulary)
        
        # 2. Compute IDF: log((N + 1) / (df + 1)) + 1
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for t, idx in self.vocabulary.items():
            df = doc_freq.get(t, 0)
            self.idf[idx] = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0

        # 3. Build normalized document vectors
        matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for i, ch in enumerate(chunks):
            text = f"{ch.doc_title} {ch.section_title} {ch.text}"
            tokens = tokenize_transit_text(text)
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                if t in self.vocabulary:
                    idx = self.vocabulary[t]
                    # Sublinear TF scaling: 1 + log(tf)
                    tf = 1.0 + math.log(c)
                    matrix[i, idx] = tf * self.idf[idx]
            
            # L2 normalize
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm

        self.chunk_matrix = matrix

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Dense cosine similarity search."""
        if not self.chunks or self.chunk_matrix is None or len(self.vocabulary) == 0:
            return []

        tokens = tokenize_transit_text(query)
        q_vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            if t in self.vocabulary:
                idx = self.vocabulary[t]
                tf = 1.0 + math.log(c)
                q_vec[idx] = tf * self.idf[idx]

        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm
        else:
            return [(ch, 0.0) for ch in self.chunks[:top_k]]

        # Cosine dot-product
        scores = np.dot(self.chunk_matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[idx], float(scores[idx])) for idx in top_indices]


class KeywordBM25Index:
    """
    BM25 Keyword search index for exact equipment, acronym, and incident matching.
    """
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25_model: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    def fit_and_index(self, chunks: List[Chunk]):
        self.chunks = chunks
        if not chunks:
            return

        self.tokenized_corpus = []
        for ch in chunks:
            content = f"{ch.doc_title} {ch.section_title} {ch.text}"
            self.tokenized_corpus.append(tokenize_transit_text(content))

        if BM25_AVAILABLE and self.tokenized_corpus:
            self.bm25_model = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """BM25 search with raw relevance scores."""
        if not self.chunks or not self.bm25_model:
            return []

        tokens = tokenize_transit_text(query)
        if not tokens:
            return [(ch, 0.0) for ch in self.chunks[:top_k]]

        raw_scores = self.bm25_model.get_scores(tokens)
        top_indices = np.argsort(raw_scores)[::-1][:top_k]

        return [(self.chunks[idx], float(raw_scores[idx])) for idx in top_indices]


class HybridRetriever:
    """
    Proposed Novelty 2 & 3:
    Hybrid Retrieval combining Dense Semantic Search with BM25 Keyword Search
    and Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.dense_index = DenseSemanticIndex()
        self.bm25_index = KeywordBM25Index()

        self.dense_index.fit_and_index(chunks)
        self.bm25_index.fit_and_index(chunks)

    def retrieve(self, query: str, top_k: int = 5, mode: str = "hybrid", 
                 alpha: float = 0.5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Retrieve relevant sections.
        Modes:
          - 'hybrid': Combines BM25 and Dense with RRF and weighted scoring
          - 'dense_only': Semantic vector search only
          - 'bm25_only': Keyword search only
        """
        if not self.chunks:
            return []

        fetch_k = min(len(self.chunks), max(top_k * 3, 20))

        # 1. Dense retrieval
        dense_results = self.dense_index.search(query, top_k=fetch_k)
        dense_rank_map = {res[0].chunk_id: (rank + 1, res[1]) for rank, res in enumerate(dense_results)}

        # 2. BM25 retrieval
        bm25_results = self.bm25_index.search(query, top_k=fetch_k)
        bm25_rank_map = {res[0].chunk_id: (rank + 1, res[1]) for rank, res in enumerate(bm25_results)}

        if mode == "dense_only":
            candidates = []
            for ch, score in dense_results[:top_k]:
                candidates.append({
                    "chunk": ch,
                    "score": round(score, 4),
                    "dense_score": round(score, 4),
                    "bm25_score": round(bm25_rank_map.get(ch.chunk_id, (999, 0.0))[1], 4),
                    "retrieval_mode": "dense_only"
                })
            return candidates

        if mode == "bm25_only":
            candidates = []
            for ch, score in bm25_results[:top_k]:
                candidates.append({
                    "chunk": ch,
                    "score": round(score, 4),
                    "dense_score": round(dense_rank_map.get(ch.chunk_id, (999, 0.0))[1], 4),
                    "bm25_score": round(score, 4),
                    "retrieval_mode": "bm25_only"
                })
            return candidates

        # Hybrid Reciprocal Rank Fusion (RRF)
        all_chunk_ids = set(dense_rank_map.keys()).union(set(bm25_rank_map.keys()))
        chunk_lookup = {ch.chunk_id: ch for ch in self.chunks}

        scored_candidates = []
        max_dense = max([score for _, score in dense_results], default=1.0) or 1.0
        max_bm25 = max([score for _, score in bm25_results], default=1.0) or 1.0

        for cid in all_chunk_ids:
            ch = chunk_lookup.get(cid)
            if not ch:
                continue

            dense_rank, dense_raw = dense_rank_map.get(cid, (fetch_k + 1, 0.0))
            bm25_rank, bm25_raw = bm25_rank_map.get(cid, (fetch_k + 1, 0.0))

            # RRF score
            rrf_score = (1.0 / (rrf_k + dense_rank)) + (1.0 / (rrf_k + bm25_rank))

            # Normalized weighted score
            dense_norm = max(0.0, dense_raw / max_dense)
            bm25_norm = max(0.0, bm25_raw / max_bm25)
            combined_score = (alpha * dense_norm) + ((1.0 - alpha) * bm25_norm)

            # Combined hybrid ranking metric
            final_metric = (0.6 * rrf_score * 100.0) + (0.4 * combined_score)

            scored_candidates.append({
                "chunk": ch,
                "score": round(final_metric, 4),
                "dense_score": round(dense_raw, 4),
                "bm25_score": round(bm25_raw, 4),
                "rrf_score": round(rrf_score, 6),
                "dense_rank": dense_rank,
                "bm25_rank": bm25_rank,
                "retrieval_mode": "hybrid"
            })

        # Sort descending by hybrid metric
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]

