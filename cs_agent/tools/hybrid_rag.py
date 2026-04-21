"""
混合检索：BM25 + FAISS dense（bge-small-zh）+ BGE cross-encoder reranker，RRF 融合。
self_rag_retrieve() 在此基础上加了反思层：reranker top-1 分低于 0.5 时重写查询再检一次。
reranker 拉不到就降级到 RRF 融合分；dense 挂了就退到纯 BM25。
"""
from __future__ import annotations

import math
import logging
from functools import lru_cache
from typing import Optional

import jieba

from cs_agent.tools.rag_tool import _ensure_index, _get_embedder  # reuse FAISS/docs

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
#  BM25 over the same doc set
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def _build_bm25():
    from rank_bm25 import BM25Okapi
    _, docs = _ensure_index()
    tokenized = [list(jieba.cut(d["content"])) for d in docs]
    return BM25Okapi(tokenized), docs


def bm25_search(query: str, top_k: int = 10) -> list[tuple[int, float]]:
    bm25, docs = _build_bm25()
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


# ------------------------------------------------------------
#  Dense (reuse existing FAISS)
# ------------------------------------------------------------
def dense_search(query: str, top_k: int = 10) -> list[tuple[int, float]]:
    import numpy as np
    index, _ = _ensure_index()
    emb = _get_embedder().encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(emb, top_k)
    return [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]


# ------------------------------------------------------------
#  Reciprocal Rank Fusion
# ------------------------------------------------------------
def rrf_fuse(ranked_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (doc_id, _) in enumerate(lst):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ------------------------------------------------------------
#  Cross-encoder reranker
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("BAAI/bge-reranker-base", device="cpu", max_length=512)
    except Exception as e:
        logger.warning(f"Reranker 加载失败，将跳过重排: {e}")
        return None


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    reranker = _get_reranker()
    if reranker is None or not candidates:
        return candidates[:top_k]
    pairs = [(query, c["content"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]


# ------------------------------------------------------------
#  Unified hybrid retrieval
# ------------------------------------------------------------
def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    prefetch: int = 15,
    use_reranker: bool = True,
) -> list[dict]:
    """
    混合检索入口：BM25 + Dense → RRF → 可选 reranker。
    """
    _, docs = _ensure_index()

    dense_hits, bm25_hits = [], []
    try:
        dense_hits = dense_search(query, top_k=prefetch)
    except Exception as e:
        logger.warning(f"Dense 检索失败: {e}")
    try:
        bm25_hits = bm25_search(query, top_k=prefetch)
    except Exception as e:
        logger.warning(f"BM25 检索失败: {e}")

    fused = rrf_fuse([dense_hits, bm25_hits])[:prefetch]

    candidates = []
    for idx, score in fused:
        d = docs[idx]
        candidates.append({
            "content": d["content"],
            "source": d["source"],
            "type": d.get("type", ""),
            "score": score,
        })

    if use_reranker:
        candidates = rerank(query, candidates, top_k=top_k)
    else:
        candidates = candidates[:top_k]

    return candidates


# ------------------------------------------------------------
#  Self-RAG reflection: judge sufficiency, rewrite if poor
# ------------------------------------------------------------
def self_rag_retrieve(
    query: str,
    top_k: int = 5,
    max_iter: int = 2,
    quality_threshold: float = 0.5,
) -> tuple[list[dict], list[str]]:
    """
    Self-RAG 轻量版：
    - 初次检索后按 top-1 rerank_score 判分
    - 不足则让 LLM 重写查询再检索
    - 返回最终 chunks + 检索轨迹 trace

    返回 (chunks, trace)。
    """
    from cs_agent.llm_client import llm_generate
    trace = [f"[query] {query}"]
    best: list[dict] = []
    current = query

    for step in range(max_iter):
        hits = hybrid_retrieve(current, top_k=top_k)
        trace.append(f"[iter {step}] {len(hits)} hits, top_score={hits[0].get('rerank_score', hits[0].get('score', 0)):.3f}" if hits else f"[iter {step}] 0 hits")
        if hits and (hits[0].get("rerank_score", hits[0].get("score", 0)) >= quality_threshold):
            return hits, trace
        if not best or (hits and len(hits) > len(best)):
            best = hits
        if step < max_iter - 1:
            rewrite_prompt = (
                f"原始问题：{query}\n"
                "请将其重写为更便于向量检索的中文短句（提取关键实体+意图，去除口语），"
                "直接输出重写后的查询，不要解释。"
            )
            rewritten = llm_generate(rewrite_prompt, max_tokens=64)
            if isinstance(rewritten, str) and rewritten.strip():
                current = rewritten.strip().split("\n")[0][:100]
                trace.append(f"[rewrite] {current}")
    return best, trace
