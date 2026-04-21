"""
RAG 检索工具：FAISS 向量库 + bge-small-zh embedding。
首次调用时自动构建索引，后续从磁盘加载。
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from functools import lru_cache

import numpy as np

_KB_DIR = Path(__file__).parent.parent / "knowledge"
_INDEX_DIR = Path(__file__).parent.parent / "knowledge" / "faiss_index"


def _build_docs() -> list[dict]:
    docs = []
    vehicles = json.loads((_KB_DIR / "vehicles.json").read_text(encoding="utf-8"))
    for i, v in enumerate(vehicles):
        s = v.get("specs", {})
        parts = [f"车型：{v.get('model', v.get('brand', ''))}，品牌：{v.get('brand', '')}"]
        if v.get("price"):
            parts.append(f"价格：{v['price']}")
        if s.get("CLTC续航km"):
            parts.append(f"CLTC续航：{s['CLTC续航km']}km")
        if s.get("电池容量kWh"):
            parts.append(f"电池容量：{s['电池容量kWh']}kWh")
        if s.get("百公里加速s"):
            parts.append(f"0-100km/h加速：{s['百公里加速s']}秒")
        if s.get("轴距mm"):
            parts.append(f"轴距：{s['轴距mm']}mm")
        if s.get("最大功率kW"):
            parts.append(f"最大功率：{s['最大功率kW']}kW")
        if s.get("最大扭矩Nm"):
            parts.append(f"最大扭矩：{s['最大扭矩Nm']}N·m")
        if s.get("最高车速kmh"):
            parts.append(f"最高车速：{s['最高车速kmh']}km/h")
        if s.get("快充时间min"):
            parts.append(f"快充时间：{s['快充时间min']}分钟")
        if v.get("summary"):
            # strip navigation noise — keep only first meaningful paragraph
            raw = v["summary"]
            clean = re.sub(r'\s+', ' ', raw[:500]).strip()
            # cut at sidebar noise signals
            for noise in ["智能助手", "下载App", "首页\n", "选车\n", "排行榜"]:
                idx = clean.find(noise)
                if idx > 0:
                    clean = clean[:idx].strip()
            if len(clean) > 50:
                parts.append(clean[:300])
        text = "，".join(parts)
        docs.append({"content": text, "source": f"vehicle_{i}", "type": "vehicle"})

    faqs = json.loads((_KB_DIR / "faq.json").read_text(encoding="utf-8"))
    for f in faqs:
        docs.append({
            "content": f"问：{f['question']}\n答：{f['answer']}",
            "source": f"faq_{f['id']}",
            "type": "faq",
            "category": f.get("category", ""),
        })
    return docs


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    # Force CPU: RTX 5070Ti (sm_120/Blackwell) not yet supported by PyTorch 2.6
    return SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")


def _ensure_index():
    import faiss

    idx_file = _INDEX_DIR / "index.faiss"
    docs_file = _INDEX_DIR / "docs.pkl"

    if idx_file.exists() and docs_file.exists():
        index = faiss.read_index(str(idx_file))
        docs = pickle.loads(docs_file.read_bytes())
        return index, docs

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    docs = _build_docs()
    embedder = _get_embedder()
    texts = [d["content"] for d in docs]
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    faiss.write_index(index, str(idx_file))
    docs_file.write_bytes(pickle.dumps(docs))
    return index, docs


def rag_retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3) -> list[dict]:
    """
    检索与 query 最相关的知识库片段。
    返回 list[dict]，每个元素包含 content / source / score。
    """
    import faiss

    index, docs = _ensure_index()
    embedder = _get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or float(score) < score_threshold:
            continue
        results.append({
            "content": docs[idx]["content"],
            "source": docs[idx]["source"],
            "score": float(score),
        })
    return results


def rebuild_index() -> None:
    """强制重建 FAISS 索引（知识库更新后调用）。"""
    import faiss
    import shutil
    if _INDEX_DIR.exists():
        shutil.rmtree(_INDEX_DIR)
    _ensure_index.cache_clear()
    _get_embedder.cache_clear()
    _ensure_index()
