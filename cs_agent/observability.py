"""
语义缓存 + 调用 trace。
缓存用 bge embedding 做余弦匹配，阈值 0.92，命中直接返回历史答案跳过检索。
trace 是个 100 条的 ring buffer，UI 侧边栏实时展示用。
"""
from __future__ import annotations
import time
import threading
from collections import deque
from functools import lru_cache

import numpy as np

_LOCK = threading.Lock()
_CACHE_LIMIT = 200
_THRESHOLD = 0.92

_cache_embeddings: list[np.ndarray] = []
_cache_keys: list[str] = []
_cache_values: list[dict] = []

_TRACES: deque = deque(maxlen=100)


@lru_cache(maxsize=1)
def _embedder():
    from cs_agent.tools.rag_tool import _get_embedder
    return _get_embedder()


def cache_lookup(query: str) -> dict | None:
    if not _cache_embeddings:
        return None
    try:
        q = _embedder().encode([query], normalize_embeddings=True).astype(np.float32)
    except Exception:
        return None
    with _LOCK:
        if not _cache_embeddings:
            return None
        mat = np.vstack(_cache_embeddings)
        scores = mat @ q[0]
        idx = int(np.argmax(scores))
        if float(scores[idx]) >= _THRESHOLD:
            v = dict(_cache_values[idx])
            v["_cache_hit"] = True
            v["_cache_score"] = float(scores[idx])
            return v
    return None


def cache_put(query: str, value: dict) -> None:
    try:
        q = _embedder().encode([query], normalize_embeddings=True).astype(np.float32)[0]
    except Exception:
        return
    with _LOCK:
        if len(_cache_embeddings) >= _CACHE_LIMIT:
            _cache_embeddings.pop(0)
            _cache_keys.pop(0)
            _cache_values.pop(0)
        _cache_embeddings.append(q)
        _cache_keys.append(query)
        _cache_values.append(value)


def cache_clear() -> None:
    with _LOCK:
        _cache_embeddings.clear()
        _cache_keys.clear()
        _cache_values.clear()


# ---------- tracer ----------
def trace_record(record: dict) -> None:
    record.setdefault("ts", time.time())
    _TRACES.append(record)


def get_recent_traces(n: int = 20) -> list[dict]:
    return list(_TRACES)[-n:]


class Timer:
    def __init__(self):
        self.start = 0.0
        self.elapsed_ms = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = int((time.perf_counter() - self.start) * 1000)
