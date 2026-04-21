"""充电节点。处理充电桩查询、充电故障、续航焦虑等问题。"""
from __future__ import annotations
from cs_agent.state import CSState
from cs_agent.tools.hybrid_rag import hybrid_retrieve
from cs_agent.llm_client import llm_generate
from cs_agent.observability import Timer

_SYSTEM = (
    "你是新能源汽车充电服务专家。解答充电相关问题，包括：\n"
    "充电桩查找、充电价格、充电速度、充电故障排查、续航规划等。\n"
    "回答简洁实用，200字以内，优先给出可操作的建议。"
)


def charging_node(state: CSState) -> dict:
    query = state["query"]
    mem_ctx = state.get("memory_context", "")

    with Timer() as t:
        chunks = hybrid_retrieve(query, top_k=4, use_reranker=True)

    context = "\n".join(f"- {c['content'][:200]}" for c in chunks[:3]) if chunks else ""
    prompt = f"用户问题：{query}"
    if mem_ctx:
        prompt = f"用户背景：{mem_ctx}\n\n{prompt}"
    if context:
        prompt += f"\n\n参考知识：\n{context}"
    prompt += "\n\n请给出充电方面的专业解答："

    answer = llm_generate(prompt, system=_SYSTEM, max_tokens=300)

    return {
        "answer": answer,
        "retrieved_chunks": chunks,
        "retrieval_trace": [f"[charging] retrieved {len(chunks)} chunks in {t.elapsed_ms}ms"],
        "sources": [c["source"] for c in chunks],
        "elapsed_ms": t.elapsed_ms,
    }
