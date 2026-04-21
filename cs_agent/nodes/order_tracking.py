"""订单节点。处理交付进度、提车流程、尾款支付等问题。"""
from __future__ import annotations
from cs_agent.state import CSState
from cs_agent.tools.hybrid_rag import hybrid_retrieve
from cs_agent.llm_client import llm_generate
from cs_agent.observability import Timer

_SYSTEM = (
    "你是新能源汽车订单服务专员。解答订单相关问题，包括：\n"
    "订单状态查询、交付时间预估、提车流程、尾款支付等。\n"
    "如需查询具体订单信息，提示用户联系交付中心或在APP中查看。\n"
    "回答简洁，150字以内。"
)


def order_tracking_node(state: CSState) -> dict:
    query = state["query"]
    mem_ctx = state.get("memory_context", "")

    with Timer() as t:
        chunks = hybrid_retrieve(query + " 交付 订单 提车", top_k=3, use_reranker=False)

    context = "\n".join(f"- {c['content'][:150]}" for c in chunks[:2]) if chunks else ""
    prompt = f"用户问题：{query}"
    if mem_ctx:
        prompt = f"用户背景：{mem_ctx}\n\n{prompt}"
    if context:
        prompt += f"\n\n参考信息：\n{context}"
    prompt += "\n\n请解答订单/交付相关问题："

    answer = llm_generate(prompt, system=_SYSTEM, max_tokens=250)

    return {
        "answer": answer,
        "retrieved_chunks": chunks,
        "retrieval_trace": [f"[order_tracking] retrieved {len(chunks)} chunks in {t.elapsed_ms}ms"],
        "sources": [c["source"] for c in chunks],
        "elapsed_ms": t.elapsed_ms,
    }
