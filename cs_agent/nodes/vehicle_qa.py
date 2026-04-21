"""
车型问答节点。Hybrid RAG + Self-RAG 检索，检索质量不足时自动重写查询。
命中的结果写入语义缓存，下次相似问题直接短路。
"""
from __future__ import annotations
from cs_agent.state import CSState
from cs_agent.tools.hybrid_rag import self_rag_retrieve
from cs_agent.llm_client import llm_generate
from cs_agent.observability import cache_put, Timer, trace_record

_SYSTEM_TMPL = (
    "你是新能源汽车专业客服顾问，熟悉比亚迪、小鹏、理想、蔚来、问界、特斯拉、"
    "小米汽车、极氪、零跑、深蓝、阿维塔、岚图、仰望、腾势、智己等主流品牌。\n"
    "请基于提供的参考信息，用简洁专业的语言回答问题。如信息不足，请如实说明。\n"
    "回答控制在 350 字以内，重点突出。{mem}"
)


def vehicle_qa_node(state: CSState) -> dict:
    query = state["query"]
    mem_ctx = state.get("memory_context", "")
    system = _SYSTEM_TMPL.format(mem=f"\n用户背景：{mem_ctx}" if mem_ctx else "")

    with Timer() as t:
        chunks, trace = self_rag_retrieve(query, top_k=5, max_iter=2)

    if not chunks:
        answer = "抱歉，暂时没有找到相关信息。建议联系对应品牌官方客服或到店咨询。"
        return {"answer": answer, "retrieved_chunks": [], "sources": [], "retrieval_trace": trace}

    context = "\n\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks[:4]))
    sources = list({c["source"] for c in chunks})

    prompt = (
        f"参考信息：\n{context}\n\n"
        f"用户问题：{state['query']}\n\n"
        "请基于参考信息给出准确回答："
    )
    answer = llm_generate(prompt, system=system, max_tokens=512)

    result = {
        "answer": answer,
        "retrieved_chunks": chunks,
        "sources": sources,
        "retrieval_trace": trace,
        "elapsed_ms": t.elapsed_ms,
    }

    # 写入语义缓存
    cache_put(query, {**result, "intent": "vehicle_qa", "entities": state.get("entities", [])})

    trace_record({
        "intent": "vehicle_qa",
        "query": query,
        "sources": sources,
        "hits": len(chunks),
        "elapsed_ms": t.elapsed_ms,
    })

    return result
