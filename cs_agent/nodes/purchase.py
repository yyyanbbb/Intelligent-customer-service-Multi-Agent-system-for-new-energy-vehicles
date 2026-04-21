"""
购车咨询节点。从记忆中读取预算/偏好注入查询，检索知识库后优先通过
tool-calling 返回结构化推荐，降级时直接输出文本答案。
"""
from __future__ import annotations
import json
from cs_agent.state import CSState
from cs_agent.tools.hybrid_rag import hybrid_retrieve
from cs_agent.llm_client import llm_chat, llm_generate
from cs_agent.schemas import VehicleRecommendation
from cs_agent.observability import Timer, trace_record, cache_put
from cs_agent.tools.lc_tools import RAG_TOOL_SCHEMA

_SYSTEM_TMPL = (
    "你是新能源汽车资深销售顾问，熟悉市场上主流新能源品牌和车型。\n"
    "请根据用户需求推荐合适的车型，客观分析优劣，不过度推销。{mem}\n"
    "如用户在对比车型，请列出关键差异点。回答控制在 400 字以内。"
)

_RECOMMEND_TOOL = [RAG_TOOL_SCHEMA]


def purchase_node(state: CSState) -> dict:
    query = state["query"]
    entities = state.get("entities", [])
    mem_ctx = state.get("memory_context", "")

    models = [e["text"] for e in entities if e["label"] == "vehicle_model"]
    budget = [e["text"] for e in entities if e["label"] == "budget"]

    system = _SYSTEM_TMPL.format(mem=f"\n用户背景：{mem_ctx}" if mem_ctx else "")

    # 构建增强查询
    search_q = query
    if models:
        search_q += " " + " ".join(models) + " 参数配置价格"
    if budget:
        search_q += f" {budget[0]} 推荐"

    with Timer() as t:
        chunks = hybrid_retrieve(search_q, top_k=6, use_reranker=True)

    context = "\n\n".join(f"[{i+1}] {c['content'][:300]}" for i, c in enumerate(chunks[:5]))

    extra = ""
    if models:
        extra += f"\n用户关注车型：{', '.join(models)}"
    if budget:
        extra += f"\n用户预算：{', '.join(budget)}"

    # 尝试 tool-calling 结构化输出
    structured: dict = {}
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"参考信息：\n{context}\n\n"
            f"用户咨询：{query}{extra}\n\n"
            "请调用 recommend_vehicles 给出结构化推荐，并同时给出自然语言回答："
        )},
    ]
    result = llm_chat(messages, max_tokens=700, temperature=0.4, tools=_RECOMMEND_TOOL)

    if isinstance(result, dict) and result.get("tool_calls"):
        tc = result["tool_calls"][0]
        args = tc.get("function", {}).get("arguments", tc.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        try:
            rec = VehicleRecommendation(**args)
            structured = rec.model_dump()
        except Exception:
            pass
        # 再调一次拿自然语言答案
        answer_prompt = (
            f"参考信息：\n{context}\n\n"
            f"用户咨询：{query}{extra}\n\n"
            "请给出专业的购车建议："
        )
        answer = llm_generate(answer_prompt, system=system, max_tokens=600)
    else:
        answer = result if isinstance(result, str) else llm_generate(
            f"参考信息：\n{context}\n\n用户咨询：{query}{extra}\n\n请给出购车建议：",
            system=system, max_tokens=600,
        )

    result_dict = {
        "answer": answer,
        "retrieved_chunks": chunks,
        "sources": [c["source"] for c in chunks],
        "structured": structured,
        "elapsed_ms": t.elapsed_ms,
    }

    cache_put(query, {**result_dict, "intent": "purchase", "entities": entities})

    trace_record({
        "intent": "purchase",
        "query": query,
        "primary_pick": structured.get("primary_pick", ""),
        "elapsed_ms": t.elapsed_ms,
    })

    return result_dict
