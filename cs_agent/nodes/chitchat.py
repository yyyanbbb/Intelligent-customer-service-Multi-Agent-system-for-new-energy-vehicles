"""闲聊/兜底节点。注入用户背景记忆，友好回应并自然引导到业务场景。"""
from __future__ import annotations
from cs_agent.state import CSState
from cs_agent.llm_client import llm_generate

_SYSTEM_TMPL = (
    "你是新能源汽车智能客服助手，熟悉市面上主流新能源品牌和车型。"
    "对闲聊类问题请友好简短回应，并自然引导用户了解新能源汽车产品和服务。"
    "回答控制在 150 字以内，亲切自然。{mem}"
)


def chitchat_node(state: CSState) -> dict:
    mem_ctx = state.get("memory_context", "")
    system = _SYSTEM_TMPL.format(mem=f"\n用户背景：{mem_ctx}" if mem_ctx else "")
    answer = llm_generate(state["query"], system=system, max_tokens=256)
    return {"answer": answer, "retrieved_chunks": [], "sources": [], "retrieval_trace": [], "elapsed_ms": 0}
