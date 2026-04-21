"""
LangGraph 主图。包含 9 个业务节点 + 安全检查 + 缓存短路逻辑。
SqliteSaver 按 thread_id 持久化图状态，safety_check 用 interrupt_before 暂停等待人工确认。
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage

from cs_agent.state import CSState
from cs_agent.nodes import (
    router_node, vehicle_qa_node, aftersales_node,
    purchase_node, chitchat_node,
    charging_node, order_tracking_node, complaint_node, roadside_node,
)
from cs_agent.checkpointer import get_checkpointer


_DEDICATED_NODES = {
    "vehicle_qa", "aftersales", "purchase", "charging",
    "order_tracking", "complaint", "roadside",
}


def _route(state: CSState) -> str:
    if state.get("cache_hit"):
        return "__cache__"
    intent = state.get("intent", "chitchat")
    return intent if intent in _DEDICATED_NODES else "chitchat"


def _route_aftersales(state: CSState) -> str:
    """售后有严重/紧急故障且未经人工确认时，转 safety_check 暂停。"""
    structured = state.get("structured", {})
    if structured.get("severity") in ("critical", "urgent") and not state.get("_safety_approved"):
        return "safety_check"
    return "add_ai_msg"


def _add_human_message(state: CSState) -> dict:
    return {"messages": [HumanMessage(content=state.get("query", ""))]}


def _add_ai_message(state: CSState) -> dict:
    return {"messages": [AIMessage(content=state.get("answer", ""))]}


def _passthrough(state: CSState) -> dict:
    return {}


def _safety_check_node(state: CSState) -> dict:
    """
    暂停占位节点，interrupt_before 会在这里挂起图执行。
    前端确认后用 graph.update_state({'_safety_approved': True}) + graph.invoke(None) 恢复。
    """
    return {"_safety_approved": True}


def build_graph() -> CompiledStateGraph:
    g = StateGraph(CSState)

    g.add_node("add_human_msg", _add_human_message)
    g.add_node("router", router_node)
    g.add_node("vehicle_qa", vehicle_qa_node)
    g.add_node("aftersales", aftersales_node)
    g.add_node("purchase", purchase_node)
    g.add_node("chitchat", chitchat_node)
    g.add_node("charging", charging_node)
    g.add_node("order_tracking", order_tracking_node)
    g.add_node("complaint", complaint_node)
    g.add_node("roadside", roadside_node)
    g.add_node("safety_check", _safety_check_node)
    g.add_node("cache_passthrough", _passthrough)
    g.add_node("add_ai_msg", _add_ai_message)

    g.set_entry_point("add_human_msg")
    g.add_edge("add_human_msg", "router")
    g.add_conditional_edges(
        "router",
        _route,
        {
            "vehicle_qa": "vehicle_qa",
            "aftersales": "aftersales",
            "purchase": "purchase",
            "chitchat": "chitchat",
            "charging": "charging",
            "order_tracking": "order_tracking",
            "complaint": "complaint",
            "roadside": "roadside",
            "__cache__": "cache_passthrough",
        },
    )

    # 售后节点：严重故障先过 safety_check
    g.add_conditional_edges("aftersales", _route_aftersales, {
        "safety_check": "safety_check",
        "add_ai_msg": "add_ai_msg",
    })
    g.add_edge("safety_check", "add_ai_msg")

    for node in ("vehicle_qa", "purchase", "chitchat", "cache_passthrough",
                 "charging", "order_tracking", "complaint", "roadside"):
        g.add_edge(node, "add_ai_msg")
    g.add_edge("add_ai_msg", END)

    checkpointer = get_checkpointer()
    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["safety_check"],  # 触发 Human-in-the-Loop
    )


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def get_graph_mermaid() -> str:
    """返回图的 Mermaid 文本，主要用于调试。"""
    try:
        return get_graph().get_graph().draw_mermaid()
    except Exception as e:
        return f"```\n# 图可视化失败: {e}\n```"


def chat(query: str, session_id: str = "default", history: list | None = None, backend: str | None = None) -> dict:
    """
    同步对话接口，session_id 同时作为 checkpointing 的 thread_id。
    如果图被 interrupt_before 暂停，返回的 _pending_interrupt=True，
    调用方用 resume_after_safety() 处理后续确认。
    """
    if backend:
        import os
        os.environ["LLM_BACKEND"] = backend

    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    init_state: CSState = {
        "messages": history or [],
        "query": query,
        "intent": "",
        "intent_confidence": 0.0,
        "entities": [],
        "retrieved_chunks": [],
        "retrieval_trace": [],
        "ticket_id": "",
        "answer": "",
        "structured": {},
        "sources": [],
        "session_id": session_id,
        "step_count": 0,
        "memory_context": "",
        "backend": "",
        "cache_hit": False,
        "elapsed_ms": 0,
    }

    result = graph.invoke(init_state, config=config)  # type: ignore[arg-type]

    # 检查是否被 interrupt_before 暂停
    state_snapshot = graph.get_state(config)  # type: ignore[arg-type]
    is_interrupted = bool(state_snapshot.next)

    return {
        "answer": result.get("answer", ""),
        "intent": result.get("intent", ""),
        "intent_confidence": result.get("intent_confidence", 0.0),
        "entities": result.get("entities", []),
        "sources": result.get("sources", []),
        "ticket_id": result.get("ticket_id", ""),
        "structured": result.get("structured", {}),
        "messages": result.get("messages", []),
        "cache_hit": result.get("cache_hit", False),
        "backend": result.get("backend", ""),
        "retrieval_trace": result.get("retrieval_trace", []),
        "elapsed_ms": result.get("elapsed_ms", 0),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "_pending_interrupt": is_interrupted,
        "_session_id": session_id,
    }


def resume_after_safety(session_id: str, approved: bool = True) -> dict:
    """人工确认后继续执行图。approved=False 时取消工单并退出。"""
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    if approved:
        graph.update_state(config, {"_safety_approved": True}, as_node="safety_check")  # type: ignore[arg-type]
        result = graph.invoke(None, config=config)  # type: ignore[arg-type]
    else:
        graph.update_state(config, {"ticket_id": "", "_safety_approved": False}, as_node="safety_check")  # type: ignore[arg-type]
        result = graph.invoke(None, config=config)  # type: ignore[arg-type]
    return {
        "answer": result.get("answer", ""),
        "ticket_id": result.get("ticket_id", ""),
        "_pending_interrupt": False,
    }
