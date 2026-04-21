"""道路救援节点。最高优先级，立即创建紧急工单并给出救援指引。"""
from __future__ import annotations
from cs_agent.state import CSState
from cs_agent.tools.ticket_tool import create_ticket
from cs_agent.llm_client import llm_generate
from cs_agent.observability import Timer, trace_record

_SYSTEM = (
    "你是新能源汽车紧急道路救援专员。处理救援请求时：\n"
    "1. 立即确认用户安全，提示开启双闪并在安全区域等待\n"
    "2. 告知救援热线 400-xxx-xxxx（24小时）\n"
    "3. 提示在APP '我的→道路救援' 一键呼叫，可定位\n"
    "4. 预估救援到达时间（城区30分钟，高速60分钟）\n"
    "回答简短紧凑，优先保障用户安全。"
)


def roadside_node(state: CSState) -> dict:
    query = state["query"]

    with Timer() as t:
        ticket = create_ticket(
            description=f"[紧急救援] {query}",
            components=[],
            faults=["道路救援"],
            vehicle_model="",
        )
        ticket_id = ticket["ticket_id"]

        prompt = (
            f"用户救援请求：{query}\n\n"
            f"救援工单编号：{ticket_id}\n\n"
            "请立即给出救援指引："
        )
        answer = llm_generate(prompt, system=_SYSTEM, max_tokens=250)

    if ticket_id not in answer:
        answer += f"\n\n🚨 紧急救援工单已创建（编号：{ticket_id}），救援人员正在派遣中。"

    trace_record({
        "intent": "roadside",
        "query": query,
        "ticket_id": ticket_id,
        "severity": "critical",
        "elapsed_ms": t.elapsed_ms,
    })

    return {
        "answer": answer,
        "ticket_id": ticket_id,
        "retrieved_chunks": [],
        "sources": [],
        "structured": {"severity": "critical", "ticket_id": ticket_id},
        "elapsed_ms": t.elapsed_ms,
    }
