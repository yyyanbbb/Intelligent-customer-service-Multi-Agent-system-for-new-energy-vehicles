"""投诉节点。真诚道歉、上报处理、自动建工单，高优先级。"""
from __future__ import annotations
from cs_agent.state import CSState
from cs_agent.tools.ticket_tool import create_ticket
from cs_agent.llm_client import llm_generate
from cs_agent.observability import Timer, trace_record

_SYSTEM = (
    "你是新能源汽车售后客服投诉专员。处理用户投诉时：\n"
    "1. 首先真诚道歉，表示重视\n"
    "2. 明确说明会上报并跟进处理\n"
    "3. 告知用户投诉编号和预期处理时限（工作日内回复）\n"
    "4. 提供官方投诉渠道（400热线）\n"
    "回答真诚、专业，200字以内，不要推诿责任。"
)


def complaint_node(state: CSState) -> dict:
    query = state["query"]

    with Timer() as t:
        ticket = create_ticket(
            description=query,
            components=[],
            faults=[],
            vehicle_model="",
        )
        ticket_id = ticket["ticket_id"]

        prompt = (
            f"用户投诉内容：{query}\n\n"
            f"已生成投诉工单编号：{ticket_id}\n\n"
            "请给出专业、真诚的投诉受理回复："
        )
        answer = llm_generate(prompt, system=_SYSTEM, max_tokens=300)

    if ticket_id not in answer:
        answer += f"\n\n📋 投诉工单已建立（编号：{ticket_id}），专属客服将在1个工作日内联系您。"

    trace_record({
        "intent": "complaint",
        "query": query,
        "ticket_id": ticket_id,
        "elapsed_ms": t.elapsed_ms,
    })

    return {
        "answer": answer,
        "ticket_id": ticket_id,
        "retrieved_chunks": [],
        "sources": [],
        "structured": {"severity": "urgent", "ticket_id": ticket_id},
        "elapsed_ms": t.elapsed_ms,
    }
