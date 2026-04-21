"""
售后节点。检索故障知识库，NER 抽取部件/故障实体，输出结构化诊断结果。
含安全关键词时自动提升 severity，触发 safety_check 节点暂停等待人工确认。
"""
from __future__ import annotations
import json
from cs_agent.state import CSState
from cs_agent.tools.ner_tool import extract_entities
from cs_agent.tools.hybrid_rag import hybrid_retrieve
from cs_agent.tools.ticket_tool import create_ticket
from cs_agent.llm_client import llm_generate
from cs_agent.schemas import DiagnosisResult
from cs_agent.observability import Timer, trace_record
from cs_agent.tools.lc_tools import TICKET_TOOL_SCHEMA

_SYSTEM = (
    "你是新能源汽车售后服务专家。请先安抚用户情绪，然后给出专业的初步诊断建议。\n"
    "如涉及安全问题（刹车/起火/冒烟/失灵），务必提醒立即停车并联系救援。\n"
    "回答结构化，分步骤说明，控制在 400 字以内。"
)

_SAFETY_KEYWORDS = {"刹车", "起火", "冒烟", "失灵", "漏气", "爆炸", "救援"}

_DIAGNOSIS_TOOL = [TICKET_TOOL_SCHEMA]


def aftersales_node(state: CSState) -> dict:
    query = state["query"]
    entities = state.get("entities", []) or extract_entities(query)

    components = [e["text"] for e in entities if e["label"] == "component"]
    faults = [e["text"] for e in entities if e["label"] == "fault"]
    vehicles = [e["text"] for e in entities if e["label"] == "vehicle_model"]

    # 安全词拦截 → 立即提升优先级
    is_safety = any(kw in query for kw in _SAFETY_KEYWORDS)

    with Timer() as t:
        search_q = query + (" " + " ".join(components) if components else "")
        chunks = hybrid_retrieve(search_q, top_k=4, use_reranker=True)

    ticket = None
    ticket_id = ""
    if components or faults or is_safety:
        ticket = create_ticket(
            description=query,
            components=components,
            faults=faults,
            vehicle_model=vehicles[0] if vehicles else "",
        )
        ticket_id = ticket["ticket_id"]

    context = "\n".join(f"- {c['content'][:200]}" for c in chunks[:3]) if chunks else "暂无匹配知识库"
    entity_info = ""
    if components:
        entity_info += f"\n识别部件：{', '.join(components)}"
    if faults:
        entity_info += f"\n识别故障：{', '.join(faults)}"
    if vehicles:
        entity_info += f"\n识别车型：{', '.join(vehicles)}"
    if is_safety:
        entity_info += "\n⚠️ 涉及安全风险，需优先处理"

    prompt = (
        f"用户反馈：{query}{entity_info}\n\n"
        f"参考知识：\n{context}\n\n"
        "请给出专业的诊断建议和处理方案："
    )
    answer = llm_generate(prompt, system=_SYSTEM, max_tokens=512)

    ticket_note = (
        f"\n\n📋 已自动生成服务工单（编号：{ticket_id}，优先级：{ticket['priority']}），"
        "售后团队24小时内跟进。"
    ) if ticket else ""
    answer += ticket_note

    structured = DiagnosisResult(
        severity="critical" if is_safety else ("urgent" if faults else "normal"),
        likely_causes=faults,
        immediate_actions=["立即停车并开启双闪"] if is_safety else [],
        requires_service=bool(components or faults),
        safety_warning="涉及安全隐患，请立即处理！" if is_safety else "",
    ).model_dump()

    trace_record({
        "intent": "aftersales",
        "query": query,
        "ticket_id": ticket_id,
        "severity": structured["severity"],
        "elapsed_ms": t.elapsed_ms,
    })

    return {
        "answer": answer,
        "retrieved_chunks": chunks,
        "sources": [c["source"] for c in chunks],
        "ticket_id": ticket_id,
        "entities": entities,
        "structured": structured,
        "elapsed_ms": t.elapsed_ms,
    }
