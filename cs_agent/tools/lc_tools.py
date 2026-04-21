"""
LangChain @tool 标准工具定义。
用 convert_to_openai_tool() 自动生成 JSON schema，替代各节点手写 dict，
Pydantic 入参自动校验，降低工具调用解析出错概率。
"""
from __future__ import annotations
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool


@tool
def rag_search(query: str, top_k: int = 5) -> list[dict]:
    """在新能源汽车知识库中混合检索（BM25+向量+重排），返回最相关知识片段。"""
    from cs_agent.tools.hybrid_rag import hybrid_retrieve
    return hybrid_retrieve(query, top_k=top_k)


@tool
def ner_extract(text: str) -> list[dict]:
    """从文本中抽取车型、品牌、部件、故障、功能、预算等实体。"""
    from cs_agent.tools.ner_tool import extract_entities
    return extract_entities(text)


@tool
def submit_ticket(
    description: str,
    vehicle_model: str = "",
    components: list[str] = None,
    faults: list[str] = None,
) -> dict:
    """创建售后服务工单，返回工单号和优先级。"""
    from cs_agent.tools.ticket_tool import create_ticket
    return create_ticket(
        description=description,
        vehicle_model=vehicle_model,
        components=components or [],
        faults=faults or [],
    )


@tool
def classify_intent(intent: str, confidence: float) -> dict:
    """将用户消息分类到处理节点。
    intent 可选: vehicle_qa / aftersales / purchase / charging /
    order_tracking / complaint / account / insurance / test_drive /
    navigation / roadside / chitchat
    """
    return {"intent": intent, "confidence": confidence}


# 自动生成 OpenAI 格式 schema（各节点直接使用）
RAG_TOOL_SCHEMA = convert_to_openai_tool(rag_search)
NER_TOOL_SCHEMA = convert_to_openai_tool(ner_extract)
TICKET_TOOL_SCHEMA = convert_to_openai_tool(submit_ticket)
INTENT_TOOL_SCHEMA = convert_to_openai_tool(classify_intent)

# 修正 enum 约束（convert_to_openai_tool 不支持 Literal，手动补充）
INTENT_TOOL_SCHEMA["function"]["parameters"]["properties"]["intent"]["enum"] = [
    "vehicle_qa", "aftersales", "purchase", "charging",
    "order_tracking", "complaint", "account", "insurance",
    "test_drive", "navigation", "roadside", "chitchat",
]
INTENT_TOOL_SCHEMA["function"]["description"] = (
    "将用户消息分类到最合适的处理节点。"
    "vehicle_qa=车型参数/功能咨询, aftersales=故障/维修/保养, "
    "purchase=购车/比价/金融/置换, charging=充电/充电站/续航焦虑, "
    "order_tracking=订单/交付/进度查询, complaint=投诉/负面反馈/赔偿, "
    "account=账户/APP/会员/积分, insurance=保险/理赔/上牌, "
    "test_drive=试驾预约/体验中心, navigation=导航/地图/OTA升级, "
    "roadside=道路救援/紧急求助/拖车, chitchat=闲聊/问候"
)
