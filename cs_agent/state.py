"""LangGraph 对话状态。total=False 让所有字段可选，节点只需返回自己修改的那部分。"""
from __future__ import annotations
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class CSState(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # 完整对话历史
    query: str                                 # 当前用户输入
    intent: str                                # router 识别的意图
    intent_confidence: float                   # 意图置信度
    entities: list[dict]                       # 抽取到的实体列表
    retrieved_chunks: list[dict]               # RAG 检索结果
    retrieval_trace: list[str]                 # Self-RAG 检索轨迹
    ticket_id: str                             # 售后工单号（仅 aftersales）
    answer: str                                # 最终答案
    structured: dict                           # 结构化产物（Pydantic dump）
    sources: list[str]                         # 答案溯源
    session_id: str                            # 会话 ID
    step_count: int                            # ReAct 步数计数
    memory_context: str                        # 长期记忆注入文本
    backend: str                               # 当前 LLM 后端
    cache_hit: bool                            # 是否命中语义缓存
    elapsed_ms: int                            # 总耗时
    _safety_approved: bool                     # Human-in-the-Loop 安全确认标志
