"""各节点的 Pydantic 输出 schema。purchase 输出 VehicleRecommendation，aftersales 输出 DiagnosisResult。"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Entity(BaseModel):
    text: str
    label: Literal["vehicle_model", "brand", "component", "fault", "feature", "budget"]
    start: int = 0
    end: int = 0


class VehicleRecommendation(BaseModel):
    """购车节点结构化输出。"""
    primary_pick: str = Field(description="首推车型完整名")
    alternatives: list[str] = Field(default_factory=list, description="2-3 个备选车型")
    reasons: list[str] = Field(default_factory=list, description="推荐理由")
    price_range: str = Field(default="", description="价格区间")
    next_action: str = Field(default="预约试驾", description="建议下一步")


class DiagnosisResult(BaseModel):
    """售后节点结构化输出。"""
    severity: Literal["info", "normal", "urgent", "critical"] = "normal"
    likely_causes: list[str] = Field(default_factory=list)
    immediate_actions: list[str] = Field(default_factory=list, description="用户应立刻执行")
    requires_service: bool = False
    safety_warning: str = ""


class AgentTrace(BaseModel):
    """单轮 agent 执行轨迹（用于 UI/observability）。"""
    intent: str
    backend: str
    retrieval_trace: list[str] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    cache_hit: bool = False
    elapsed_ms: int = 0
