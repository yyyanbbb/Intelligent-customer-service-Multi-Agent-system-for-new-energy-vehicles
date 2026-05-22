from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


TaskType = Literal["purchase", "aftersales", "charging", "insurance", "complaint", "faq", "chitchat"]
TaskStatus = Literal[
    "created",
    "running",
    "awaiting_user_input",
    "awaiting_confirmation",
    "completed",
    "failed",
]
StepKind = Literal["ask_user", "call_tool", "confirm", "summarize"]
AgentName = Literal[
    "supervisor",
    "planner_agent",
    "purchase_agent",
    "aftersales_agent",
    "charging_agent",
    "insurance_agent",
    "complaint_agent",
    "legacy_agent",
]


def new_task_id() -> str:
    return f"task-{uuid4().hex[:12]}"


class PlanStep(BaseModel):
    step_id: str
    owner_agent: AgentName
    kind: StepKind
    title: str
    tool_name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    success_criteria: str = ""
    on_failure: str = "replan"
    status: Literal["pending", "completed", "skipped"] = "pending"


class ToolResult(BaseModel):
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    evidence: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    retryable: bool = False


class PendingConfirmation(BaseModel):
    confirmation_id: str
    prompt: str
    tool_name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    owner_agent: AgentName


class CompletedAction(BaseModel):
    step_id: str
    tool_name: str
    status: Literal["completed", "failed"] = "completed"
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class TaskState(BaseModel):
    task_id: str = Field(default_factory=new_task_id)
    session_id: str = "default"
    goal: str
    task_type: TaskType
    task_status: TaskStatus = "created"
    active_agent: AgentName = "supervisor"
    plan: list[PlanStep] = Field(default_factory=list)
    current_step: int = 0
    collected_info: dict[str, Any] = Field(default_factory=dict)
    tool_outputs: dict[str, Any] = Field(default_factory=dict)
    pending_questions: list[str] = Field(default_factory=list)
    pending_confirmations: list[PendingConfirmation] = Field(default_factory=list)
    completed_actions: list[CompletedAction] = Field(default_factory=list)
    error_log: list[str] = Field(default_factory=list)
    user_visible_result: dict[str, Any] = Field(default_factory=dict)
    history: list[str] = Field(default_factory=list)
    last_user_input: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_response(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload["result"] = payload["user_visible_result"]
        return payload

    def record_action(self, action: CompletedAction) -> None:
        for index, existing in enumerate(self.completed_actions):
            if existing.step_id == action.step_id:
                self.completed_actions[index] = action
                return
        self.completed_actions.append(action)
