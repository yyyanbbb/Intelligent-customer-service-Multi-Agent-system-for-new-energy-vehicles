from __future__ import annotations

import importlib.util

from task_agent.agents import AftersalesAgent, ChargingAgent, ComplaintAgent, InsuranceAgent, PurchaseAgent, Supervisor
from task_agent.models import TaskState
from task_agent.planner import PlannerAgent


class TaskRuntime:
    def __init__(self):
        self.supervisor = Supervisor()
        self.planner = PlannerAgent()
        self.purchase = PurchaseAgent()
        self.aftersales = AftersalesAgent(self.planner.replan_after_failure)
        self.charging = ChargingAgent()
        self.insurance = InsuranceAgent()
        self.complaint = ComplaintAgent()

    def bootstrap(self, state: TaskState) -> TaskState:
        state.plan = self.planner.build_plan(state)
        return self.supervisor.ingest(state, state.goal)

    def continue_with_input(self, state: TaskState, user_input: str) -> TaskState:
        return self.supervisor.ingest(state, user_input)

    def advance(self, state: TaskState) -> TaskState:
        state.pending_questions = []
        if state.task_type == "purchase":
            return self._sync_plan_progress(self.purchase.run(state))
        if state.task_type == "aftersales":
            return self._sync_plan_progress(self.aftersales.run(state))
        if state.task_type == "charging":
            return self._sync_plan_progress(self.charging.run(state))
        if state.task_type == "insurance":
            return self._sync_plan_progress(self.insurance.run(state))
        if state.task_type == "complaint":
            return self._sync_plan_progress(self.complaint.run(state))
        if state.task_type == "chitchat":
            return self._sync_plan_progress(self._local_chitchat(state))
        return self._sync_plan_progress(self._legacy_or_fallback(state))

    def _local_chitchat(self, state: TaskState) -> TaskState:
        state.active_agent = "legacy_agent"
        state.task_status = "completed"
        state.pending_questions = []
        state.pending_confirmations = []
        state.user_visible_result = {
            "answer": "你好，我是 EV Task Agent。你可以让我帮你完成购车推荐、试驾预约、售后工单、道路救援、长途充电规划、保险理赔或投诉升级。",
            "intent": "chitchat",
            "sources": [],
        }
        return state

    def _legacy_or_fallback(self, state: TaskState) -> TaskState:
        dependency_error = self._legacy_dependency_error()
        if dependency_error:
            return self._legacy_fallback(state, dependency_error)
        try:
            from cs_agent.graph import chat as legacy_chat

            legacy = legacy_chat(state.last_user_input or state.goal, session_id=state.session_id)
        except Exception as exc:
            return self._legacy_fallback(state, f"{type(exc).__name__}: {exc}")

        state.active_agent = "legacy_agent"
        state.task_status = "completed"
        state.user_visible_result = {
            "answer": legacy["answer"],
            "intent": legacy["intent"],
            "sources": legacy.get("sources", []),
        }
        return state

    def _legacy_dependency_error(self) -> str:
        try:
            if importlib.util.find_spec("langgraph.checkpoint.sqlite") is None:
                return "missing dependency: langgraph.checkpoint.sqlite"
        except ModuleNotFoundError as exc:
            return f"missing dependency: {exc.name}"
        return ""

    def _legacy_fallback(self, state: TaskState, reason: str) -> TaskState:
        state.active_agent = "legacy_agent"
        state.task_status = "completed"
        state.pending_questions = []
        state.pending_confirmations = []
        state.error_log.append(f"legacy_agent unavailable: {reason}")
        state.user_visible_result = {
            "answer": "当前旧版 FAQ 兼容图不可用。任务型主线仍可处理购车、售后、充电、保险和投诉闭环任务；请换成具体任务目标，或稍后再试 FAQ。",
            "intent": state.task_type,
            "sources": [],
            "error_code": "legacy_unavailable",
        }
        return state

    def apply_confirmation(self, state: TaskState, confirmation_id: str, approved: bool) -> TaskState:
        if state.task_type == "purchase":
            return self._sync_plan_progress(self.purchase.apply_confirmation(state, confirmation_id, approved))
        if state.task_type == "aftersales":
            return self._sync_plan_progress(self.aftersales.apply_confirmation(state, confirmation_id, approved))
        if state.task_type == "charging":
            state.task_status = "completed"
            return self._sync_plan_progress(state)
        if state.task_type == "insurance":
            return self._sync_plan_progress(self.insurance.apply_confirmation(state, confirmation_id, approved))
        if state.task_type == "complaint":
            return self._sync_plan_progress(self.complaint.apply_confirmation(state, confirmation_id, approved))
        state.task_status = "completed"
        return self._sync_plan_progress(state)

    def _sync_plan_progress(self, state: TaskState) -> TaskState:
        completed_step_ids = {action.step_id for action in state.completed_actions if action.status == "completed"}
        completed_tools = {action.tool_name for action in state.completed_actions if action.status == "completed"}

        for step in state.plan:
            if step.step_id in completed_step_ids or (step.tool_name and step.tool_name in completed_tools):
                step.status = "completed"
            elif step.kind == "ask_user" and state.task_status in {"awaiting_confirmation", "completed"}:
                step.status = "completed"
            elif step.kind == "summarize" and state.task_status == "completed":
                step.status = "completed"

        if state.task_status == "completed":
            state.current_step = len(state.plan)
            return state

        for index, step in enumerate(state.plan):
            if step.status == "pending":
                state.current_step = index
                break
        else:
            state.current_step = len(state.plan)
        return state
