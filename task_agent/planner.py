from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from task_agent.models import AgentName, PlanStep, TaskState


PlanGenerator = Callable[[TaskState], str | dict[str, Any] | list[dict[str, Any]]]


class PlannerAgent:
    name = "planner_agent"

    def __init__(self, plan_generator: PlanGenerator | None = None):
        self.plan_generator = plan_generator
        if self.plan_generator is None and os.environ.get("TASK_AGENT_PLANNER_MODE", "").lower() == "llm":
            self.plan_generator = self._llm_plan_generator

    def build_plan(self, state: TaskState) -> list[PlanStep]:
        if self.plan_generator is not None:
            generated = self._build_structured_plan(state)
            if generated:
                return generated
        return self._fallback_plan(state)

    def replan_after_failure(self, state: TaskState, failed_tool: str, reason: str) -> list[PlanStep]:
        retry_step = PlanStep(
            step_id=f"replan-{failed_tool}-{len(state.error_log) + 1}",
            owner_agent=self._owner_for_task(state),
            kind="call_tool",
            title=f"Retry or replace failed tool: {failed_tool}",
            tool_name=failed_tool,
            success_criteria=f"Recover from failure: {reason}",
            on_failure="ask_user",
        )
        insert_at = min(max(state.current_step + 1, 0), len(state.plan))
        state.plan.insert(insert_at, retry_step)
        state.error_log.append(f"replan: {failed_tool} - {reason}")
        return state.plan

    def _build_structured_plan(self, state: TaskState) -> list[PlanStep]:
        try:
            raw_plan = self.plan_generator(state)
            steps = self._coerce_steps(raw_plan)
            if not steps:
                raise ValueError("planner returned no steps")
            return [PlanStep.model_validate(step) for step in steps]
        except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            state.error_log.append(f"planner: structured output rejected - {exc}")
            return []

    def _coerce_steps(self, raw_plan: str | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(raw_plan, str):
            raw_plan = self._extract_json(raw_plan)
        if isinstance(raw_plan, dict):
            steps = raw_plan.get("steps", [])
        else:
            steps = raw_plan
        if not isinstance(steps, list):
            raise ValueError("planner output must contain a steps list")
        return steps

    def _extract_json(self, text: str) -> dict[str, Any] | list[dict[str, Any]]:
        payload = text.strip()
        if not payload:
            raise ValueError("planner returned empty output")
        if "```" in payload:
            fenced = [part.strip() for part in payload.split("```") if part.strip()]
            payload = next(
                (
                    part.removeprefix("json").strip()
                    for part in fenced
                    if part.startswith("{") or part.startswith("[") or part.startswith("json")
                ),
                payload,
            )
        return json.loads(payload)

    def _llm_plan_generator(self, state: TaskState) -> str | dict[str, Any] | list[dict[str, Any]]:
        from cs_agent.llm_client import llm_chat

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task planner. Return JSON only with a top-level 'steps' array. "
                    "Each step must include step_id, owner_agent, kind, title, success_criteria, "
                    "and may include tool_name, args, on_failure. Allowed owner_agent values are "
                    "purchase_agent, aftersales_agent, legacy_agent. Allowed kind values are "
                    "ask_user, call_tool, confirm, summarize."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": state.goal,
                        "task_type": state.task_type,
                        "collected_info": state.collected_info,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return llm_chat(messages, temperature=0.1, max_tokens=900)

    def _owner_for_task(self, state: TaskState) -> AgentName:
        if state.task_type == "purchase":
            return "purchase_agent"
        if state.task_type == "aftersales":
            return "aftersales_agent"
        if state.task_type == "charging":
            return "charging_agent"
        if state.task_type == "insurance":
            return "insurance_agent"
        if state.task_type == "complaint":
            return "complaint_agent"
        return "legacy_agent"

    def _fallback_plan(self, state: TaskState) -> list[PlanStep]:
        if state.task_type == "purchase":
            return [
                PlanStep(
                    step_id="purchase-collect",
                    owner_agent="purchase_agent",
                    kind="ask_user",
                    title="收集购车需求",
                    success_criteria="预算、充电条件、用途和空间需求齐备",
                ),
                PlanStep(
                    step_id="purchase-search",
                    owner_agent="purchase_agent",
                    kind="call_tool",
                    title="筛选候选车型",
                    tool_name="search_vehicles",
                    success_criteria="返回至少 2 个候选车型",
                ),
                PlanStep(
                    step_id="purchase-compare",
                    owner_agent="purchase_agent",
                    kind="call_tool",
                    title="生成候选车型对比",
                    tool_name="compare_vehicles",
                    success_criteria="输出结构化对比表",
                ),
                PlanStep(
                    step_id="purchase-cost",
                    owner_agent="purchase_agent",
                    kind="call_tool",
                    title="估算 5 年用车成本",
                    tool_name="calculate_cost",
                    success_criteria="输出结构化 TCO 成本估算",
                ),
                PlanStep(
                    step_id="purchase-charging",
                    owner_agent="purchase_agent",
                    kind="call_tool",
                    title="查询附近充电资源和补贴",
                    tool_name="search_charging_stations",
                    success_criteria="输出城市补能资源和地方补贴信息",
                ),
                PlanStep(
                    step_id="purchase-report",
                    owner_agent="purchase_agent",
                    kind="call_tool",
                    title="生成购车对比报告",
                    tool_name="generate_comparison_report",
                    success_criteria="输出可交付的结构化购车报告",
                ),
                PlanStep(
                    step_id="purchase-confirm",
                    owner_agent="purchase_agent",
                    kind="confirm",
                    title="确认试驾预约",
                    tool_name="book_test_drive",
                    success_criteria="用户确认预约",
                ),
                PlanStep(
                    step_id="purchase-summary",
                    owner_agent="purchase_agent",
                    kind="summarize",
                    title="汇总购车结果",
                    success_criteria="输出推荐和预约凭证",
                ),
            ]
        if state.task_type == "aftersales":
            return [
                PlanStep(
                    step_id="aftersales-collect",
                    owner_agent="aftersales_agent",
                    kind="ask_user",
                    title="收集车辆和故障信息",
                    success_criteria="VIN、城市和时间信息齐备",
                ),
                PlanStep(
                    step_id="aftersales-diagnose",
                    owner_agent="aftersales_agent",
                    kind="call_tool",
                    title="诊断故障并检索知识",
                    tool_name="search_knowledge_base",
                    success_criteria="形成严重级别和建议",
                ),
                PlanStep(
                    step_id="aftersales-centers",
                    owner_agent="aftersales_agent",
                    kind="call_tool",
                    title="查询服务中心",
                    tool_name="search_service_centers",
                    success_criteria="返回至少 1 家服务中心",
                ),
                PlanStep(
                    step_id="aftersales-confirm",
                    owner_agent="aftersales_agent",
                    kind="confirm",
                    title="确认提交工单和预约",
                    tool_name="create_service_ticket",
                    success_criteria="用户确认写入动作",
                ),
                PlanStep(
                    step_id="aftersales-summary",
                    owner_agent="aftersales_agent",
                    kind="summarize",
                    title="汇总售后结果",
                    success_criteria="输出工单和预约凭证",
                ),
            ]
        if state.task_type == "charging":
            return [
                PlanStep(
                    step_id="charging-collect",
                    owner_agent="charging_agent",
                    kind="ask_user",
                    title="收集长途充电规划信息",
                    success_criteria="出发地、目的地和车型齐备",
                ),
                PlanStep(
                    step_id="charging-route",
                    owner_agent="charging_agent",
                    kind="call_tool",
                    title="规划长途路线",
                    tool_name="plan_route",
                    success_criteria="输出总里程、途经点和高速费估算",
                ),
                PlanStep(
                    step_id="charging-stations",
                    owner_agent="charging_agent",
                    kind="call_tool",
                    title="检索沿途充电站",
                    tool_name="search_charging_stations_along_route",
                    success_criteria="按有效续航间隔返回沿途补能点",
                ),
                PlanStep(
                    step_id="charging-plan",
                    owner_agent="charging_agent",
                    kind="call_tool",
                    title="生成分段充电方案",
                    tool_name="generate_charging_plan",
                    success_criteria="输出每段里程、到达电量、充电时长和建议充到多少",
                ),
                PlanStep(
                    step_id="charging-cost",
                    owner_agent="charging_agent",
                    kind="call_tool",
                    title="估算行程时间和费用",
                    tool_name="estimate_trip_cost",
                    success_criteria="输出充电费、高速费和总耗时",
                ),
                PlanStep(
                    step_id="charging-report",
                    owner_agent="charging_agent",
                    kind="call_tool",
                    title="生成充电行程单",
                    tool_name="generate_trip_report",
                    success_criteria="输出可交付的结构化行程单",
                ),
                PlanStep(
                    step_id="charging-summary",
                    owner_agent="charging_agent",
                    kind="summarize",
                    title="汇总充电规划结果",
                    success_criteria="输出停靠方案、费用估算和注意事项",
                ),
            ]
        if state.task_type == "insurance":
            return [
                PlanStep(
                    step_id="insurance-collect",
                    owner_agent="insurance_agent",
                    kind="ask_user",
                    title="收集事故和保险报案信息",
                    success_criteria="VIN、城市、时间、事故类型、受损位置和人员伤亡信息齐备",
                ),
                PlanStep(
                    step_id="insurance-estimate",
                    owner_agent="insurance_agent",
                    kind="call_tool",
                    title="估算维修费用",
                    tool_name="estimate_repair_cost",
                    success_criteria="输出维修项目、费用区间和取证清单",
                ),
                PlanStep(
                    step_id="insurance-impact",
                    owner_agent="insurance_agent",
                    kind="call_tool",
                    title="计算走保影响",
                    tool_name="calculate_claim_impact",
                    success_criteria="比较维修费与次年保费上涨影响",
                ),
                PlanStep(
                    step_id="insurance-confirm",
                    owner_agent="insurance_agent",
                    kind="confirm",
                    title="确认提交保险报案",
                    tool_name="file_insurance_claim",
                    success_criteria="用户确认后生成报案凭证",
                ),
                PlanStep(
                    step_id="insurance-summary",
                    owner_agent="insurance_agent",
                    kind="summarize",
                    title="汇总保险理赔结果",
                    success_criteria="输出理赔建议、报案凭证和下一步材料清单",
                ),
            ]
        if state.task_type == "complaint":
            return [
                PlanStep(
                    step_id="complaint-collect",
                    owner_agent="complaint_agent",
                    kind="ask_user",
                    title="收集投诉和维修记录",
                    success_criteria="VIN、城市、问题、维修次数和未解决状态齐备",
                ),
                PlanStep(
                    step_id="complaint-assess",
                    owner_agent="complaint_agent",
                    kind="call_tool",
                    title="评估投诉升级等级",
                    tool_name="assess_complaint_level",
                    success_criteria="输出投诉等级、是否升级和响应 SLA",
                ),
                PlanStep(
                    step_id="complaint-policy",
                    owner_agent="complaint_agent",
                    kind="call_tool",
                    title="查询政策和三包依据",
                    tool_name="search_policy_or_warranty",
                    success_criteria="输出可引用的政策依据",
                ),
                PlanStep(
                    step_id="complaint-confirm",
                    owner_agent="complaint_agent",
                    kind="confirm",
                    title="确认提交升级投诉",
                    tool_name="create_complaint_ticket",
                    success_criteria="用户确认后生成投诉工单",
                ),
                PlanStep(
                    step_id="complaint-summary",
                    owner_agent="complaint_agent",
                    kind="summarize",
                    title="汇总投诉处理结果",
                    success_criteria="输出投诉工单、政策依据和处理时限",
                ),
            ]
        return [
            PlanStep(
                step_id="legacy-answer",
                owner_agent="legacy_agent",
                kind="summarize",
                title="走兼容回答路径",
                success_criteria="返回旧客服结果",
            )
        ]
