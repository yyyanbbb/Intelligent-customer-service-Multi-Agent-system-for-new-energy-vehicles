from __future__ import annotations

from collections.abc import Callable
from typing import Any

from task_agent.models import CompletedAction, PendingConfirmation, TaskState
from task_agent.parsing import (
    aftersales_missing_fields,
    charging_missing_fields,
    complaint_missing_fields,
    insurance_missing_fields,
    merge_collected_info,
    purchase_booking_missing_fields,
    purchase_missing_fields,
)
from task_agent.tools import REGISTRY, compare_vehicles, get_vehicle_detail


class Supervisor:
    name = "supervisor"

    def ingest(self, state: TaskState, text: str) -> TaskState:
        state.history.append(text)
        state.last_user_input = text
        state.collected_info = merge_collected_info(state.collected_info, text)
        return state


class PurchaseAgent:
    name = "purchase_agent"

    def run(self, state: TaskState) -> TaskState:
        state.active_agent = self.name
        missing = purchase_missing_fields(state.collected_info)
        if missing:
            state.task_status = "awaiting_user_input"
            state.pending_questions = missing
            state.pending_confirmations = []
            return state

        tool_result = REGISTRY.call(
            "search_vehicles",
            {
                "budget_max": state.collected_info.get("budget_max", 10**9),
                "need_suv": state.collected_info.get("space_preference") == "SUV",
                "preferred_models": state.collected_info.get("candidate_models", []),
                "use_case": state.collected_info.get("use_case", ""),
                "charging_condition": state.collected_info.get("charging_condition", ""),
            },
        )
        state.tool_outputs["search_vehicles"] = tool_result
        if not tool_result["ok"]:
            state.task_status = "awaiting_user_input"
            state.pending_questions = ["我暂时没有筛到合适车型，能再补充预算或品牌偏好吗？"]
            state.error_log.append(f"search_vehicles: {tool_result['error']}")
            return state

        vehicles = tool_result["data"]["vehicles"]
        models = [vehicle["model"] for vehicle in vehicles[:2]]
        compare_result = compare_vehicles(models)
        state.tool_outputs["compare_vehicles"] = compare_result

        primary_detail = get_vehicle_detail({"model_id": models[0] if models else ""})
        state.tool_outputs["get_vehicle_detail"] = primary_detail
        selected_model = state.collected_info.get("selected_model") or models[0]
        annual_km = int(state.collected_info.get("daily_commute_km", 40)) * 365
        cost_result = REGISTRY.call(
            "calculate_cost",
            {
                "model_id": selected_model,
                "annual_km": annual_km,
                "electricity_price": 0.62 if state.collected_info.get("charging_condition") == "home_charger" else 0.9,
                "years": 5,
            },
        )
        state.tool_outputs["calculate_cost"] = cost_result
        if cost_result["ok"]:
            state.user_visible_result["ownership_cost"] = cost_result["data"]

        if state.collected_info.get("city"):
            subsidy_result = REGISTRY.call(
                "check_subsidy",
                {"model_id": selected_model, "city": state.collected_info["city"]},
            )
            charging_result = REGISTRY.call(
                "search_charging_stations",
                {"city": state.collected_info["city"], "radius_km": 10},
            )
            state.tool_outputs["check_subsidy"] = subsidy_result
            state.tool_outputs["search_charging_stations"] = charging_result
            if subsidy_result["ok"]:
                state.user_visible_result["subsidy"] = subsidy_result["data"]
            if charging_result["ok"]:
                state.user_visible_result["charging_stations"] = charging_result["data"]["stations"]

        recommendation = {
            "primary_model": models[0] if models else "",
            "alternatives": models[1:],
            "reason": "优先满足预算、通勤里程和家用空间需求。",
        }
        state.user_visible_result["recommendation"] = recommendation
        state.user_visible_result["comparison"] = compare_result["data"]["comparison"]
        report_result = REGISTRY.call(
            "generate_comparison_report",
            {
                "model_ids": models,
                "recommendation": recommendation,
                "ownership_cost": state.user_visible_result.get("ownership_cost", {}),
                "subsidy": state.user_visible_result.get("subsidy", {}),
                "charging_stations": state.user_visible_result.get("charging_stations", []),
            },
        )
        state.tool_outputs["generate_comparison_report"] = report_result
        if report_result["ok"]:
            state.user_visible_result["comparison_report"] = report_result["data"]
        state.record_action(
            CompletedAction(
                step_id="purchase-search",
                tool_name="search_vehicles",
                summary="已筛选候选车型",
                evidence=tool_result["evidence"],
                data={"vehicles": vehicles},
            )
        )
        state.record_action(
            CompletedAction(
                step_id="purchase-compare",
                tool_name="compare_vehicles",
                summary="已生成候选车型对比",
                evidence=compare_result["evidence"],
                data=compare_result["data"],
            )
        )
        if report_result["ok"]:
            state.record_action(
                CompletedAction(
                    step_id="purchase-report",
                    tool_name="generate_comparison_report",
                    summary="已生成购车对比报告",
                    evidence=report_result["evidence"],
                    data=report_result["data"],
                )
            )
        if cost_result["ok"]:
            state.record_action(
                CompletedAction(
                    step_id="purchase-cost",
                    tool_name="calculate_cost",
                    summary="已生成 5 年用车成本估算",
                    evidence=cost_result["evidence"],
                    data=cost_result["data"],
                )
            )
        if state.collected_info.get("city") and state.tool_outputs.get("search_charging_stations", {}).get("ok"):
            state.record_action(
                CompletedAction(
                    step_id="purchase-charging",
                    tool_name="search_charging_stations",
                    summary="已查询附近补能资源",
                    evidence=state.tool_outputs["search_charging_stations"]["evidence"],
                    data=state.tool_outputs["search_charging_stations"]["data"],
                )
            )

        booking_missing = purchase_booking_missing_fields(state.collected_info)
        if booking_missing:
            state.task_status = "awaiting_user_input"
            state.pending_questions = booking_missing
            return state

        state.task_status = "awaiting_confirmation"
        state.pending_questions = []
        state.pending_confirmations = [
            PendingConfirmation(
                confirmation_id="confirm-1",
                prompt=f"是否为您预约 {state.collected_info['selected_model']} 在 {state.collected_info['city']} {state.collected_info['time_slot']} 的试驾？",
                tool_name="book_test_drive",
                owner_agent=self.name,
                payload={
                    "vehicle_model": state.collected_info["selected_model"],
                    "city": state.collected_info["city"],
                    "time_slot": state.collected_info["time_slot"],
                    "name": "演示用户",
                },
            )
        ]
        return state

    def apply_confirmation(self, state: TaskState, confirmation_id: str, approved: bool) -> TaskState:
        state.active_agent = self.name
        confirmation = next((item for item in state.pending_confirmations if item.confirmation_id == confirmation_id), None)
        if confirmation is None:
            raise KeyError(f"Unknown confirmation_id: {confirmation_id}")
        state.pending_confirmations = []
        if not approved:
            state.task_status = "awaiting_user_input"
            state.pending_questions = ["好的，我先不预约试驾。您可以告诉我想改哪个车型、城市或时间段。"]
            return state

        result = REGISTRY.call("book_test_drive", confirmation.payload)
        state.tool_outputs["book_test_drive"] = result
        state.user_visible_result["booking"] = result["data"]
        state.record_action(
            CompletedAction(
                step_id="purchase-confirm",
                tool_name="book_test_drive",
                summary="已生成试驾预约凭证",
                evidence=result["evidence"],
                data=result["data"],
            )
        )
        state.task_status = "completed"
        return state


class ChargingAgent:
    name = "charging_agent"

    def run(self, state: TaskState) -> TaskState:
        state.active_agent = self.name
        missing = charging_missing_fields(state.collected_info)
        if missing:
            state.task_status = "awaiting_user_input"
            state.pending_questions = missing
            state.pending_confirmations = []
            return state

        route_result = REGISTRY.call(
            "plan_route",
            {
                "origin": state.collected_info["origin"],
                "destination": state.collected_info["destination"],
            },
        )
        state.tool_outputs["plan_route"] = route_result
        if not route_result["ok"]:
            state.task_status = "awaiting_user_input"
            state.pending_questions = ["请补充有效的出发地和目的地。"]
            state.error_log.append(f"plan_route: {route_result['error']}")
            return state

        stations_result = REGISTRY.call(
            "search_charging_stations_along_route",
            {"route": route_result["data"], "interval_km": 350},
        )
        state.tool_outputs["search_charging_stations_along_route"] = stations_result

        plan_result = REGISTRY.call(
            "generate_charging_plan",
            {
                "route": route_result["data"],
                "stations": stations_result["data"].get("stations", []),
                "vehicle_model": state.collected_info["selected_model"],
            },
        )
        state.tool_outputs["generate_charging_plan"] = plan_result

        cost_result = REGISTRY.call(
            "estimate_trip_cost",
            {"route": route_result["data"], "charging_plan": plan_result["data"]},
        )
        state.tool_outputs["estimate_trip_cost"] = cost_result

        report_result = REGISTRY.call(
            "generate_trip_report",
            {
                "route": route_result["data"],
                "charging_plan": plan_result["data"],
                "trip_cost": cost_result["data"],
            },
        )
        state.tool_outputs["generate_trip_report"] = report_result

        state.user_visible_result.update(
            {
                "route": route_result["data"],
                "charging_stations_along_route": stations_result["data"].get("stations", []),
                "charging_plan": plan_result["data"],
                "trip_cost": cost_result["data"],
                "trip_report": report_result["data"],
            }
        )
        for step_id, tool_name, result, summary in (
            ("charging-route", "plan_route", route_result, "已规划长途路线"),
            ("charging-stations", "search_charging_stations_along_route", stations_result, "已检索沿途充电站"),
            ("charging-plan", "generate_charging_plan", plan_result, "已生成分段充电计划"),
            ("charging-cost", "estimate_trip_cost", cost_result, "已估算行程费用和时间"),
            ("charging-report", "generate_trip_report", report_result, "已生成充电行程单"),
        ):
            state.record_action(
                CompletedAction(
                    step_id=step_id,
                    tool_name=tool_name,
                    summary=summary,
                    evidence=result["evidence"],
                    data=result["data"],
                )
            )
        state.pending_questions = []
        state.pending_confirmations = []
        state.task_status = "completed"
        return state


class InsuranceAgent:
    name = "insurance_agent"

    def run(self, state: TaskState) -> TaskState:
        state.active_agent = self.name
        missing = insurance_missing_fields(state.collected_info)
        if missing:
            state.task_status = "awaiting_user_input"
            state.pending_questions = missing
            state.pending_confirmations = []
            return state

        damage_result = REGISTRY.call(
            "estimate_repair_cost",
            {
                "damage_area": state.collected_info["damage_area"],
                "severity": "minor",
            },
        )
        state.tool_outputs["estimate_repair_cost"] = damage_result
        impact_result = REGISTRY.call(
            "calculate_claim_impact",
            {
                "repair_cost": damage_result["data"].get("estimated_repair_cost", 0),
                "no_claim_years": state.collected_info.get("no_claim_years", 1),
            },
        )
        state.tool_outputs["calculate_claim_impact"] = impact_result

        state.user_visible_result["damage_estimate"] = damage_result["data"]
        state.user_visible_result["claim_impact"] = impact_result["data"]
        state.record_action(
            CompletedAction(
                step_id="insurance-estimate",
                tool_name="estimate_repair_cost",
                summary="已估算维修费用",
                evidence=damage_result["evidence"],
                data=damage_result["data"],
            )
        )
        state.record_action(
            CompletedAction(
                step_id="insurance-impact",
                tool_name="calculate_claim_impact",
                summary="已计算走保影响",
                evidence=impact_result["evidence"],
                data=impact_result["data"],
            )
        )

        state.task_status = "awaiting_confirmation"
        state.pending_questions = []
        state.pending_confirmations = [
            PendingConfirmation(
                confirmation_id="confirm-1",
                prompt=(
                    f"预计维修 {damage_result['data'].get('estimated_repair_cost', 0)} 元，"
                    f"建议 {impact_result['data'].get('recommendation', '')}。是否提交保险报案？"
                ),
                tool_name="file_insurance_claim",
                owner_agent=self.name,
                payload={
                    "vin": state.collected_info["vin"],
                    "city": state.collected_info["city"],
                    "time_slot": state.collected_info["time_slot"],
                    "accident_type": state.collected_info["accident_type"],
                    "damage_area": state.collected_info["damage_area"],
                    "repair_cost": damage_result["data"].get("estimated_repair_cost", 0),
                },
            )
        ]
        return state

    def apply_confirmation(self, state: TaskState, confirmation_id: str, approved: bool) -> TaskState:
        state.active_agent = self.name
        confirmation = next((item for item in state.pending_confirmations if item.confirmation_id == confirmation_id), None)
        if confirmation is None:
            raise KeyError(f"Unknown confirmation_id: {confirmation_id}")
        state.pending_confirmations = []
        if not approved:
            state.task_status = "awaiting_user_input"
            state.pending_questions = ["好的，我先不提交保险报案。您可以选择自费维修或重新补充事故信息。"]
            return state

        result = REGISTRY.call("file_insurance_claim", confirmation.payload)
        state.tool_outputs["file_insurance_claim"] = result
        state.user_visible_result["claim"] = result["data"]
        state.record_action(
            CompletedAction(
                step_id="insurance-confirm",
                tool_name="file_insurance_claim",
                summary="已生成保险报案凭证",
                evidence=result["evidence"],
                data=result["data"],
            )
        )
        state.task_status = "completed"
        return state


class ComplaintAgent:
    name = "complaint_agent"

    def run(self, state: TaskState) -> TaskState:
        state.active_agent = self.name
        missing = complaint_missing_fields(state.collected_info)
        if missing:
            state.task_status = "awaiting_user_input"
            state.pending_questions = missing
            state.pending_confirmations = []
            return state

        issue = state.collected_info.get("complaint_issue", state.goal)
        assessment = REGISTRY.call(
            "assess_complaint_level",
            {
                "issue": issue,
                "repair_count": state.collected_info.get("repair_count", 0),
                "unresolved": state.collected_info.get("unresolved", False),
            },
        )
        policy = REGISTRY.call("search_policy_or_warranty", {"query": "三包 同一故障 多次维修 未解决"})
        state.tool_outputs["assess_complaint_level"] = assessment
        state.tool_outputs["search_policy_or_warranty"] = policy
        state.user_visible_result["complaint_assessment"] = assessment["data"]
        state.user_visible_result["policy_basis"] = policy["data"]
        state.record_action(
            CompletedAction(
                step_id="complaint-assess",
                tool_name="assess_complaint_level",
                summary="已评估投诉升级等级",
                evidence=assessment["evidence"],
                data=assessment["data"],
            )
        )
        state.record_action(
            CompletedAction(
                step_id="complaint-policy",
                tool_name="search_policy_or_warranty",
                summary="已查询投诉政策依据",
                evidence=policy["evidence"],
                data=policy["data"],
            )
        )

        state.task_status = "awaiting_confirmation"
        state.pending_questions = []
        state.pending_confirmations = [
            PendingConfirmation(
                confirmation_id="confirm-1",
                prompt=(
                    f"同一问题已维修 {state.collected_info.get('repair_count', 0)} 次且仍未解决，"
                    f"投诉等级为 {assessment['data'].get('level', '')}。是否提交升级投诉？"
                ),
                tool_name="create_complaint_ticket",
                owner_agent=self.name,
                payload={
                    "vin": state.collected_info["vin"],
                    "city": state.collected_info["city"],
                    "issue": issue,
                    "repair_count": state.collected_info.get("repair_count", 0),
                    "work_orders": state.collected_info.get("work_orders", []),
                    "priority": assessment["data"].get("level", "normal"),
                    "policy_basis": policy["evidence"],
                    "expected_response_hours": assessment["data"].get("sla_hours", 72),
                },
            )
        ]
        return state

    def apply_confirmation(self, state: TaskState, confirmation_id: str, approved: bool) -> TaskState:
        state.active_agent = self.name
        confirmation = next((item for item in state.pending_confirmations if item.confirmation_id == confirmation_id), None)
        if confirmation is None:
            raise KeyError(f"Unknown confirmation_id: {confirmation_id}")
        state.pending_confirmations = []
        if not approved:
            state.task_status = "awaiting_user_input"
            state.pending_questions = ["好的，我先不提交投诉。您可以补充维修记录或调整诉求后再继续。"]
            return state

        ticket = REGISTRY.call("create_complaint_ticket", confirmation.payload)
        state.tool_outputs["create_complaint_ticket"] = ticket
        state.user_visible_result["complaint"] = ticket["data"]
        state.record_action(
            CompletedAction(
                step_id="complaint-confirm",
                tool_name="create_complaint_ticket",
                summary="已生成投诉工单",
                evidence=ticket["evidence"],
                data=ticket["data"],
            )
        )
        state.task_status = "completed"
        return state


class AftersalesAgent:
    name = "aftersales_agent"
    _critical_faults = {"刹车", "失灵", "起火", "冒烟"}

    def __init__(self, replan_on_failure: Callable[[TaskState, str, str], object] | None = None):
        self.replan_on_failure = replan_on_failure

    def run(self, state: TaskState) -> TaskState:
        state.active_agent = self.name
        missing = aftersales_missing_fields(state.collected_info)
        if missing:
            state.task_status = "awaiting_user_input"
            state.pending_questions = missing
            state.pending_confirmations = []
            return state

        diagnosis = self._diagnose(state.collected_info)
        knowledge = REGISTRY.call("search_knowledge_base", {"query": state.goal})
        state.tool_outputs["search_knowledge_base"] = knowledge
        state.user_visible_result["diagnosis"] = diagnosis

        centers = self._call_with_retry("search_service_centers", {"city": state.collected_info["city"]}, state)
        if not centers["ok"]:
            state.task_status = "awaiting_user_input"
            state.pending_questions = [f"我暂时查不到 {state.collected_info['city']} 的服务中心，请换一个城市或时间再试。"]
            return state

        state.user_visible_result["service_centers"] = centers["data"]["centers"]
        state.record_action(
            CompletedAction(
                step_id="aftersales-centers",
                tool_name="search_service_centers",
                summary="已查询到服务中心",
                evidence=centers["evidence"],
                data=centers["data"],
            )
        )

        state.task_status = "awaiting_confirmation"
        state.pending_questions = []
        wants_roadside = state.collected_info.get("roadside_preference") == "requested"
        confirmation_tool = "request_roadside_assistance" if wants_roadside else "create_service_ticket"
        confirmation_prompt = (
            f"当前故障涉及行车安全。是否为您提交工单，并派发道路救援拖至 {centers['data']['centers'][0]['name']}？"
            if wants_roadside
            else f"是否为您提交工单并预约 {state.collected_info['city']} {state.collected_info['time_slot']} 到店检查？"
        )
        self._sync_confirmation_plan_step(state, wants_roadside)
        state.pending_confirmations = [
            PendingConfirmation(
                confirmation_id="confirm-1",
                prompt=confirmation_prompt,
                tool_name=confirmation_tool,
                owner_agent=self.name,
                payload={
                    "vehicle_model": state.collected_info.get("selected_model", "Model Y"),
                    "issue_description": state.goal,
                    "components": state.collected_info.get("components", []),
                    "faults": state.collected_info.get("faults", []),
                    "vin": state.collected_info["vin"],
                    "city": state.collected_info["city"],
                    "time_slot": state.collected_info["time_slot"],
                    "service_center": centers["data"]["centers"][0]["name"],
                    "request_roadside": wants_roadside,
                },
            )
        ]
        return state

    def _sync_confirmation_plan_step(self, state: TaskState, wants_roadside: bool) -> None:
        for step in state.plan:
            if step.step_id != "aftersales-confirm":
                continue
            if wants_roadside:
                step.title = "确认道路救援和工单"
                step.tool_name = "request_roadside_assistance"
                step.success_criteria = "用户确认派发道路救援并生成工单"
            else:
                step.title = "确认提交工单和预约"
                step.tool_name = "create_service_ticket"
                step.success_criteria = "用户确认写入动作"
            return

    def apply_confirmation(self, state: TaskState, confirmation_id: str, approved: bool) -> TaskState:
        state.active_agent = self.name
        confirmation = next((item for item in state.pending_confirmations if item.confirmation_id == confirmation_id), None)
        if confirmation is None:
            raise KeyError(f"Unknown confirmation_id: {confirmation_id}")
        state.pending_confirmations = []
        if not approved:
            state.task_status = "awaiting_user_input"
            state.pending_questions = ["好的，我先不提交工单。您可以告诉我是否要改时间、城市，或仅保留诊断结果。"]
            return state

        ticket_result = REGISTRY.call("create_service_ticket", confirmation.payload)
        state.tool_outputs["create_service_ticket"] = ticket_result
        state.user_visible_result["ticket"] = ticket_result["data"]
        state.record_action(
            CompletedAction(
                step_id="aftersales-confirm-ticket",
                tool_name="create_service_ticket",
                summary="已生成售后工单",
                evidence=ticket_result["evidence"],
                data=ticket_result["data"],
            )
        )

        if confirmation.payload.get("request_roadside"):
            roadside_result = REGISTRY.call(
                "request_roadside_assistance",
                {
                    "vehicle_model": confirmation.payload["vehicle_model"],
                    "city": confirmation.payload["city"],
                    "issue_type": confirmation.payload["issue_description"],
                    "vin": confirmation.payload["vin"],
                    "service_center": confirmation.payload["service_center"],
                    "ticket_id": ticket_result["data"]["ticket_id"],
                },
            )
            state.tool_outputs["request_roadside_assistance"] = roadside_result
            state.user_visible_result["roadside_assistance"] = roadside_result["data"]
            state.record_action(
                CompletedAction(
                    step_id="aftersales-confirm-roadside",
                    tool_name="request_roadside_assistance",
                    summary="已派发道路救援",
                    evidence=roadside_result["evidence"],
                    data=roadside_result["data"],
                )
            )
        else:
            appointment_result = REGISTRY.call(
                "book_service_appointment",
                {
                    "vehicle_model": confirmation.payload["vehicle_model"],
                    "city": confirmation.payload["city"],
                    "time_slot": confirmation.payload["time_slot"],
                    "ticket_id": ticket_result["data"]["ticket_id"],
                    "service_center": confirmation.payload["service_center"],
                },
            )
            state.tool_outputs["book_service_appointment"] = appointment_result
            state.user_visible_result["appointment"] = appointment_result["data"]
            state.record_action(
                CompletedAction(
                    step_id="aftersales-confirm-appointment",
                    tool_name="book_service_appointment",
                    summary="已生成到店预约凭证",
                    evidence=appointment_result["evidence"],
                    data=appointment_result["data"],
                )
            )
        state.task_status = "completed"
        return state

    def _diagnose(self, info: dict[str, Any]) -> dict[str, Any]:
        raw_faults = " ".join(info.get("faults", []) + info.get("components", []))
        severity = "normal"
        if any(keyword in raw_faults or keyword in info.get("last_user_input", "") for keyword in self._critical_faults):
            severity = "critical"
        elif info.get("faults"):
            severity = "urgent"
        return {
            "severity": severity,
            "likely_causes": info.get("faults", []),
            "immediate_actions": ["尽量减少驾驶，避免急刹和高速行驶。"] if severity in {"urgent", "critical"} else [],
        }

    def _call_with_retry(self, tool_name: str, payload: dict[str, Any], state: TaskState) -> dict[str, Any]:
        result = REGISTRY.call(tool_name, payload)
        if result["ok"]:
            return result
        state.error_log.append(f"{tool_name}: {result['error']}")
        if result["retryable"]:
            if self.replan_on_failure is not None:
                self.replan_on_failure(state, tool_name, result["error"])
            retry = REGISTRY.call(tool_name, payload)
            if retry["ok"]:
                state.error_log.append(f"{tool_name}: recovered after retry")
                return retry
            state.error_log.append(f"{tool_name}: retry failed - {retry['error']}")
            return retry
        return result
