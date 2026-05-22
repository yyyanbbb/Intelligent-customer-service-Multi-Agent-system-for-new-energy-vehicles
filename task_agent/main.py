from __future__ import annotations

import argparse
import json
import sys
from typing import TextIO

from task_agent.service import TaskService


_APPROVE_COMMANDS = {"yes", "y", "approve", "ok", "确认", "同意", "是"}
_REJECT_COMMANDS = {"no", "n", "reject", "cancel", "拒绝", "取消", "否"}


def _handle_interactive_turn(service: TaskService, task_id: str, user_input: str) -> tuple[str, dict]:
    text = user_input.strip()
    if task_id:
        current = service.get_task_status(task_id)
        pending = current.get("pending_confirmations", [])
        lowered = text.lower()
        if current.get("task_status") in {"completed", "failed"}:
            result = service.start_task(text, session_id="cli-task")
            return result["task_id"], result
        if pending and lowered in _APPROVE_COMMANDS | _REJECT_COMMANDS:
            approved = lowered in _APPROVE_COMMANDS
            return task_id, service.confirm_task_action(task_id, pending[0]["confirmation_id"], approved=approved)
        return task_id, service.continue_task(task_id, text)

    result = service.start_task(text, session_id="cli-task")
    return result["task_id"], result


def _run_once(service: TaskService, query: str, *, auto_confirm: bool = False) -> dict:
    result = service.start_task(query, session_id="cli-task")
    if auto_confirm and result.get("pending_confirmations"):
        result = service.confirm_task_action(
            result["task_id"],
            result["pending_confirmations"][0]["confirmation_id"],
            approved=True,
        )
    return result


def _print_json(payload: dict, stream: TextIO | None = None) -> None:
    output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    stream = stream or sys.stdout
    try:
        stream.write(output)
        stream.flush()
    except UnicodeEncodeError:
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            raise
        buffer.write(output.encode("utf-8"))
        buffer.flush()


def _exit_code_for_report(report: dict) -> int:
    return 0 if report.get("ok") is True else 1


def _format_summary(state: dict) -> str:
    result = state.get("result", {})
    lines = [
        f"任务: {state.get('task_id', '')}",
        f"状态: {state.get('task_status', '')}",
        f"负责 Agent: {state.get('active_agent', '')}",
    ]

    recommendation = result.get("recommendation", {})
    if recommendation:
        alternatives = "、".join(recommendation.get("alternatives", [])) or "无"
        lines.append(f"推荐车型: {recommendation.get('primary_model', '')}；备选: {alternatives}")

    comparison = result.get("comparison", [])
    if comparison:
        compact = "；".join(f"{item.get('model', '')} {item.get('price', '')} / {item.get('range_km', 0)}km" for item in comparison[:3])
        lines.append(f"车型对比: {compact}")

    report = result.get("comparison_report", {})
    if report:
        lines.append(f"报告编号: {report.get('report_id', '')} / 推荐: {report.get('recommended_model', '')}")

    ownership_cost = result.get("ownership_cost", {})
    if ownership_cost:
        lines.append(
            "5年用车成本: "
            f"{ownership_cost.get('tco_total', 0):,.0f}元；"
            f"年电费约 {ownership_cost.get('energy_cost_year', 0):,.0f}元"
        )

    subsidy = result.get("subsidy", {})
    if subsidy:
        lines.append(f"补贴: {subsidy.get('city', '')} 预估 {subsidy.get('estimated_amount', 0):,.0f}元")

    stations = result.get("charging_stations", [])
    if stations:
        station_names = "、".join(station.get("name", "") for station in stations[:2])
        lines.append(f"附近补能: {station_names}")

    booking = result.get("booking", {})
    if booking:
        lines.append(
            "预约凭证: "
            f"{booking.get('booking_id', '')} / {booking.get('vehicle_model', '')} / "
            f"{booking.get('city', '')} {booking.get('time_slot', '')}"
        )

    ticket = result.get("ticket", {})
    if ticket:
        lines.append(f"工单凭证: {ticket.get('ticket_id', '')}")

    roadside = result.get("roadside_assistance", {})
    if roadside:
        lines.append(
            "道路救援: "
            f"{roadside.get('rescue_id', '')} / {roadside.get('city', '')} / "
            f"预计 {roadside.get('eta_minutes', 0)} 分钟"
        )

    appointment = result.get("appointment", {})
    if appointment:
        lines.append(
            "到店预约: "
            f"{appointment.get('appointment_id', '')} / {appointment.get('service_center', '')} / "
            f"{appointment.get('time_slot', '')}"
        )

    trip_report = result.get("trip_report", {})
    route = result.get("route", {})
    charging_plan = result.get("charging_plan", {})
    trip_cost = result.get("trip_cost", {})
    if trip_report:
        lines.append(
            "充电行程单: "
            f"{trip_report.get('report_id', '')} / "
            f"{route.get('origin', '')} -> {route.get('destination', '')} / "
            f"{len(charging_plan.get('stops', []))} 次补能 / "
            f"预估 {trip_cost.get('estimated_total_cost', 0):,.0f}元"
        )

    damage_estimate = result.get("damage_estimate", {})
    claim_impact = result.get("claim_impact", {})
    if damage_estimate and claim_impact:
        lines.append(
            "理赔评估: "
            f"{damage_estimate.get('damage_area', '')} / "
            f"维修约 {damage_estimate.get('estimated_repair_cost', 0):,.0f}元 / "
            f"建议 {claim_impact.get('recommendation', '')}"
        )

    claim = result.get("claim", {})
    if claim:
        lines.append(f"保险报案: {claim.get('claim_id', '')} / {claim.get('city', '')} / {claim.get('status', '')}")

    complaint_assessment = result.get("complaint_assessment", {})
    if complaint_assessment:
        lines.append(
            "投诉评估: "
            f"{complaint_assessment.get('issue', '')} / "
            f"{complaint_assessment.get('repair_count', 0)} 次维修 / "
            f"等级 {complaint_assessment.get('level', '')}"
        )

    complaint = result.get("complaint", {})
    if complaint:
        lines.append(
            "投诉工单: "
            f"{complaint.get('complaint_id', '')} / {complaint.get('city', '')} / "
            f"{complaint.get('status', '')} / {complaint.get('expected_response_hours', 0)}小时响应"
        )

    pending_questions = state.get("pending_questions", [])
    if pending_questions:
        lines.append("待补充: " + "；".join(pending_questions))

    pending_confirmations = state.get("pending_confirmations", [])
    if pending_confirmations:
        lines.append("待确认: " + pending_confirmations[0].get("prompt", ""))

    return "\n".join(lines)


def _print_text(text: str, stream: TextIO | None = None) -> None:
    stream = stream or sys.stdout
    try:
        stream.write(text + "\n")
        stream.flush()
    except UnicodeEncodeError:
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            raise
        buffer.write((text + "\n").encode("utf-8"))
        buffer.flush()


def _interactive() -> None:
    service = TaskService()
    print("EV Task Agent 交互模式。输入 exit 退出；待确认时输入 确认/yes 或 拒绝/no。")
    task_id = ""
    while True:
        query = input("你: ").strip()
        if query.lower() in {"exit", "quit"}:
            return
        task_id, result = _handle_interactive_turn(service, task_id, query)
        print(result["result"])
        if result["task_status"] == "awaiting_confirmation":
            print("待确认:", result["pending_confirmations"][0]["prompt"])


def main() -> int:
    parser = argparse.ArgumentParser(description="EV task agent")
    parser.add_argument("--query", "-q", default="", help="单次执行")
    parser.add_argument("--auto-confirm", action="store_true", help="单次执行时自动确认第一个待确认动作")
    parser.add_argument("--summary", action="store_true", help="单次执行时输出适合演示的精简摘要")
    parser.add_argument("--ui", action="store_true", help="启动任务型 Web UI")
    parser.add_argument("--health", action="store_true", help="运行本地诊断并输出 JSON")
    parser.add_argument("--smoke", action="store_true", help="运行上线前 6 条任务主线 smoke 并输出 JSON")
    parser.add_argument("--launch-check", action="store_true", help="运行 health、smoke、compile 和 DB 污染检查")
    args = parser.parse_args()
    if args.health:
        from task_agent.diagnostics import run_diagnostics

        report = run_diagnostics()
        _print_json(report)
        return _exit_code_for_report(report)
    if args.smoke:
        from task_agent.diagnostics import run_launch_smoke

        report = run_launch_smoke()
        _print_json(report)
        return _exit_code_for_report(report)
    if args.launch_check:
        from task_agent.diagnostics import run_launch_check

        report = run_launch_check()
        _print_json(report)
        return _exit_code_for_report(report)
    if args.ui:
        from task_agent.app import launch_ui

        launch_ui(server_name="0.0.0.0", server_port=7861)
        return 0
    if args.query:
        result = _run_once(TaskService(), args.query, auto_confirm=args.auto_confirm)
        if args.summary:
            _print_text(_format_summary(result))
        else:
            _print_json(result)
        return 0
    _interactive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
