from __future__ import annotations

from pathlib import Path
import io

import pytest


@pytest.fixture()
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TASK_AGENT_DB_PATH", str(tmp_path / "cli.db"))
    from task_agent.service import TaskService

    return TaskService()


def _make_pending_purchase(service):
    created = service.start_task("我想买台电车", session_id="cli-confirm")
    progressed = service.continue_task(
        created["task_id"],
        "预算25万，家里有充电桩，每天通勤60公里，家用SUV。我选小鹏G6，上海，明天下午试驾。",
    )
    assert progressed["task_status"] == "awaiting_confirmation"
    return progressed["task_id"]


def test_cli_turn_approves_pending_confirmation(service):
    from task_agent.main import _handle_interactive_turn

    task_id = _make_pending_purchase(service)

    next_task_id, result = _handle_interactive_turn(service, task_id, "确认")

    assert next_task_id == task_id
    assert result["task_status"] == "completed"
    assert result["result"]["booking"]["booking_id"].startswith("TD-")


def test_cli_turn_rejects_pending_confirmation(service):
    from task_agent.main import _handle_interactive_turn

    task_id = _make_pending_purchase(service)

    next_task_id, result = _handle_interactive_turn(service, task_id, "no")

    assert next_task_id == task_id
    assert result["task_status"] == "awaiting_user_input"
    assert result["pending_questions"]


def test_cli_run_once_can_auto_confirm_completed_task(service):
    from task_agent.main import _run_once

    result = _run_once(
        service,
        "预算25万，家里有充电桩，每天通勤60公里，家用SUV。我选小鹏G6，上海，明天下午试驾。",
        auto_confirm=True,
    )

    assert result["task_status"] == "completed"
    assert result["result"]["booking"]["booking_id"].startswith("TD-")


def test_cli_turn_starts_new_task_after_previous_task_completed(service):
    from task_agent.main import _handle_interactive_turn

    completed = service.start_task(
        "下周要从上海开到成都，Model Y 长续航，帮我规划充电方案",
        session_id="cli-completed-then-new",
    )
    assert completed["task_status"] == "completed"

    next_task_id, result = _handle_interactive_turn(service, completed["task_id"], "今天倒车时刮了右后门，帮我走保险")

    assert next_task_id != completed["task_id"]
    assert result["task_type"] == "insurance"
    assert result["task_status"] == "awaiting_user_input"


def test_cli_json_print_falls_back_to_utf8_when_stdout_encoding_rejects_character():
    from task_agent.main import _print_json

    raw = io.BytesIO()
    gbk_stdout = io.TextIOWrapper(raw, encoding="gbk", errors="strict")

    _print_json({"text": "Tesla®"}, stream=gbk_stdout)
    gbk_stdout.flush()

    assert "Tesla®" in raw.getvalue().decode("utf-8")


def test_cli_summary_is_concise_and_hides_internal_tool_outputs(service):
    from task_agent.main import _format_summary, _run_once

    result = _run_once(
        service,
        "预算25万，家里有充电桩，每天通勤60公里，家用SUV。我选小鹏G6，上海，明天下午试驾。",
        auto_confirm=True,
    )

    summary = _format_summary(result)

    assert "completed" in summary
    assert "推荐车型" in summary
    assert "预约凭证" in summary
    assert "报告编号" in summary
    assert "CR-" in summary
    assert "5年用车成本" in summary
    assert "tool_outputs" not in summary
    assert len(summary) < 1200


def test_cli_summary_includes_roadside_assistance(service):
    from task_agent.main import _format_summary

    created = service.start_task("我的Model Y刹车失灵，需要道路救援拖到服务中心", session_id="cli-roadside")
    progressed = service.continue_task(
        created["task_id"],
        "VIN12345678901234567，上海，明天下午处理，需要道路救援。",
    )
    completed = service.confirm_task_action(
        progressed["task_id"],
        progressed["pending_confirmations"][0]["confirmation_id"],
        approved=True,
    )

    summary = _format_summary(completed)

    assert "道路救援" in summary
    assert "RA-" in summary


def test_cli_summary_includes_charging_trip_plan(service):
    from task_agent.main import _format_summary

    result = service.start_task("下周要从上海开到成都，Model Y 长续航，帮我规划充电方案", session_id="cli-charging")

    summary = _format_summary(result)

    assert "充电行程单" in summary
    assert "TP-" in summary
    assert "上海 -> 成都" in summary


def test_cli_summary_includes_insurance_claim(service):
    from task_agent.main import _format_summary

    created = service.start_task("今天倒车时刮了右后门，帮我走保险", session_id="cli-insurance")
    progressed = service.continue_task(
        created["task_id"],
        "VIN12345678901234567，上海，今天下午，单方事故，没有人员伤亡，右后门刮擦，走保险。",
    )
    completed = service.confirm_task_action(
        progressed["task_id"],
        progressed["pending_confirmations"][0]["confirmation_id"],
        approved=True,
    )

    summary = _format_summary(completed)

    assert "保险报案" in summary
    assert "IC-" in summary


def test_cli_summary_includes_complaint_ticket(service):
    from task_agent.main import _format_summary

    created = service.start_task("车门异响修了三次还没好，我要投诉", session_id="cli-complaint")
    progressed = service.continue_task(
        created["task_id"],
        "VIN12345678901234567，上海，车门异响，已经去4S店维修3次，工单SV001、SV002、SV003，仍未解决，我要升级投诉。",
    )
    completed = service.confirm_task_action(
        progressed["task_id"],
        progressed["pending_confirmations"][0]["confirmation_id"],
        approved=True,
    )

    summary = _format_summary(completed)

    assert "投诉工单" in summary
    assert "CP-" in summary
