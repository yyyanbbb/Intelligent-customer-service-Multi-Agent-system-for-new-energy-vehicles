from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture()
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TASK_AGENT_DB_PATH", str(tmp_path / "task_agent.db"))
    from task_agent.service import TaskService

    return TaskService()


def test_purchase_flow_collects_info_then_books_test_drive(service):
    created = service.start_task("我想买台电车", session_id="purchase-1")

    assert created["task_type"] == "purchase"
    assert created["task_status"] == "awaiting_user_input"
    assert "预算" in "".join(created["pending_questions"])

    progressed = service.continue_task(
        created["task_id"],
        "预算25万，家里有充电桩，每天通勤60公里，家用SUV。我选小鹏G6，上海，明天下午试驾。",
    )

    assert progressed["task_status"] == "awaiting_confirmation"
    assert progressed["pending_confirmations"]
    assert progressed["active_agent"] == "purchase_agent"
    assert progressed["result"]["recommendation"]["primary_model"]
    assert progressed["result"]["comparison"]
    assert progressed["result"]["ownership_cost"]["tco_total"] > 0
    assert progressed["result"]["comparison_report"]["report_id"].startswith("CR-")
    assert progressed["result"]["comparison_report"]["recommended_model"]
    assert progressed["result"]["subsidy"]["city"] == "上海"
    assert progressed["result"]["charging_stations"]

    completed = service.confirm_task_action(created["task_id"], confirmation_id="confirm-1", approved=True)

    assert completed["task_status"] == "completed"
    assert completed["result"]["booking"]["booking_id"].startswith("TD-")
    assert completed["completed_actions"][-1]["tool_name"] == "book_test_drive"


def test_plan_progress_tracks_next_step_and_done(service):
    created = service.start_task("我想买台电车", session_id="purchase-progress")
    progressed = service.continue_task(
        created["task_id"],
        "预算25万，家里有充电桩，每天通勤60公里，家用SUV。我选小鹏G6，上海，明天下午试驾。",
    )

    by_id = {step["step_id"]: step for step in progressed["plan"]}
    assert by_id["purchase-search"]["status"] == "completed"
    assert by_id["purchase-compare"]["status"] == "completed"
    assert by_id["purchase-cost"]["status"] == "completed"
    assert by_id["purchase-charging"]["status"] == "completed"
    assert by_id["purchase-report"]["status"] == "completed"
    assert by_id["purchase-report"]["tool_name"] == "generate_comparison_report"
    assert progressed["plan"][progressed["current_step"]]["step_id"] == "purchase-confirm"

    completed = service.confirm_task_action(created["task_id"], confirmation_id="confirm-1", approved=True)

    by_id = {step["step_id"]: step for step in completed["plan"]}
    assert by_id["purchase-confirm"]["status"] == "completed"
    assert by_id["purchase-summary"]["status"] == "completed"
    assert completed["current_step"] == len(completed["plan"])


def test_purchase_multiturn_booking_does_not_duplicate_completed_actions(service):
    created = service.start_task("我想买台电车", session_id="purchase-dedupe")
    recommended = service.continue_task(
        created["task_id"],
        "预算25万，家里有充电桩，每天通勤60公里，家用SUV。",
    )

    assert recommended["task_status"] == "awaiting_user_input"
    assert "recommendation" in recommended["result"]
    first_action_ids = [action["step_id"] for action in recommended["completed_actions"]]
    assert first_action_ids.count("purchase-search") == 1
    assert first_action_ids.count("purchase-compare") == 1

    progressed = service.continue_task(
        created["task_id"],
        "我选小鹏G6，上海，明天下午试驾。",
    )

    action_ids = [action["step_id"] for action in progressed["completed_actions"]]
    assert progressed["task_status"] == "awaiting_confirmation"
    assert action_ids.count("purchase-search") == 1
    assert action_ids.count("purchase-compare") == 1
    assert action_ids.count("purchase-cost") == 1
    assert action_ids.count("purchase-report") == 1


def test_aftersales_flow_handles_safety_then_creates_ticket_and_appointment(service):
    created = service.start_task("我的Model Y昨天开始刹车异响，帮我处理一下", session_id="aftersales-1")

    assert created["task_type"] == "aftersales"
    assert created["task_status"] == "awaiting_user_input"
    assert any("VIN" in question or "城市" in question for question in created["pending_questions"])

    progressed = service.continue_task(
        created["task_id"],
        "VIN12345678901234567，上海，明天下午到店。我不需要道路救援，直接预约维修。",
    )

    assert progressed["task_status"] == "awaiting_confirmation"
    assert progressed["pending_confirmations"]
    assert progressed["result"]["diagnosis"]["severity"] in {"urgent", "critical"}
    assert progressed["result"]["service_centers"]

    completed = service.confirm_task_action(created["task_id"], confirmation_id="confirm-1", approved=True)

    assert completed["task_status"] == "completed"
    assert completed["result"]["ticket"]["ticket_id"].startswith("SV-")
    assert completed["result"]["appointment"]["appointment_id"].startswith("SA-")


def test_aftersales_flow_dispatches_roadside_assistance_when_requested(service):
    created = service.start_task(
        "我的Model Y刹车失灵，需要道路救援拖到服务中心",
        session_id="roadside-1",
    )

    progressed = service.continue_task(
        created["task_id"],
        "VIN12345678901234567，上海，明天下午处理，需要道路救援。",
    )

    assert progressed["task_status"] == "awaiting_confirmation"
    assert progressed["pending_confirmations"][0]["tool_name"] == "request_roadside_assistance"
    assert progressed["result"]["diagnosis"]["severity"] == "critical"
    by_id = {step["step_id"]: step for step in progressed["plan"]}
    assert by_id["aftersales-confirm"]["tool_name"] == "request_roadside_assistance"
    assert "救援" in by_id["aftersales-confirm"]["title"]

    completed = service.confirm_task_action(created["task_id"], confirmation_id="confirm-1", approved=True)

    assert completed["task_status"] == "completed"
    assert completed["result"]["ticket"]["ticket_id"].startswith("SV-")
    assert completed["result"]["roadside_assistance"]["rescue_id"].startswith("RA-")
    assert completed["completed_actions"][-1]["tool_name"] == "request_roadside_assistance"
    by_id = {step["step_id"]: step for step in completed["plan"]}
    assert by_id["aftersales-confirm"]["status"] == "completed"


def test_charging_trip_flow_generates_route_and_charging_plan(service):
    result = service.start_task(
        "下周要从上海开到成都，Model Y 长续航，帮我规划充电方案",
        session_id="charging-1",
    )

    assert result["task_type"] == "charging"
    assert result["task_status"] == "completed"
    assert result["active_agent"] == "charging_agent"
    assert result["result"]["route"]["origin"] == "上海"
    assert result["result"]["route"]["destination"] == "成都"
    assert result["result"]["charging_plan"]["stops"]
    assert result["result"]["trip_cost"]["estimated_total_cost"] > 0
    assert result["result"]["trip_report"]["report_id"].startswith("TP-")

    by_id = {step["step_id"]: step for step in result["plan"]}
    assert by_id["charging-route"]["status"] == "completed"
    assert by_id["charging-stations"]["status"] == "completed"
    assert by_id["charging-plan"]["status"] == "completed"


def test_insurance_claim_flow_estimates_damage_then_files_claim(service):
    created = service.start_task("今天倒车时刮了右后门，帮我走保险", session_id="insurance-1")

    assert created["task_type"] == "insurance"
    assert created["task_status"] == "awaiting_user_input"
    assert any("VIN" in question or "城市" in question for question in created["pending_questions"])

    progressed = service.continue_task(
        created["task_id"],
        "VIN12345678901234567，上海，今天下午，单方事故，没有人员伤亡，右后门刮擦，走保险。",
    )

    assert progressed["task_status"] == "awaiting_confirmation"
    assert progressed["active_agent"] == "insurance_agent"
    assert progressed["pending_confirmations"][0]["tool_name"] == "file_insurance_claim"
    assert progressed["result"]["damage_estimate"]["estimated_repair_cost"] > 0
    assert progressed["result"]["claim_impact"]["recommendation"] in {"claim", "self_pay"}

    completed = service.confirm_task_action(created["task_id"], confirmation_id="confirm-1", approved=True)

    assert completed["task_status"] == "completed"
    assert completed["result"]["claim"]["claim_id"].startswith("IC-")
    assert completed["completed_actions"][-1]["tool_name"] == "file_insurance_claim"
    by_id = {step["step_id"]: step for step in completed["plan"]}
    assert by_id["insurance-confirm"]["status"] == "completed"


def test_complaint_flow_escalates_repeated_unresolved_service_issue(service):
    created = service.start_task("车门异响修了三次还没好，我要投诉", session_id="complaint-1")

    assert created["task_type"] == "complaint"
    assert created["task_status"] == "awaiting_user_input"
    assert any("VIN" in question or "维修" in question for question in created["pending_questions"])

    progressed = service.continue_task(
        created["task_id"],
        "VIN12345678901234567，上海，车门异响，已经去4S店维修3次，工单SV001、SV002、SV003，仍未解决，我要升级投诉。",
    )

    assert progressed["task_status"] == "awaiting_confirmation"
    assert progressed["active_agent"] == "complaint_agent"
    assert progressed["pending_confirmations"][0]["tool_name"] == "create_complaint_ticket"
    assert progressed["result"]["complaint_assessment"]["level"] == "high"
    assert progressed["result"]["policy_basis"]["policies"]

    completed = service.confirm_task_action(created["task_id"], confirmation_id="confirm-1", approved=True)

    assert completed["task_status"] == "completed"
    assert completed["result"]["complaint"]["complaint_id"].startswith("CP-")
    assert completed["completed_actions"][-1]["tool_name"] == "create_complaint_ticket"
    by_id = {step["step_id"]: step for step in completed["plan"]}
    assert by_id["complaint-confirm"]["status"] == "completed"


def test_replans_when_service_center_lookup_fails(service, monkeypatch: pytest.MonkeyPatch):
    from task_agent import tools as task_tools

    original = task_tools.search_service_centers

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(task_tools, "search_service_centers", original)
        return {
            "ok": False,
            "data": {"centers": []},
            "error": "provider timeout",
            "evidence": ["mock provider timeout"],
            "requires_confirmation": False,
            "retryable": True,
        }

    monkeypatch.setattr(task_tools, "search_service_centers", fail_once)

    created = service.start_task(
        "我的Model Y刹车异响，VIN12345678901234567，上海，明天下午到店检查。",
        session_id="replan-1",
    )

    assert created["task_status"] == "awaiting_confirmation"
    assert created["error_log"]
    assert any("search_service_centers" in item for item in created["error_log"])
    assert any(step["step_id"].startswith("replan-search_service_centers-") for step in created["plan"])


def test_state_can_be_recovered_from_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TASK_AGENT_DB_PATH", str(tmp_path / "task_agent.db"))

    from task_agent.service import TaskService

    first = TaskService()
    created = first.start_task("我想买台电车", session_id="restore-1")
    first.continue_task(created["task_id"], "预算20万，家里有充电桩，通勤60公里，家用SUV")

    second = TaskService()
    restored = second.get_task_status(created["task_id"])

    assert restored["task_id"] == created["task_id"]
    assert restored["collected_info"]["budget"] == "20万"
    assert restored["task_status"] in {"awaiting_user_input", "awaiting_confirmation"}


def test_unknown_task_id_returns_structured_failure(service):
    status = service.get_task_status("task-missing")
    continued = service.continue_task("task-missing", "补充信息")
    confirmed = service.confirm_task_action("task-missing", confirmation_id="confirm-1", approved=True)

    for response in (status, continued, confirmed):
        assert response["task_id"] == "task-missing"
        assert response["task_status"] == "failed"
        assert response["result"]["error_code"] == "unknown_task"
        assert "Unknown task_id" in response["result"]["message"]


def test_unknown_confirmation_id_returns_structured_error_without_mutating_task(service):
    created = service.start_task("我想买台电车", session_id="bad-confirmation")
    pending = service.continue_task(
        created["task_id"],
        "预算25万，家里有充电桩，每天通勤60公里，家用SUV。我选小鹏G6，上海，明天下午试驾。",
    )

    response = service.confirm_task_action(pending["task_id"], confirmation_id="confirm-missing", approved=True)
    restored = service.get_task_status(pending["task_id"])

    assert response["task_status"] == "awaiting_confirmation"
    assert response["result"]["error_code"] == "unknown_confirmation"
    assert "confirm-missing" in response["result"]["message"]
    assert response["pending_confirmations"][0]["confirmation_id"] == "confirm-1"
    assert "recommendation" in response["result"]
    assert restored["task_status"] == "awaiting_confirmation"
    assert restored["pending_confirmations"][0]["confirmation_id"] == "confirm-1"
    assert "error_code" not in restored["result"]


def test_chitchat_does_not_depend_on_legacy_graph(service, monkeypatch: pytest.MonkeyPatch):
    fake_graph = types.ModuleType("cs_agent.graph")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("chitchat should not load legacy graph")

    fake_graph.chat = fail_if_called
    monkeypatch.setitem(sys.modules, "cs_agent.graph", fake_graph)

    result = service.start_task("\u4f60\u597d", session_id="chitchat-local")

    assert result["task_type"] == "chitchat"
    assert result["task_status"] == "completed"
    assert result["result"]["intent"] == "chitchat"
    assert result["result"]["answer"]


def test_faq_returns_fallback_when_legacy_graph_is_unavailable(service, monkeypatch: pytest.MonkeyPatch):
    fake_graph = types.ModuleType("cs_agent.graph")

    def unavailable(*args, **kwargs):
        raise TimeoutError("AsyncSqliteSaver init timed out after 30s")

    fake_graph.chat = unavailable
    monkeypatch.setitem(sys.modules, "cs_agent.graph", fake_graph)

    result = service.start_task("faq", session_id="faq-fallback")

    assert result["task_type"] == "faq"
    assert result["task_status"] == "completed"
    assert result["result"]["intent"] == "faq"
    assert result["result"]["sources"] == []
    assert result["error_log"]
    assert "legacy_agent" in result["error_log"][-1]


def test_faq_skips_legacy_graph_when_sqlite_checkpointer_is_missing(service, monkeypatch: pytest.MonkeyPatch):
    def fake_find_spec(name: str):
        if name == "langgraph.checkpoint.sqlite":
            return None
        return importlib.util.find_spec(name)

    fake_graph = types.ModuleType("cs_agent.graph")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("legacy graph should be skipped when sqlite checkpointer is missing")

    fake_graph.chat = fail_if_called
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setitem(sys.modules, "cs_agent.graph", fake_graph)

    result = service.start_task("faq", session_id="faq-no-sqlite")

    assert result["task_type"] == "faq"
    assert result["task_status"] == "completed"
    assert result["result"]["error_code"] == "legacy_unavailable"
    assert "langgraph.checkpoint.sqlite" in result["error_log"][-1]
