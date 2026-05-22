from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest


def test_read_only_tool_contracts_return_structured_result():
    from task_agent.tools import compare_vehicles, search_vehicles

    search_result = search_vehicles({"budget_max": 260000, "need_suv": True})
    assert search_result["ok"] is True
    assert isinstance(search_result["data"]["vehicles"], list)
    assert "evidence" in search_result
    assert search_result["requires_confirmation"] is False

    vehicles = search_result["data"]["vehicles"][:2]
    compare_result = compare_vehicles([vehicle["model"] for vehicle in vehicles])
    assert compare_result["ok"] is True
    assert compare_result["data"]["comparison"]
    assert compare_result["retryable"] is False


def test_search_vehicles_filters_suspicious_scraped_prices_from_top_recommendations():
    from task_agent.tools import search_vehicles

    result = search_vehicles(
        {
            "budget_max": 250000,
            "need_suv": True,
            "preferred_models": ["G6"],
            "use_case": "family",
            "charging_condition": "home_charger",
        }
    )

    models = [vehicle["model"] for vehicle in result["data"]["vehicles"][:3]]

    assert models[0] == "小鹏 G6"
    assert "理想 L8" not in models
    assert "理想 L9" not in models


def test_mobility_tools_return_verifiable_outputs():
    from task_agent.tools import calculate_cost, check_subsidy, generate_comparison_report, search_charging_stations

    stations = search_charging_stations({"city": "上海", "radius_km": 8})
    assert stations["ok"] is True
    assert stations["data"]["stations"]
    assert stations["data"]["stations"][0]["provider"] == "static-mobility-provider"

    cost = calculate_cost(
        {
            "model_id": "小鹏G6",
            "annual_km": 20000,
            "electricity_price": 0.62,
            "years": 5,
        }
    )
    assert cost["ok"] is True
    assert cost["data"]["years"] == 5
    assert cost["data"]["energy_cost_total"] > 0
    assert cost["data"]["tco_total"] > cost["data"]["energy_cost_total"]

    subsidy = check_subsidy({"model_id": "小鹏G6", "city": "上海"})
    assert subsidy["ok"] is True
    assert subsidy["data"]["city"] == "上海"
    assert "policy_items" in subsidy["data"]

    report = generate_comparison_report(
        {
            "model_ids": ["小鹏G6", "方程豹 豹5"],
            "recommendation": {"primary_model": "小鹏 G6", "reason": "test"},
            "ownership_cost": {"tco_total": 180000, "energy_cost_total": 8000},
            "subsidy": {"policy_items": ["local policy"]},
            "charging_stations": [{"name": "station-a"}],
        }
    )
    assert report["ok"] is True
    assert report["data"]["report_id"].startswith("CR-")
    assert report["data"]["recommended_model"]
    assert {"executive_summary", "vehicle_table", "cost_snapshot", "next_actions"} <= set(report["data"])


def test_tool_registry_exposes_mobility_tools():
    from task_agent.tools import REGISTRY

    tool_names = {tool.name for tool in REGISTRY.list_tools()}

    assert {"search_charging_stations", "calculate_cost", "check_subsidy", "generate_comparison_report"} <= tool_names
    assert {"plan_route", "search_charging_stations_along_route", "generate_charging_plan", "estimate_trip_cost"} <= tool_names
    assert {"estimate_repair_cost", "calculate_claim_impact", "file_insurance_claim"} <= tool_names
    assert {"assess_complaint_level", "create_complaint_ticket", "track_complaint"} <= tool_names


def test_charging_trip_tools_return_verifiable_itinerary():
    from task_agent.tools import estimate_trip_cost, generate_charging_plan, plan_route, search_charging_stations_along_route

    route = plan_route({"origin": "上海", "destination": "成都"})
    assert route["ok"] is True
    assert route["data"]["distance_km"] > 1000

    stations = search_charging_stations_along_route({"route": route["data"], "interval_km": 350})
    assert stations["ok"] is True
    assert len(stations["data"]["stations"]) >= 3

    plan = generate_charging_plan({"route": route["data"], "stations": stations["data"]["stations"], "vehicle_model": "Model Y"})
    assert plan["ok"] is True
    assert plan["data"]["stops"]
    assert plan["data"]["effective_range_km"] > 0

    cost = estimate_trip_cost({"route": route["data"], "charging_plan": plan["data"]})
    assert cost["ok"] is True
    assert cost["data"]["estimated_total_cost"] > cost["data"]["charging_cost"]


def test_insurance_tools_return_claim_decision_and_voucher():
    from task_agent.tools import calculate_claim_impact, estimate_repair_cost, file_insurance_claim

    damage = estimate_repair_cost({"damage_area": "右后门刮擦", "severity": "minor"})
    assert damage["ok"] is True
    assert damage["data"]["estimated_repair_cost"] > 0

    impact = calculate_claim_impact({"repair_cost": damage["data"]["estimated_repair_cost"], "no_claim_years": 2})
    assert impact["ok"] is True
    assert impact["data"]["recommendation"] in {"claim", "self_pay"}
    assert impact["data"]["premium_increase_estimate"] >= 0

    claim = file_insurance_claim(
        {
            "vin": "VIN12345678901234567",
            "city": "上海",
            "accident_time": "今天下午",
            "damage_area": "右后门刮擦",
            "repair_cost": damage["data"]["estimated_repair_cost"],
        }
    )
    assert claim["ok"] is True
    assert claim["requires_confirmation"] is True
    assert claim["data"]["claim_id"].startswith("IC-")


def test_complaint_tools_return_escalation_and_tracking_voucher():
    from task_agent.tools import assess_complaint_level, create_complaint_ticket, track_complaint

    assessment = assess_complaint_level({"issue": "车门异响", "repair_count": 3, "unresolved": True})
    assert assessment["ok"] is True
    assert assessment["data"]["level"] == "high"
    assert assessment["data"]["escalate"] is True

    ticket = create_complaint_ticket(
        {
            "vin": "VIN12345678901234567",
            "city": "上海",
            "issue": "车门异响",
            "repair_count": 3,
            "policy_basis": ["同一问题多次维修未解决"],
        }
    )
    assert ticket["ok"] is True
    assert ticket["requires_confirmation"] is True
    assert ticket["data"]["complaint_id"].startswith("CP-")

    tracked = track_complaint({"complaint_id": ticket["data"]["complaint_id"]})
    assert tracked["ok"] is True
    assert tracked["data"]["status"] in {"submitted", "in_review"}


def test_tool_registry_normalizes_partial_results_and_exceptions():
    from task_agent.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register("partial_tool", "read", False, lambda _payload: {"ok": True, "data": {"value": 1}})
    registry.register("broken_tool", "read", False, lambda _payload: (_ for _ in ()).throw(RuntimeError("boom")))
    registry.register("invalid_contract_tool", "read", False, lambda _payload: {"data": {"value": 1}})

    partial = registry.call("partial_tool", {})
    assert partial == {
        "ok": True,
        "data": {"value": 1},
        "error": "",
        "evidence": [],
        "requires_confirmation": False,
        "retryable": False,
    }

    broken = registry.call("broken_tool", {})
    assert broken["ok"] is False
    assert broken["error"] == "boom"
    assert broken["retryable"] is True

    unknown = registry.call("missing_tool", {})
    assert unknown["ok"] is False
    assert unknown["error"] == "Unknown tool: missing_tool"
    assert unknown["retryable"] is False

    invalid = registry.call("invalid_contract_tool", {})
    assert invalid["ok"] is False
    assert "invalid_contract_tool returned invalid result" in invalid["error"]
    assert invalid["retryable"] is True


def test_write_tool_contracts_require_confirmation():
    from task_agent.tools import (
        book_service_appointment,
        book_test_drive,
        create_complaint_ticket,
        create_service_ticket,
        file_insurance_claim,
        request_roadside_assistance,
    )

    ticket = create_service_ticket(
        {
            "vehicle_model": "Model Y",
            "issue_description": "刹车异响",
            "vin": "VIN12345678901234567",
        }
    )
    assert ticket["ok"] is True
    assert ticket["requires_confirmation"] is True

    booking = book_test_drive(
        {
            "vehicle_model": "小鹏G6",
            "city": "上海",
            "time_slot": "明天下午",
            "name": "测试用户",
        }
    )
    assert booking["ok"] is True
    assert booking["requires_confirmation"] is True

    appointment = book_service_appointment(
        {
            "vehicle_model": "Model Y",
            "city": "上海",
            "time_slot": "明天下午",
            "ticket_id": "SV-TEST",
        }
    )
    assert appointment["ok"] is True
    assert appointment["requires_confirmation"] is True

    roadside = request_roadside_assistance(
        {
            "vehicle_model": "Model Y",
            "city": "上海",
            "issue_type": "刹车失灵",
            "vin": "VIN12345678901234567",
            "service_center": "上海浦东新能源服务中心",
        }
    )
    assert roadside["ok"] is True
    assert roadside["requires_confirmation"] is True
    assert roadside["data"]["rescue_id"].startswith("RA-")
    assert roadside["data"]["status"] == "dispatched"

    claim = file_insurance_claim(
        {
            "vin": "VIN12345678901234567",
            "city": "上海",
            "accident_time": "今天下午",
            "damage_area": "右后门刮擦",
        }
    )
    assert claim["ok"] is True
    assert claim["requires_confirmation"] is True

    complaint = create_complaint_ticket(
        {
            "vin": "VIN12345678901234567",
            "city": "上海",
            "issue": "车门异响",
            "repair_count": 3,
        }
    )
    assert complaint["ok"] is True
    assert complaint["requires_confirmation"] is True


def test_mcp_server_exposes_new_task_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TASK_AGENT_DB_PATH", str(tmp_path / "task_agent.db"))

    from cs_agent import mcp_server

    tool_names = {tool["name"] for tool in mcp_server.TOOLS}
    assert {"start_task", "continue_task", "confirm_task_action", "get_task_status"} <= tool_names

    created = mcp_server._handle_start_task({"query": "我想买台电车", "session_id": "mcp-user"})
    assert created["task_id"]
    assert created["task_type"] == "purchase"

    status = mcp_server._handle_get_task_status({"task_id": created["task_id"]})
    assert status["task_id"] == created["task_id"]


def test_mcp_ask_ev_agent_returns_fallback_when_legacy_graph_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    from cs_agent import mcp_server

    fake_graph = types.ModuleType("cs_agent.graph")

    def unavailable(*args, **kwargs):
        raise TimeoutError("AsyncSqliteSaver init timed out after 30s")

    fake_graph.chat = unavailable
    monkeypatch.setitem(sys.modules, "cs_agent.graph", fake_graph)

    response = mcp_server._handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "ask_ev_agent", "arguments": {"query": "faq"}},
        }
    )

    assert response is not None
    assert "error" not in response
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["error_code"] == "legacy_unavailable"
    assert payload["intent"] == "faq"
    assert payload["answer"]


def test_task_entrypoint_is_available():
    import run

    assert hasattr(run, "_run_task")
