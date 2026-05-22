from __future__ import annotations

import json
import sys
from pathlib import Path


def test_run_diagnostics_reports_core_health(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TASK_AGENT_DB_PATH", str(tmp_path / "diagnostics.db"))

    from task_agent.diagnostics import run_diagnostics

    report = run_diagnostics()

    assert report["ok"] is True
    assert report["summary"]["passed"] == report["summary"]["total"]
    assert report["checks"]["tool_registry"]["ok"] is True
    assert report["checks"]["environment"]["ok"] is True
    assert report["checks"]["environment"]["python_executable"]
    assert "wsl_setup_hint" in report["checks"]["environment"]
    assert report["checks"]["storage_roundtrip"]["ok"] is True
    assert report["checks"]["task_eval"]["ok"] is True
    assert report["checks"]["task_eval"]["metrics"]["closed_loop_completion_rate"] == 1.0


def test_run_diagnostics_without_db_env_does_not_create_default_db(monkeypatch):
    monkeypatch.delenv("TASK_AGENT_DB_PATH", raising=False)
    default_db = Path("cs_agent/knowledge/task_agent.db")
    if default_db.exists():
        default_db.unlink()

    from task_agent.diagnostics import run_diagnostics

    report = run_diagnostics()

    assert report["ok"] is True
    assert not default_db.exists()
    assert "task_agent_diag_" in report["checks"]["storage_roundtrip"]["db_path"]


def test_cli_health_prints_diagnostics_json(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("TASK_AGENT_DB_PATH", str(tmp_path / "cli_health.db"))
    monkeypatch.setattr(sys, "argv", ["task-agent", "--health"])

    from task_agent.main import main

    assert main() == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["ok"] is True
    assert "tool_registry" in payload["checks"]
    assert "environment" in payload["checks"]


def test_run_launch_smoke_covers_all_demo_flows_without_default_db(monkeypatch):
    monkeypatch.delenv("TASK_AGENT_DB_PATH", raising=False)
    default_db = Path("cs_agent/knowledge/task_agent.db")
    if default_db.exists():
        default_db.unlink()

    from task_agent.diagnostics import run_launch_smoke

    report = run_launch_smoke()

    assert report["ok"] is True
    assert report["summary"]["passed"] == report["summary"]["total"] == 6
    assert not default_db.exists()
    flow_names = {flow["name"] for flow in report["flows"]}
    assert {
        "purchase_flow",
        "aftersales_flow",
        "roadside_assistance_flow",
        "charging_trip_flow",
        "insurance_claim_flow",
        "complaint_escalation_flow",
    } <= flow_names
    assert all(flow["final_status"] == "completed" for flow in report["flows"])
    assert all(flow["voucher_keys"] for flow in report["flows"])


def test_cli_smoke_prints_launch_smoke_json(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("TASK_AGENT_DB_PATH", str(tmp_path / "cli_smoke.db"))
    monkeypatch.setattr(sys, "argv", ["task-agent", "--smoke"])

    from task_agent.main import main

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["summary"]["total"] == 6
    assert "flows" in payload


def test_run_launch_check_combines_health_smoke_compile_and_db_guard(monkeypatch):
    monkeypatch.delenv("TASK_AGENT_DB_PATH", raising=False)
    default_db = Path("cs_agent/knowledge/task_agent.db")
    if default_db.exists():
        default_db.unlink()

    from task_agent import diagnostics

    monkeypatch.setattr(diagnostics, "_check_pytest", lambda: {"ok": True, "command": "pytest -q", "returncode": 0})

    report = diagnostics.run_launch_check()

    assert report["ok"] is True
    assert report["checks"]["tests"]["ok"] is True
    assert report["checks"]["health"]["ok"] is True
    assert report["checks"]["smoke"]["ok"] is True
    assert report["checks"]["compatibility"]["ok"] is True
    assert report["checks"]["dependency_metadata"]["ok"] is True
    assert report["checks"]["compile"]["ok"] is True
    assert report["checks"]["package_build"]["ok"] is True
    assert report["checks"]["package_build"]["required_files"]["cs_agent/knowledge/vehicles.json"] is True
    assert report["checks"]["package_build"]["required_files"]["cs_agent/knowledge/faq.json"] is True
    assert report["checks"]["package_build"]["entry_points"]["ev-task-agent"] == "task_agent.main:main"
    assert report["checks"]["package_build"]["entry_points"]["ev-task-mcp"] == "cs_agent.mcp_server:main"
    assert report["checks"]["installed_package"]["ok"] is True
    assert report["checks"]["installed_package"]["health_ok"] is True
    assert report["checks"]["installed_package"]["script_ok"] is True
    assert report["checks"]["static"]["ok"] is True
    assert report["checks"]["default_db_clean"]["ok"] is True
    assert report["summary"]["passed"] == report["summary"]["total"] == 10
    assert not default_db.exists()


def test_cli_launch_check_prints_combined_gate_json(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("TASK_AGENT_DB_PATH", str(tmp_path / "cli_launch_check.db"))
    monkeypatch.setattr(sys, "argv", ["task-agent", "--launch-check"])

    from task_agent import diagnostics
    from task_agent.main import main

    monkeypatch.setattr(diagnostics, "_check_pytest", lambda: {"ok": True, "command": "pytest -q", "returncode": 0})

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert {
        "tests",
        "health",
        "smoke",
        "compatibility",
        "dependency_metadata",
        "compile",
        "package_build",
        "installed_package",
        "static",
        "default_db_clean",
    } <= set(payload["checks"])


def test_compatibility_check_covers_chitchat_route_without_default_db(monkeypatch):
    monkeypatch.delenv("TASK_AGENT_DB_PATH", raising=False)
    default_db = Path("cs_agent/knowledge/task_agent.db")
    if default_db.exists():
        default_db.unlink()

    from task_agent import diagnostics

    report = diagnostics._check_compatibility_routes()

    assert report["ok"] is True
    assert report["routes"]["chitchat"]["task_status"] == "completed"
    assert report["routes"]["chitchat"]["intent"] == "chitchat"
    assert report["routes"]["mcp_ask_ev_agent"]["ok"] is True
    assert report["routes"]["mcp_ask_ev_agent"]["error_code"] == "legacy_unavailable"
    assert not default_db.exists()


def test_dependency_metadata_check_requires_synced_runtime_dependency_files():
    from task_agent import diagnostics

    report = diagnostics._check_dependency_metadata()

    assert report["ok"] is True
    assert report["pyproject_only"] == []
    assert report["requirements_only"] == []
    assert report["critical"]["langgraph-checkpoint-sqlite"]["in_pyproject"] is True
    assert report["critical"]["langgraph-checkpoint-sqlite"]["in_requirements"] is True
    assert "<6.0.0" in report["critical"]["gradio"]["pyproject_spec"]


def test_cli_launch_check_returns_nonzero_when_gate_fails(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["task-agent", "--launch-check"])

    from task_agent import diagnostics
    from task_agent.main import main

    monkeypatch.setattr(
        diagnostics,
        "run_launch_check",
        lambda: {
            "ok": False,
            "summary": {"passed": 5, "total": 6},
            "checks": {"tests": {"ok": False, "returncode": 1}},
        },
    )

    assert main() == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["summary"]["passed"] == 5


def test_cli_diagnostic_commands_return_nonzero_when_report_fails(monkeypatch, capsys):
    from task_agent import diagnostics
    from task_agent.main import main

    for flag, function_name in [("--health", "run_diagnostics"), ("--smoke", "run_launch_smoke")]:
        monkeypatch.setattr(sys, "argv", ["task-agent", flag])
        monkeypatch.setattr(diagnostics, function_name, lambda: {"ok": False, "checks": {}})

        assert main() == 1

        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False


def test_environment_check_accepts_pytest_command_when_import_is_missing(monkeypatch):
    from task_agent import diagnostics

    monkeypatch.setattr(diagnostics.importlib.util, "find_spec", lambda name: None if name == "pytest" else object())
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: "pytest" if name == "pytest" else "")

    result = diagnostics._check_environment()

    assert result["ok"] is True
    assert result["pytest_available"] is True
    assert result["pytest_importable"] is False
    assert result["pytest_command"] == "pytest"
