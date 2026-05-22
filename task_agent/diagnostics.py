from __future__ import annotations

import compileall
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any, Callable

from task_agent.evaluation import SCENARIOS, run_task_eval
from task_agent.models import TaskState
from task_agent.service import TaskService
from task_agent.storage import TaskStateStore
from task_agent.tools import REGISTRY


CheckFn = Callable[[], dict[str, Any]]


def run_diagnostics() -> dict[str, Any]:
    if os.getenv("TASK_AGENT_DB_PATH"):
        return _run_diagnostic_checks()

    previous_db_path = os.environ.get("TASK_AGENT_DB_PATH")
    with tempfile.TemporaryDirectory(prefix="task_agent_diag_") as temp_dir:
        os.environ["TASK_AGENT_DB_PATH"] = str(Path(temp_dir) / "task_agent.db")
        try:
            return _run_diagnostic_checks()
        finally:
            if previous_db_path is None:
                os.environ.pop("TASK_AGENT_DB_PATH", None)
            else:
                os.environ["TASK_AGENT_DB_PATH"] = previous_db_path


def run_launch_smoke() -> dict[str, Any]:
    if os.getenv("TASK_AGENT_DB_PATH"):
        return _run_launch_smoke_flows()

    previous_db_path = os.environ.get("TASK_AGENT_DB_PATH")
    with tempfile.TemporaryDirectory(prefix="task_agent_smoke_") as temp_dir:
        os.environ["TASK_AGENT_DB_PATH"] = str(Path(temp_dir) / "task_agent.db")
        try:
            return _run_launch_smoke_flows()
        finally:
            if previous_db_path is None:
                os.environ.pop("TASK_AGENT_DB_PATH", None)
            else:
                os.environ["TASK_AGENT_DB_PATH"] = previous_db_path


def run_launch_check() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {
        "tests": _check_pytest(),
        "health": run_diagnostics(),
        "smoke": run_launch_smoke(),
        "compatibility": _check_compatibility_routes(),
        "dependency_metadata": _check_dependency_metadata(),
        "compile": _check_compileall(),
        "package_build": _check_package_build(),
        "installed_package": _check_installed_package_smoke(),
        "static": _check_static_quality(),
        "default_db_clean": _check_default_db_clean(),
    }
    passed = sum(1 for check in checks.values() if check["ok"])
    total = len(checks)
    return {
        "ok": passed == total,
        "summary": {"passed": passed, "total": total},
        "checks": checks,
    }


def _check_compileall() -> dict[str, Any]:
    paths = ["task_agent", "cs_agent", "run.py"]
    results = []
    for path in paths:
        target = Path(path)
        if target.is_dir():
            ok = compileall.compile_dir(str(target), quiet=1)
        else:
            ok = compileall.compile_file(str(target), quiet=1)
        results.append({"path": path, "ok": bool(ok)})
    return {"ok": all(item["ok"] for item in results), "paths": results}


def _check_pytest() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "ok": completed.returncode == 0,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _check_package_build() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="task_agent_build_") as temp_dir:
        command = [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", temp_dir]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        wheels = sorted(Path(temp_dir).glob("*.whl"))
        required_files = _inspect_wheel_required_files(wheels[0]) if wheels else {}
        entry_points = _inspect_wheel_entry_points(wheels[0]) if wheels else {}
        entry_points_ok = entry_points.get("ev-task-agent") == "task_agent.main:main" and entry_points.get("ev-task-mcp") == "cs_agent.mcp_server:main"
        return {
            "ok": completed.returncode == 0 and bool(wheels) and all(required_files.values()) and entry_points_ok,
            "command": " ".join(command),
            "returncode": completed.returncode,
            "wheels": [wheel.name for wheel in wheels],
            "required_files": required_files,
            "entry_points": entry_points,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }


def _check_installed_package_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="task_agent_install_") as temp_dir:
        root = Path(temp_dir)
        wheel_dir = root / "wheel"
        wheel_dir.mkdir()
        build_command = [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheel_dir)]
        build_completed = subprocess.run(build_command, capture_output=True, text=True, check=False)
        wheels = sorted(wheel_dir.glob("*.whl"))
        if build_completed.returncode != 0 or not wheels:
            return {
                "ok": False,
                "stage": "build",
                "command": " ".join(build_command),
                "returncode": build_completed.returncode,
                "stdout": build_completed.stdout.strip(),
                "stderr": build_completed.stderr.strip(),
                "health_ok": False,
            }

        venv_dir = root / "venv"
        venv_command = [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)]
        venv_completed = subprocess.run(venv_command, capture_output=True, text=True, check=False)
        venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if venv_completed.returncode != 0 or not venv_python.exists():
            return {
                "ok": False,
                "stage": "venv",
                "command": " ".join(venv_command),
                "returncode": venv_completed.returncode,
                "stdout": venv_completed.stdout.strip(),
                "stderr": venv_completed.stderr.strip(),
                "health_ok": False,
            }

        install_command = [str(venv_python), "-m", "pip", "install", "--no-deps", str(wheels[0])]
        install_completed = subprocess.run(install_command, capture_output=True, text=True, check=False)
        if install_completed.returncode != 0:
            return {
                "ok": False,
                "stage": "install",
                "command": " ".join(install_command),
                "returncode": install_completed.returncode,
                "stdout": install_completed.stdout.strip(),
                "stderr": install_completed.stderr.strip(),
                "health_ok": False,
            }

        run_dir = root / "run"
        run_dir.mkdir()
        env = os.environ.copy()
        env["TASK_AGENT_DB_PATH"] = str(root / "installed_task_agent.db")
        health_command = [str(venv_python), "-m", "task_agent.main", "--health"]
        health_completed = subprocess.run(
            health_command,
            cwd=run_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        payload: dict[str, Any] = {}
        if health_completed.stdout.strip():
            try:
                payload = json.loads(health_completed.stdout)
            except json.JSONDecodeError:
                payload = {}
        health_ok = health_completed.returncode == 0 and payload.get("ok") is True
        script_path = venv_dir / ("Scripts/ev-task-agent.exe" if os.name == "nt" else "bin/ev-task-agent")
        script_command = [str(script_path), "--health"]
        script_completed = subprocess.run(
            script_command,
            cwd=run_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        script_payload: dict[str, Any] = {}
        if script_completed.stdout.strip():
            try:
                script_payload = json.loads(script_completed.stdout)
            except json.JSONDecodeError:
                script_payload = {}
        script_ok = script_completed.returncode == 0 and script_payload.get("ok") is True
        return {
            "ok": health_ok and script_ok,
            "stage": "health",
            "wheel": wheels[0].name,
            "command": " ".join(health_command),
            "returncode": health_completed.returncode,
            "health_ok": health_ok,
            "script_command": " ".join(script_command),
            "script_returncode": script_completed.returncode,
            "script_ok": script_ok,
            "tool_registry_ok": payload.get("checks", {}).get("tool_registry", {}).get("ok", False),
            "storage_roundtrip_ok": payload.get("checks", {}).get("storage_roundtrip", {}).get("ok", False),
            "stdout": health_completed.stdout.strip()[:1000],
            "stderr": health_completed.stderr.strip()[:1000],
            "script_stdout": script_completed.stdout.strip()[:1000],
            "script_stderr": script_completed.stderr.strip()[:1000],
        }


def _inspect_wheel_required_files(wheel_path: Path) -> dict[str, bool]:
    required = [
        "cs_agent/knowledge/vehicles.json",
        "cs_agent/knowledge/faq.json",
    ]
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
    return {path: path in names for path in required}


def _inspect_wheel_entry_points(wheel_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel_path) as wheel:
        entry_point_files = [name for name in wheel.namelist() if name.endswith(".dist-info/entry_points.txt")]
        if not entry_point_files:
            return {}
        content = wheel.read(entry_point_files[0]).decode("utf-8")
    scripts: dict[str, str] = {}
    in_console_scripts = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_console_scripts = line == "[console_scripts]"
            continue
        if in_console_scripts and "=" in line:
            name, target = line.split("=", 1)
            scripts[name.strip()] = target.strip()
    return scripts


def _check_static_quality() -> dict[str, Any]:
    command = [sys.executable, "-m", "ruff", "check", "task_agent", "tests/task_agent"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "ok": completed.returncode == 0,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _check_default_db_clean() -> dict[str, Any]:
    db_path = Path("cs_agent") / "knowledge" / "task_agent.db"
    exists = db_path.exists()
    return {"ok": not exists, "path": str(db_path), "exists": exists}


def _check_dependency_metadata() -> dict[str, Any]:
    pyproject_dependencies = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    requirements = [
        line.split("#", 1)[0].strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    ]
    requirements = [line for line in requirements if line]
    pyproject_specs = {_dependency_name(spec): spec for spec in pyproject_dependencies}
    requirement_specs = {_dependency_name(spec): spec for spec in requirements}
    pyproject_only = sorted(set(pyproject_specs) - set(requirement_specs))
    requirements_only = sorted(set(requirement_specs) - set(pyproject_specs))
    critical = {
        "langgraph-checkpoint-sqlite": {
            "in_pyproject": "langgraph-checkpoint-sqlite" in pyproject_specs,
            "in_requirements": "langgraph-checkpoint-sqlite" in requirement_specs,
            "pyproject_spec": pyproject_specs.get("langgraph-checkpoint-sqlite", ""),
            "requirements_spec": requirement_specs.get("langgraph-checkpoint-sqlite", ""),
        },
        "gradio": {
            "in_pyproject": "gradio" in pyproject_specs,
            "in_requirements": "gradio" in requirement_specs,
            "pyproject_spec": pyproject_specs.get("gradio", ""),
            "requirements_spec": requirement_specs.get("gradio", ""),
        },
    }
    critical_ok = (
        critical["langgraph-checkpoint-sqlite"]["in_pyproject"]
        and critical["langgraph-checkpoint-sqlite"]["in_requirements"]
        and "<6.0.0" in critical["gradio"]["pyproject_spec"]
        and "<6.0.0" in critical["gradio"]["requirements_spec"]
    )
    return {
        "ok": not pyproject_only and not requirements_only and critical_ok,
        "pyproject_only": pyproject_only,
        "requirements_only": requirements_only,
        "critical": critical,
    }


def _dependency_name(spec: str) -> str:
    for separator in ("<", ">", "=", "!", "~", ";", "["):
        if separator in spec:
            spec = spec.split(separator, 1)[0]
    return spec.strip().lower().replace("_", "-")


def _check_compatibility_routes() -> dict[str, Any]:
    if os.getenv("TASK_AGENT_DB_PATH"):
        return _run_check(_run_compatibility_routes)

    previous_db_path = os.environ.get("TASK_AGENT_DB_PATH")
    with tempfile.TemporaryDirectory(prefix="task_agent_compat_") as temp_dir:
        os.environ["TASK_AGENT_DB_PATH"] = str(Path(temp_dir) / "task_agent.db")
        try:
            return _run_check(_run_compatibility_routes)
        finally:
            if previous_db_path is None:
                os.environ.pop("TASK_AGENT_DB_PATH", None)
            else:
                os.environ["TASK_AGENT_DB_PATH"] = previous_db_path


def _run_compatibility_routes() -> dict[str, Any]:
    service = TaskService()
    chitchat = service.start_task("\u4f60\u597d", session_id="compat-chitchat")
    from cs_agent import mcp_server

    mcp_response = mcp_server._handle_request(
        {
            "jsonrpc": "2.0",
            "id": "compat-mcp",
            "method": "tools/call",
            "params": {"name": "ask_ev_agent", "arguments": {"query": "faq", "session_id": "compat-mcp"}},
        }
    )
    mcp_payload = {}
    if mcp_response and "result" in mcp_response:
        mcp_payload = json.loads(mcp_response["result"]["content"][0]["text"])
    routes = {
        "chitchat": {
            "ok": (
                chitchat.get("task_status") == "completed"
                and chitchat.get("result", {}).get("intent") == "chitchat"
                and bool(chitchat.get("result", {}).get("answer"))
            ),
            "task_status": chitchat.get("task_status", ""),
            "intent": chitchat.get("result", {}).get("intent", ""),
            "active_agent": chitchat.get("active_agent", ""),
        },
        "mcp_ask_ev_agent": {
            "ok": bool(mcp_payload.get("answer")) and "error" not in (mcp_response or {}),
            "intent": mcp_payload.get("intent", ""),
            "error_code": mcp_payload.get("error_code", ""),
            "backend": mcp_payload.get("backend", ""),
        },
    }
    return {"ok": all(route["ok"] for route in routes.values()), "routes": routes}


def _run_launch_smoke_flows() -> dict[str, Any]:
    service = TaskService()
    flows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        try:
            started = service.start_task(scenario.initial_query, session_id=f"smoke-{scenario.name}")
            progressed = service.continue_task(started["task_id"], scenario.follow_up) if scenario.follow_up else started
            final = progressed
            if progressed.get("pending_confirmations"):
                final = service.confirm_task_action(
                    progressed["task_id"],
                    progressed["pending_confirmations"][0]["confirmation_id"],
                    approved=True,
                )
            voucher_keys = [key for key in scenario.expect_final_keys if key in final.get("result", {})]
            ok = (
                progressed.get("task_status") == scenario.expect_status
                and scenario.expect_key in progressed.get("result", {})
                and final.get("task_status") == "completed"
                and len(voucher_keys) == len(scenario.expect_final_keys)
            )
            flows.append(
                {
                    "name": scenario.name,
                    "ok": ok,
                    "task_id": final.get("task_id", ""),
                    "task_type": final.get("task_type", ""),
                    "active_agent": final.get("active_agent", ""),
                    "status": progressed.get("task_status", ""),
                    "final_status": final.get("task_status", ""),
                    "voucher_keys": voucher_keys,
                    "result_keys": sorted(final.get("result", {}).keys()),
                }
            )
        except Exception as exc:
            flows.append(
                {
                    "name": scenario.name,
                    "ok": False,
                    "error": str(exc),
                    "final_status": "failed",
                    "voucher_keys": [],
                    "result_keys": [],
                }
            )
    passed = sum(1 for flow in flows if flow["ok"])
    total = len(flows)
    return {
        "ok": passed == total,
        "summary": {"passed": passed, "total": total},
        "flows": flows,
    }


def _run_diagnostic_checks() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {
        "tool_registry": _run_check(_check_tool_registry),
        "environment": _run_check(_check_environment),
        "storage_roundtrip": _run_check(_check_storage_roundtrip),
        "task_eval": _run_check(_check_task_eval),
    }
    passed = sum(1 for check in checks.values() if check["ok"])
    total = len(checks)
    return {
        "ok": passed == total,
        "summary": {"passed": passed, "total": total},
        "checks": checks,
    }


def _run_check(fn: CheckFn) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_tool_registry() -> dict[str, Any]:
    required_tools = {
        "search_vehicles",
        "get_vehicle_detail",
        "compare_vehicles",
        "search_service_centers",
        "search_charging_stations",
        "calculate_cost",
        "check_subsidy",
        "generate_comparison_report",
        "plan_route",
        "search_charging_stations_along_route",
        "generate_charging_plan",
        "estimate_trip_cost",
        "generate_trip_report",
        "estimate_repair_cost",
        "calculate_claim_impact",
        "file_insurance_claim",
        "assess_complaint_level",
        "create_complaint_ticket",
        "track_complaint",
        "book_test_drive",
        "create_service_ticket",
        "request_roadside_assistance",
        "book_service_appointment",
    }
    registered = {tool.name for tool in REGISTRY.list_tools()}
    missing = sorted(required_tools - registered)
    sample = REGISTRY.call("search_vehicles", {"budget_max": 260000, "need_suv": True})
    contract_ok = {"ok", "data", "error", "evidence", "requires_confirmation", "retryable"} <= set(sample)
    return {
        "ok": not missing and contract_ok and bool(sample["ok"]),
        "registered_count": len(registered),
        "missing": missing,
        "contract_ok": contract_ok,
    }


def _check_environment() -> dict[str, Any]:
    is_wsl = "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME"))
    coderabbit_path = shutil.which("coderabbit")
    pytest_importable = importlib.util.find_spec("pytest") is not None
    pytest_command = shutil.which("pytest") or ""
    pytest_available = pytest_importable or bool(pytest_command)
    return {
        "ok": bool(sys.executable) and pytest_available,
        "platform": platform.platform(),
        "is_wsl": is_wsl,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pytest_available": pytest_available,
        "pytest_importable": pytest_importable,
        "pytest_command": pytest_command,
        "coderabbit_path": coderabbit_path or "",
        "coderabbit_note": (
            "CodeRabbit CLI is optional for local health. On Windows, install and run it inside WSL; "
            "do not double-click the ELF binary from Explorer."
        ),
        "wsl_setup_hint": (
            "In WSL use bash commands: python3 -m venv .venv && source .venv/bin/activate && "
            "python -m pip install -e '.[dev]'. Use rm, not PowerShell Remove-Item."
        ),
        "windows_cleanup_hint": (
            "In PowerShell use Remove-Item -LiteralPath 'C:\\Users\\yanbo\\.local\\bin\\coderabbit' -Force. "
            "In WSL use rm -f /mnt/c/Users/yanbo/.local/bin/coderabbit."
        ),
    }


def _check_storage_roundtrip() -> dict[str, Any]:
    store = TaskStateStore()
    state = TaskState(goal="diagnostics storage roundtrip", task_type="faq", session_id="diagnostics")
    store.save(state)
    restored = store.load(state.task_id)
    return {
        "ok": restored.task_id == state.task_id and restored.goal == state.goal,
        "task_id": state.task_id,
        "db_path": str(store.db_path),
    }


def _check_task_eval() -> dict[str, Any]:
    metrics = run_task_eval()
    ok = (
        metrics.get("task_completion_rate") == 1.0
        and metrics.get("closed_loop_completion_rate") == 1.0
        and metrics.get("voucher_generation_rate") == 1.0
    )
    return {"ok": ok, "metrics": metrics}
