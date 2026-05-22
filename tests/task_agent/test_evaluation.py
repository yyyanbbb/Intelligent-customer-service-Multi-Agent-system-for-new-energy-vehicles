from __future__ import annotations

from pathlib import Path


def test_task_eval_measures_post_confirmation_closed_loop(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TASK_AGENT_DB_PATH", str(tmp_path / "task_eval.db"))

    from task_agent.evaluation import run_task_eval

    result = run_task_eval()

    assert result["task_completion_rate"] == 1.0
    assert result["closed_loop_completion_rate"] == 1.0
    assert result["voucher_generation_rate"] == 1.0
    assert result["n_scenarios"] == 6
    assert all(detail["passed"] for detail in result["details"])
    assert all(detail["final_status"] == "completed" for detail in result["details"])
    assert any("booking" in detail["final_result_keys"] for detail in result["details"])
    assert any("ticket" in detail["final_result_keys"] for detail in result["details"])
    assert any("roadside_assistance" in detail["final_result_keys"] for detail in result["details"])
    assert any("trip_report" in detail["final_result_keys"] for detail in result["details"])
    assert any("claim" in detail["final_result_keys"] for detail in result["details"])
    assert any("complaint" in detail["final_result_keys"] for detail in result["details"])
