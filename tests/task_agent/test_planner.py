from __future__ import annotations

import json

from task_agent.models import TaskState
from task_agent.planner import PlannerAgent


def test_planner_uses_structured_generator_output():
    def generator(_state: TaskState) -> str:
        return json.dumps(
            {
                "steps": [
                    {
                        "step_id": "custom-collect",
                        "owner_agent": "purchase_agent",
                        "kind": "ask_user",
                        "title": "Collect purchase constraints",
                        "success_criteria": "Budget and charging constraints are known",
                    },
                    {
                        "step_id": "custom-confirm",
                        "owner_agent": "purchase_agent",
                        "kind": "confirm",
                        "title": "Confirm booking",
                        "tool_name": "book_test_drive",
                        "success_criteria": "User approves the write action",
                    },
                ]
            }
        )

    state = TaskState(goal="buy an EV", task_type="purchase")
    plan = PlannerAgent(plan_generator=generator).build_plan(state)

    assert [step.step_id for step in plan] == ["custom-collect", "custom-confirm"]
    assert plan[1].tool_name == "book_test_drive"


def test_planner_falls_back_when_structured_output_is_invalid():
    state = TaskState(goal="buy an EV", task_type="purchase")
    plan = PlannerAgent(plan_generator=lambda _state: "not json").build_plan(state)

    assert plan[0].step_id == "purchase-collect"
    assert any("planner:" in item for item in state.error_log)


def test_default_rule_planner_does_not_log_fallback_as_error(monkeypatch):
    monkeypatch.delenv("TASK_AGENT_PLANNER_MODE", raising=False)
    state = TaskState(goal="buy an EV", task_type="purchase")

    plan = PlannerAgent().build_plan(state)

    assert plan[0].step_id == "purchase-collect"
    assert not state.error_log


def test_replan_inserts_retry_step_after_current_step():
    state = TaskState(goal="service my car", task_type="aftersales")
    planner = PlannerAgent()
    state.plan = planner.build_plan(state)
    state.current_step = 2

    planner.replan_after_failure(state, failed_tool="search_service_centers", reason="provider timeout")

    inserted = state.plan[3]
    assert inserted.step_id.startswith("replan-search_service_centers-")
    assert inserted.kind == "call_tool"
    assert inserted.status == "pending"
    assert "provider timeout" in inserted.success_criteria
    assert any("replan: search_service_centers" in item for item in state.error_log)
