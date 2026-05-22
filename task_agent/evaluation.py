from __future__ import annotations

from dataclasses import dataclass

from task_agent.service import TaskService


@dataclass
class EvalScenario:
    name: str
    initial_query: str
    follow_up: str
    expect_status: str
    expect_key: str
    expect_final_keys: tuple[str, ...]


SCENARIOS = [
    EvalScenario(
        name="purchase_flow",
        initial_query="我想买台电车",
        follow_up="预算25万，家里有充电桩，每天通勤60公里，家用SUV。我选小鹏G6，上海，明天下午试驾。",
        expect_status="awaiting_confirmation",
        expect_key="recommendation",
        expect_final_keys=("booking",),
    ),
    EvalScenario(
        name="aftersales_flow",
        initial_query="我的Model Y昨天开始刹车异响，帮我处理一下",
        follow_up="VIN12345678901234567，上海，明天下午到店。我不需要道路救援，直接预约维修。",
        expect_status="awaiting_confirmation",
        expect_key="diagnosis",
        expect_final_keys=("ticket", "appointment"),
    ),
    EvalScenario(
        name="roadside_assistance_flow",
        initial_query="我的Model Y刹车失灵，需要道路救援拖到服务中心",
        follow_up="VIN12345678901234567，上海，明天下午处理，需要道路救援。",
        expect_status="awaiting_confirmation",
        expect_key="diagnosis",
        expect_final_keys=("ticket", "roadside_assistance"),
    ),
    EvalScenario(
        name="charging_trip_flow",
        initial_query="下周要从上海开到成都，Model Y 长续航，帮我规划充电方案",
        follow_up="",
        expect_status="completed",
        expect_key="trip_report",
        expect_final_keys=("route", "charging_plan", "trip_cost", "trip_report"),
    ),
    EvalScenario(
        name="insurance_claim_flow",
        initial_query="今天倒车时刮了右后门，帮我走保险",
        follow_up="VIN12345678901234567，上海，今天下午，单方事故，没有人员伤亡，右后门刮擦，走保险。",
        expect_status="awaiting_confirmation",
        expect_key="claim_impact",
        expect_final_keys=("damage_estimate", "claim_impact", "claim"),
    ),
    EvalScenario(
        name="complaint_escalation_flow",
        initial_query="车门异响修了三次还没好，我要投诉",
        follow_up="VIN12345678901234567，上海，车门异响，已经去4S店维修3次，工单SV001、SV002、SV003，仍未解决，我要升级投诉。",
        expect_status="awaiting_confirmation",
        expect_key="complaint_assessment",
        expect_final_keys=("complaint_assessment", "policy_basis", "complaint"),
    ),
]


def run_task_eval() -> dict:
    service = TaskService()
    total = len(SCENARIOS)
    success = 0
    closed_loop_success = 0
    voucher_success = 0
    details = []
    for scenario in SCENARIOS:
        started = service.start_task(scenario.initial_query, session_id=f"eval-{scenario.name}")
        progressed = service.continue_task(started["task_id"], scenario.follow_up) if scenario.follow_up else started
        passed = progressed["task_status"] == scenario.expect_status and scenario.expect_key in progressed["result"]
        final = progressed
        if progressed.get("pending_confirmations"):
            final = service.confirm_task_action(
                progressed["task_id"],
                progressed["pending_confirmations"][0]["confirmation_id"],
                approved=True,
            )
        final_result = final["result"]
        final_keys = sorted(final_result.keys())
        final_passed = final["task_status"] == "completed"
        voucher_passed = all(key in final_result for key in scenario.expect_final_keys)
        success += int(passed)
        closed_loop_success += int(final_passed)
        voucher_success += int(voucher_passed)
        details.append(
            {
                "name": scenario.name,
                "passed": passed,
                "status": progressed["task_status"],
                "result_keys": sorted(progressed["result"].keys()),
                "final_status": final["task_status"],
                "final_result_keys": final_keys,
                "closed_loop_passed": final_passed,
                "voucher_passed": voucher_passed,
            }
        )
    return {
        "task_completion_rate": success / total if total else 0.0,
        "closed_loop_completion_rate": closed_loop_success / total if total else 0.0,
        "voucher_generation_rate": voucher_success / total if total else 0.0,
        "n_scenarios": total,
        "details": details,
    }
