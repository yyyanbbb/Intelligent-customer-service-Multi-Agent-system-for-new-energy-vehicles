from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pydantic import ValidationError

from cs_agent.tools.ticket_tool import create_ticket as legacy_create_ticket
from task_agent.models import ToolResult
from task_agent.providers import get_default_mobility_provider


ToolFn = Callable[[dict[str, Any]], dict[str, Any]]

_KB_DIR = Path(__file__).resolve().parent.parent / "cs_agent" / "knowledge"
_PREMIUM_BRANDS_WITH_SUSPICIOUS_LOW_PRICE = {"理想", "蔚来", "问界", "腾势"}


def _price_to_int(price_text: str) -> int:
    match = re.search(r"(\d+(?:\.\d+)?)", price_text or "")
    if not match:
        return 0
    return int(float(match.group(1)) * 10000)


def _load_vehicles() -> list[dict[str, Any]]:
    return json.loads((_KB_DIR / "vehicles.json").read_text(encoding="utf-8"))


def _load_faq() -> list[dict[str, Any]]:
    return json.loads((_KB_DIR / "faq.json").read_text(encoding="utf-8"))


def _has_suspicious_scraped_price(vehicle: dict[str, Any], price_value: int) -> bool:
    brand = vehicle.get("brand", "")
    if brand in _PREMIUM_BRANDS_WITH_SUSPICIOUS_LOW_PRICE and 0 < price_value < 150000:
        return True
    return False


def _tool_result(
    *,
    ok: bool,
    data: dict[str, Any] | None = None,
    error: str = "",
    evidence: list[str] | None = None,
    requires_confirmation: bool = False,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "data": data or {},
        "error": error,
        "evidence": evidence or [],
        "requires_confirmation": requires_confirmation,
        "retryable": retryable,
    }


@dataclass
class ToolDefinition:
    name: str
    kind: str
    requires_confirmation: bool
    fn: ToolFn


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, kind: str, requires_confirmation: bool, fn: ToolFn) -> None:
        self._tools[name] = ToolDefinition(name=name, kind=kind, requires_confirmation=requires_confirmation, fn=fn)

    def call(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        definition = self._tools.get(name)
        if definition is None:
            return _tool_result(ok=False, error=f"Unknown tool: {name}", evidence=[name], retryable=False)
        try:
            current = globals().get(name)
            raw_result = current(payload) if callable(current) else definition.fn(payload)
        except Exception as exc:
            return _tool_result(ok=False, error=str(exc), evidence=[name], retryable=True)
        try:
            return self._normalize_result(raw_result, definition)
        except ValidationError as exc:
            return _tool_result(
                ok=False,
                error=f"{name} returned invalid result: {exc.errors()}",
                evidence=[name],
                retryable=True,
            )

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def _normalize_result(self, raw_result: dict[str, Any], definition: ToolDefinition) -> dict[str, Any]:
        result = ToolResult.model_validate(raw_result).model_dump()
        if definition.requires_confirmation:
            result["requires_confirmation"] = True
        return result


def search_vehicles(filters: dict[str, Any]) -> dict[str, Any]:
    budget_max = int(filters.get("budget_max", 10**9))
    need_suv = bool(filters.get("need_suv"))
    preferred_models = set(filters.get("preferred_models", []))
    use_case = filters.get("use_case", "")
    charging_condition = filters.get("charging_condition", "")

    ranked: list[tuple[int, dict[str, Any]]] = []
    for vehicle in _load_vehicles():
        price_value = _price_to_int(vehicle.get("price", ""))
        if _has_suspicious_scraped_price(vehicle, price_value):
            continue
        if price_value and price_value > budget_max:
            continue
        summary = vehicle.get("summary", "")
        score = 0
        if preferred_models and any(model.lower() in vehicle.get("model", "").lower() for model in preferred_models):
            score += 6
        if need_suv and "SUV" in summary.upper():
            score += 3
        if use_case == "family" and any(keyword in summary for keyword in ("SUV", "大空间", "家用")):
            score += 2
        if charging_condition == "home_charger":
            score += 1
        range_km = int(vehicle.get("specs", {}).get("CLTC续航km", 0) or 0)
        score += min(range_km // 100, 6)
        ranked.append(
            (
                score,
                {
                    "brand": vehicle.get("brand", ""),
                    "model": vehicle.get("model", ""),
                    "price": vehicle.get("price", ""),
                    "price_value": price_value,
                    "range_km": range_km,
                    "source_url": vehicle.get("source_url", ""),
                    "summary": summary[:200],
                },
            )
        )

    ranked.sort(key=lambda item: (item[0], item[1]["range_km"], -item[1]["price_value"]), reverse=True)
    vehicles = [item[1] for item in ranked[:5]]
    return _tool_result(
        ok=bool(vehicles),
        data={"vehicles": vehicles},
        error="" if vehicles else "没有找到符合条件的车型",
        evidence=[vehicle["model"] for vehicle in vehicles],
        requires_confirmation=False,
        retryable=False,
    )


def get_vehicle_detail(payload: dict[str, Any]) -> dict[str, Any]:
    model_name = payload.get("model_id") or payload.get("vehicle_model") or payload.get("model", "")
    if not model_name:
        return _tool_result(ok=False, error="缺少 model_id", retryable=False)
    normalized_model_name = re.sub(r"\s+", "", model_name).lower()

    for vehicle in _load_vehicles():
        model = vehicle.get("model", "")
        normalized_model = re.sub(r"\s+", "", model).lower()
        if normalized_model_name in normalized_model or normalized_model in normalized_model_name:
            specs = vehicle.get("specs", {})
            detail = {
                "brand": vehicle.get("brand", ""),
                "model": model,
                "price": vehicle.get("price", ""),
                "range_km": int(specs.get("CLTC续航km", 0) or 0),
                "battery_kwh": int(specs.get("电池容量kWh", 0) or 0),
                "wheelbase_mm": int(specs.get("轴距mm", 0) or 0),
                "power_kw": int(specs.get("最大功率kW", 0) or 0),
                "source_url": vehicle.get("source_url", ""),
                "summary": vehicle.get("summary", "")[:240],
            }
            return _tool_result(ok=True, data=detail, evidence=[detail["source_url"]], retryable=False)
    return _tool_result(ok=False, error=f"未找到车型: {model_name}", retryable=False)


def compare_vehicles(model_ids: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        detail = get_vehicle_detail({"model_id": model_id})
        if detail["ok"]:
            rows.append(detail["data"])
    comparison = []
    for row in rows:
        comparison.append(
            {
                "model": row["model"],
                "price": row["price"],
                "range_km": row["range_km"],
                "battery_kwh": row["battery_kwh"],
                "power_kw": row["power_kw"],
            }
        )
    return _tool_result(
        ok=bool(comparison),
        data={"comparison": comparison},
        error="" if comparison else "无法生成对比结果",
        evidence=[row["model"] for row in comparison],
        requires_confirmation=False,
        retryable=False,
    )


def _estimate_efficiency_kwh_per_100km(detail: dict[str, Any]) -> float:
    battery_kwh = float(detail.get("battery_kwh", 0) or 0)
    range_km = float(detail.get("range_km", 0) or 0)
    if battery_kwh > 0 and range_km > 0:
        return round(battery_kwh / range_km * 100, 2)
    return 15.5


def search_charging_stations(payload: dict[str, Any]) -> dict[str, Any]:
    city = payload.get("city", "")
    radius_km = int(payload.get("radius_km", 10))
    provider = get_default_mobility_provider()
    stations = provider.search_charging_stations(city, radius_km)
    return _tool_result(
        ok=bool(stations),
        data={"city": city, "radius_km": radius_km, "stations": stations},
        error="" if stations else f"未找到 {city} {radius_km}km 内的充电站",
        evidence=[station["name"] for station in stations],
        retryable=False,
    )


def calculate_cost(payload: dict[str, Any]) -> dict[str, Any]:
    model_id = payload.get("model_id") or payload.get("vehicle_model") or payload.get("model", "")
    annual_km = int(payload.get("annual_km", 15000))
    electricity_price = float(payload.get("electricity_price", 0.68))
    years = int(payload.get("years", 5))
    detail_result = get_vehicle_detail({"model_id": model_id})
    if not detail_result["ok"]:
        return detail_result

    detail = detail_result["data"]
    efficiency = _estimate_efficiency_kwh_per_100km(detail)
    energy_cost_year = round(annual_km / 100 * efficiency * electricity_price, 2)
    maintenance_cost_year = 1200
    insurance_cost_year = 4500
    energy_cost_total = round(energy_cost_year * years, 2)
    recurring_cost_total = round((energy_cost_year + maintenance_cost_year + insurance_cost_year) * years, 2)
    price_value = _price_to_int(detail.get("price", ""))
    tco_total = round(price_value + recurring_cost_total, 2)
    return _tool_result(
        ok=True,
        data={
            "model_id": detail["model"],
            "years": years,
            "annual_km": annual_km,
            "electricity_price": electricity_price,
            "efficiency_kwh_per_100km": efficiency,
            "energy_cost_year": energy_cost_year,
            "energy_cost_total": energy_cost_total,
            "maintenance_cost_total": maintenance_cost_year * years,
            "insurance_cost_total": insurance_cost_year * years,
            "vehicle_price": price_value,
            "tco_total": tco_total,
        },
        evidence=[detail["model"], detail.get("source_url", "")],
        retryable=False,
    )


def check_subsidy(payload: dict[str, Any]) -> dict[str, Any]:
    city = payload.get("city", "")
    model_id = payload.get("model_id") or payload.get("vehicle_model") or payload.get("model", "")
    provider = get_default_mobility_provider()
    policy = provider.subsidy_policy(city, model_id)
    return _tool_result(
        ok=True,
        data=policy,
        evidence=policy["policy_items"],
        retryable=False,
    )


def generate_comparison_report(payload: dict[str, Any]) -> dict[str, Any]:
    model_ids = [model for model in payload.get("model_ids", []) if model]
    recommendation = payload.get("recommendation", {})
    ownership_cost = payload.get("ownership_cost", {})
    subsidy = payload.get("subsidy", {})
    charging_stations = payload.get("charging_stations", [])

    comparison_result = compare_vehicles(model_ids)
    vehicle_table = comparison_result["data"].get("comparison", []) if comparison_result["ok"] else []
    recommended_model = recommendation.get("primary_model") or (vehicle_table[0]["model"] if vehicle_table else "")
    if not recommended_model:
        return _tool_result(ok=False, error="缺少可生成报告的候选车型", retryable=False)

    report = {
        "report_id": f"CR-{uuid4().hex[:8].upper()}",
        "recommended_model": recommended_model,
        "executive_summary": f"推荐优先考虑 {recommended_model}，已结合候选车对比、5 年用车成本、地方补贴和补能资源生成报告。",
        "vehicle_table": vehicle_table,
        "cost_snapshot": {
            "years": ownership_cost.get("years", 5),
            "tco_total": ownership_cost.get("tco_total", 0),
            "energy_cost_total": ownership_cost.get("energy_cost_total", 0),
            "efficiency_kwh_per_100km": ownership_cost.get("efficiency_kwh_per_100km", 0),
        },
        "policy_snapshot": {
            "city": subsidy.get("city", ""),
            "policy_items": subsidy.get("policy_items", []),
        },
        "charging_snapshot": [
            {
                "name": station.get("name", ""),
                "address": station.get("address", ""),
                "fast_chargers": station.get("fast_chargers", 0),
            }
            for station in charging_stations[:3]
        ],
        "next_actions": [
            "确认目标车型",
            "预约试驾",
            "带着报告核对门店报价、权益和保险方案",
        ],
    }
    return _tool_result(
        ok=True,
        data=report,
        evidence=[report["report_id"], *[row["model"] for row in vehicle_table[:3]]],
        retryable=False,
    )


_ROUTES = {
    ("上海", "成都"): {
        "origin": "上海",
        "destination": "成都",
        "distance_km": 1960,
        "highway_fee": 860,
        "waypoints": ["湖州", "合肥", "武汉", "宜昌", "重庆", "成都"],
    },
    ("北京", "上海"): {
        "origin": "北京",
        "destination": "上海",
        "distance_km": 1210,
        "highway_fee": 540,
        "waypoints": ["济南", "徐州", "南京", "上海"],
    },
    ("深圳", "杭州"): {
        "origin": "深圳",
        "destination": "杭州",
        "distance_km": 1250,
        "highway_fee": 590,
        "waypoints": ["惠州", "赣州", "南昌", "衢州", "杭州"],
    },
}


def plan_route(payload: dict[str, Any]) -> dict[str, Any]:
    origin = payload.get("origin", "")
    destination = payload.get("destination", "")
    route = _ROUTES.get((origin, destination))
    if route is None:
        route = {
            "origin": origin,
            "destination": destination,
            "distance_km": 900,
            "highway_fee": 420,
            "waypoints": [origin, destination],
        }
    return _tool_result(ok=bool(origin and destination), data=route, evidence=[f"{origin}->{destination}"], retryable=False)


def search_charging_stations_along_route(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload.get("route", {})
    interval_km = int(payload.get("interval_km", 350))
    waypoints = route.get("waypoints", [])
    stations = []
    distance_km = int(route.get("distance_km", 0))
    needed_stops = max(1, distance_km // max(interval_km, 1))
    for index in range(needed_stops):
        city = waypoints[min(index + 1, len(waypoints) - 1)] if waypoints else route.get("destination", "")
        stations.append(
            {
                "station_id": f"RS-{index + 1:02d}",
                "city": city,
                "name": f"{city}高速超充站",
                "distance_from_origin_km": min((index + 1) * interval_km, distance_km),
                "fast_chargers": 8,
                "rating": 4.6,
                "amenities": ["卫生间", "餐饮", "休息区"],
            }
        )
    return _tool_result(ok=bool(stations), data={"stations": stations, "interval_km": interval_km}, evidence=[s["name"] for s in stations], retryable=False)


def _effective_trip_range_km(model: str) -> int:
    detail = get_vehicle_detail({"model_id": model})
    raw_range = int(detail.get("data", {}).get("range_km", 0) or 0) if detail.get("ok") else 0
    if raw_range <= 0:
        raw_range = 688 if "Model Y" in model else 600
    return int(raw_range * 0.7 * 0.85)


def generate_charging_plan(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload.get("route", {})
    stations = payload.get("stations", [])
    vehicle_model = payload.get("vehicle_model", "")
    effective_range = _effective_trip_range_km(vehicle_model)
    distance_km = int(route.get("distance_km", 0))
    stops = []
    previous_km = 0
    for station in stations:
        current_km = int(station.get("distance_from_origin_km", 0))
        segment_km = max(current_km - previous_km, 0)
        stops.append(
            {
                "station_id": station.get("station_id", ""),
                "station_name": station.get("name", ""),
                "city": station.get("city", ""),
                "segment_km": segment_km,
                "arrive_soc_percent": 18,
                "charge_to_percent": 82,
                "charge_minutes": 28,
                "estimated_fee": 96,
            }
        )
        previous_km = current_km
    return _tool_result(
        ok=bool(route and stops),
        data={
            "vehicle_model": vehicle_model,
            "effective_range_km": effective_range,
            "distance_km": distance_km,
            "stops": stops,
            "driving_strategy": "按高速 7 折续航并保留 15% 安全电量规划，单段尽量控制在有效续航内。",
        },
        evidence=[stop["station_name"] for stop in stops],
        retryable=False,
    )


def estimate_trip_cost(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload.get("route", {})
    charging_plan = payload.get("charging_plan", {})
    stops = charging_plan.get("stops", [])
    charging_cost = sum(float(stop.get("estimated_fee", 0)) for stop in stops)
    highway_fee = float(route.get("highway_fee", 0))
    total_minutes = round(float(route.get("distance_km", 0)) / 90 * 60 + sum(float(stop.get("charge_minutes", 0)) for stop in stops), 1)
    total_cost = round(charging_cost + highway_fee, 2)
    return _tool_result(
        ok=True,
        data={
            "charging_cost": round(charging_cost, 2),
            "highway_fee": highway_fee,
            "estimated_total_cost": total_cost,
            "estimated_total_hours": round(total_minutes / 60, 1),
        },
        evidence=[f"{len(stops)} charging stops", f"{route.get('distance_km', 0)} km"],
        retryable=False,
    )


def generate_trip_report(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload.get("route", {})
    charging_plan = payload.get("charging_plan", {})
    trip_cost = payload.get("trip_cost", {})
    report = {
        "report_id": f"TP-{uuid4().hex[:8].upper()}",
        "title": f"{route.get('origin', '')} 到 {route.get('destination', '')} 充电行程单",
        "route_summary": route,
        "charging_stops": charging_plan.get("stops", []),
        "cost_summary": trip_cost,
        "risk_notes": ["节假日建议提前 30-60 分钟预留排队时间", "山区和雨雪天气按续航再打 8 折预留"],
    }
    return _tool_result(ok=True, data=report, evidence=[report["report_id"]], retryable=False)


def estimate_repair_cost(payload: dict[str, Any]) -> dict[str, Any]:
    damage_area = payload.get("damage_area", "")
    severity = payload.get("severity", "minor")
    base_cost = 2200
    if any(keyword in damage_area for keyword in ("保险杠", "前杠", "后杠")):
        base_cost = 1800
    if "轮毂" in damage_area:
        base_cost = 1200
    if any(keyword in damage_area for keyword in ("右后门", "左后门", "车门")):
        base_cost = 2400
    multiplier = {"minor": 1.0, "medium": 1.6, "severe": 2.5}.get(severity, 1.0)
    estimated = int(base_cost * multiplier)
    return _tool_result(
        ok=bool(damage_area),
        data={
            "damage_area": damage_area,
            "severity": severity,
            "estimated_repair_cost": estimated,
            "repair_items": ["钣金修复", "喷漆", "外观校准"],
            "evidence_required": ["事故远景照片", "受损近景照片", "VIN/行驶证照片", "仪表盘里程照片"],
        },
        error="" if damage_area else "缺少受损位置",
        evidence=[damage_area],
        retryable=False,
    )


def calculate_claim_impact(payload: dict[str, Any]) -> dict[str, Any]:
    repair_cost = float(payload.get("repair_cost", 0))
    no_claim_years = int(payload.get("no_claim_years", 1))
    premium_increase = max(600, int(1200 - min(no_claim_years, 3) * 150))
    deductible = float(payload.get("deductible", 0))
    out_of_pocket_if_claim = deductible
    net_claim_benefit = repair_cost - premium_increase - out_of_pocket_if_claim
    recommendation = "claim" if net_claim_benefit > 300 else "self_pay"
    return _tool_result(
        ok=True,
        data={
            "repair_cost": repair_cost,
            "premium_increase_estimate": premium_increase,
            "deductible": deductible,
            "net_claim_benefit": round(net_claim_benefit, 2),
            "recommendation": recommendation,
            "reason": "维修费明显高于预计次年保费上涨，建议走保险。" if recommendation == "claim" else "维修费与保费上涨接近，建议优先自费。",
        },
        evidence=[f"repair_cost={repair_cost}", f"premium_increase={premium_increase}"],
        retryable=False,
    )


def file_insurance_claim(payload: dict[str, Any]) -> dict[str, Any]:
    claim = {
        "claim_id": f"IC-{uuid4().hex[:8].upper()}",
        "vin": payload.get("vin", ""),
        "city": payload.get("city", ""),
        "accident_time": payload.get("accident_time", payload.get("time_slot", "")),
        "accident_type": payload.get("accident_type", ""),
        "damage_area": payload.get("damage_area", ""),
        "repair_cost": payload.get("repair_cost", 0),
        "status": "reported",
        "next_steps": ["上传现场照片", "等待保险公司定损", "确认维修门店"],
    }
    return _tool_result(ok=True, data=claim, evidence=[claim["claim_id"]], requires_confirmation=True, retryable=False)


def assess_complaint_level(payload: dict[str, Any]) -> dict[str, Any]:
    repair_count = int(payload.get("repair_count", 0) or 0)
    unresolved = bool(payload.get("unresolved", False))
    issue = payload.get("issue", "")
    level = "normal"
    if repair_count >= 3 and unresolved:
        level = "high"
    elif repair_count >= 2 or unresolved:
        level = "medium"
    return _tool_result(
        ok=True,
        data={
            "issue": issue,
            "repair_count": repair_count,
            "unresolved": unresolved,
            "level": level,
            "escalate": level in {"medium", "high"},
            "sla_hours": 24 if level == "high" else 72,
        },
        evidence=[f"repair_count={repair_count}", f"unresolved={unresolved}"],
        retryable=False,
    )


def create_complaint_ticket(payload: dict[str, Any]) -> dict[str, Any]:
    complaint = {
        "complaint_id": f"CP-{uuid4().hex[:8].upper()}",
        "vin": payload.get("vin", ""),
        "city": payload.get("city", ""),
        "issue": payload.get("issue", ""),
        "repair_count": payload.get("repair_count", 0),
        "work_orders": payload.get("work_orders", []),
        "priority": payload.get("priority", "high"),
        "policy_basis": payload.get("policy_basis", []),
        "status": "submitted",
        "expected_response_hours": payload.get("expected_response_hours", 24),
    }
    return _tool_result(ok=True, data=complaint, evidence=[complaint["complaint_id"]], requires_confirmation=True, retryable=False)


def track_complaint(payload: dict[str, Any]) -> dict[str, Any]:
    complaint_id = payload.get("complaint_id", "")
    return _tool_result(
        ok=bool(complaint_id),
        data={
            "complaint_id": complaint_id,
            "status": "submitted",
            "latest_update": "投诉已进入升级处理队列，等待专员联系。",
        },
        error="" if complaint_id else "缺少 complaint_id",
        evidence=[complaint_id] if complaint_id else [],
        retryable=False,
    )


_SERVICE_CENTERS = [
    {"city": "上海", "name": "上海浦东新能源服务中心", "address": "浦东新区沪南路 1188 号", "phone": "400-820-1188"},
    {"city": "上海", "name": "上海虹桥试驾中心", "address": "闵行区申长路 969 号", "phone": "400-820-7788"},
    {"city": "杭州", "name": "杭州滨江服务中心", "address": "滨江区江晖路 900 号", "phone": "400-571-9000"},
    {"city": "深圳", "name": "深圳南山服务中心", "address": "南山区深南大道 10001 号", "phone": "400-755-6688"},
]


def search_service_centers(payload: dict[str, Any]) -> dict[str, Any]:
    city = payload.get("city", "")
    centers = [center for center in _SERVICE_CENTERS if center["city"] == city]
    if not centers:
        return _tool_result(
            ok=False,
            data={"centers": []},
            error=f"未找到 {city} 的服务中心",
            evidence=[f"local provider miss: {city}"],
            retryable=True,
        )
    return _tool_result(
        ok=True,
        data={"centers": centers},
        evidence=[center["name"] for center in centers],
        retryable=False,
    )


def search_policy_or_warranty(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query", "")
    rows = []
    for item in _load_faq():
        haystack = f"{item.get('question', '')} {item.get('answer', '')}"
        if any(token in haystack for token in query.split()):
            rows.append({"question": item["question"], "answer": item["answer"], "category": item.get("category", "")})
    if "质保" in query or "三包" in query:
        rows.append(
            {
                "question": "整车与三电质保",
                "answer": "整车质保通常为 5 年或 12 万公里，三电系统可达 8 年或 16 万公里，具体以品牌政策为准。",
                "category": "政策",
            }
        )
    return _tool_result(ok=True, data={"policies": rows[:5]}, evidence=[row["question"] for row in rows[:5]], retryable=False)


def search_knowledge_base(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query", "")
    hits = []
    for item in _load_faq():
        haystack = f"{item.get('question', '')} {item.get('answer', '')}"
        score = sum(1 for token in query.split() if token and token in haystack)
        if score:
            hits.append((score, item))
    hits.sort(key=lambda item: item[0], reverse=True)
    results = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "category": item.get("category", ""),
        }
        for _, item in hits[:5]
    ]
    return _tool_result(ok=True, data={"results": results}, evidence=[row["question"] for row in results], retryable=False)


def book_test_drive(payload: dict[str, Any]) -> dict[str, Any]:
    booking = {
        "booking_id": f"TD-{uuid4().hex[:8].upper()}",
        "vehicle_model": payload.get("vehicle_model", ""),
        "city": payload.get("city", ""),
        "time_slot": payload.get("time_slot", ""),
        "name": payload.get("name", "演示用户"),
        "status": "confirmed",
    }
    return _tool_result(ok=True, data=booking, evidence=[booking["booking_id"]], requires_confirmation=True, retryable=False)


def create_service_ticket(payload: dict[str, Any]) -> dict[str, Any]:
    ticket = legacy_create_ticket(
        description=payload.get("issue_description", ""),
        components=payload.get("components", []),
        faults=payload.get("faults", []),
        vehicle_model=payload.get("vehicle_model", ""),
        contact=payload.get("vin", ""),
    )
    ticket["ticket_id"] = f"SV-{uuid4().hex[:8].upper()}"
    return _tool_result(ok=True, data=ticket, evidence=[ticket["ticket_id"]], requires_confirmation=True, retryable=False)


def request_roadside_assistance(payload: dict[str, Any]) -> dict[str, Any]:
    rescue = {
        "rescue_id": f"RA-{uuid4().hex[:8].upper()}",
        "vehicle_model": payload.get("vehicle_model", ""),
        "city": payload.get("city", ""),
        "issue_type": payload.get("issue_type", payload.get("issue_description", "")),
        "vin": payload.get("vin", ""),
        "service_center": payload.get("service_center", ""),
        "eta_minutes": 45,
        "status": "dispatched",
    }
    return _tool_result(ok=True, data=rescue, evidence=[rescue["rescue_id"]], requires_confirmation=True, retryable=False)


def book_service_appointment(payload: dict[str, Any]) -> dict[str, Any]:
    appointment = {
        "appointment_id": f"SA-{uuid4().hex[:8].upper()}",
        "vehicle_model": payload.get("vehicle_model", ""),
        "city": payload.get("city", ""),
        "time_slot": payload.get("time_slot", ""),
        "ticket_id": payload.get("ticket_id", ""),
        "service_center": payload.get("service_center", ""),
        "status": "confirmed",
    }
    return _tool_result(ok=True, data=appointment, evidence=[appointment["appointment_id"]], requires_confirmation=True, retryable=False)


REGISTRY = ToolRegistry()
REGISTRY.register("search_vehicles", "read", False, search_vehicles)
REGISTRY.register("get_vehicle_detail", "read", False, get_vehicle_detail)
REGISTRY.register("compare_vehicles", "read", False, lambda payload: compare_vehicles(payload.get("model_ids", [])))
REGISTRY.register("search_service_centers", "read", False, search_service_centers)
REGISTRY.register("search_charging_stations", "read", False, search_charging_stations)
REGISTRY.register("calculate_cost", "read", False, calculate_cost)
REGISTRY.register("check_subsidy", "read", False, check_subsidy)
REGISTRY.register("generate_comparison_report", "read", False, generate_comparison_report)
REGISTRY.register("plan_route", "read", False, plan_route)
REGISTRY.register("search_charging_stations_along_route", "read", False, search_charging_stations_along_route)
REGISTRY.register("generate_charging_plan", "read", False, generate_charging_plan)
REGISTRY.register("estimate_trip_cost", "read", False, estimate_trip_cost)
REGISTRY.register("generate_trip_report", "read", False, generate_trip_report)
REGISTRY.register("estimate_repair_cost", "read", False, estimate_repair_cost)
REGISTRY.register("calculate_claim_impact", "read", False, calculate_claim_impact)
REGISTRY.register("assess_complaint_level", "read", False, assess_complaint_level)
REGISTRY.register("track_complaint", "read", False, track_complaint)
REGISTRY.register("search_policy_or_warranty", "read", False, search_policy_or_warranty)
REGISTRY.register("search_knowledge_base", "read", False, search_knowledge_base)
REGISTRY.register("book_test_drive", "write", True, book_test_drive)
REGISTRY.register("create_service_ticket", "write", True, create_service_ticket)
REGISTRY.register("request_roadside_assistance", "write", True, request_roadside_assistance)
REGISTRY.register("book_service_appointment", "write", True, book_service_appointment)
REGISTRY.register("file_insurance_claim", "write", True, file_insurance_claim)
REGISTRY.register("create_complaint_ticket", "write", True, create_complaint_ticket)
