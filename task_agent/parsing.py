from __future__ import annotations

import re
from typing import Any

from cs_agent.tools.ner_tool import extract_entities


_BUDGET_RE = re.compile(r"(\d+(?:\.\d+)?)\s*万")
_COMMUTE_RE = re.compile(r"(\d{2,4})\s*公里")
_VIN_RE = re.compile(r"\b[A-Z0-9]{17,20}\b")
_CITY_NAMES = ("上海", "北京", "杭州", "深圳", "广州", "成都", "苏州", "南京", "武汉")
_TIME_HINTS = ("今天下午", "今天上午", "今晚", "明天上午", "明天下午", "明天晚上", "后天上午", "后天下午")


def classify_task(query: str) -> str:
    lowered = query.lower()
    if any(keyword in query for keyword in ("投诉", "升级", "维权", "三包")):
        return "complaint"
    if any(keyword in query for keyword in ("保险", "理赔", "报案", "事故", "刮", "剐蹭")):
        return "insurance"
    if any(keyword in query for keyword in ("充电方案", "充电规划", "长途", "路线规划", "补能方案")):
        return "charging"
    if any(keyword in query for keyword in ("买", "选车", "推荐", "试驾", "预算", "对比")):
        return "purchase"
    if any(keyword in query for keyword in ("异响", "故障", "维修", "保养", "工单", "刹车", "救援")):
        return "aftersales"
    if any(keyword in query for keyword in ("你好", "谢谢", "你是谁")):
        return "chitchat"
    if any(keyword in lowered for keyword in ("faq", "说明书", "质保")):
        return "faq"
    return "purchase"


def merge_collected_info(existing: dict[str, Any], text: str) -> dict[str, Any]:
    info = dict(existing)
    entities = extract_entities(text)
    budget_match = _BUDGET_RE.search(text)
    if budget_match:
        info["budget"] = f"{budget_match.group(1)}万"
        info["budget_max"] = int(float(budget_match.group(1)) * 10000)

    commute_match = _COMMUTE_RE.search(text)
    if commute_match:
        info["daily_commute_km"] = int(commute_match.group(1))

    if any(keyword in text for keyword in ("有充电桩", "家里有桩", "公司有桩", "家里有充电条件")):
        info["charging_condition"] = "home_charger"
    elif any(keyword in text for keyword in ("没有充电桩", "没充电桩", "无桩")):
        info["charging_condition"] = "public_only"

    if "家用SUV" in text or "SUV" in text.upper():
        info["space_preference"] = "SUV"
    elif any(keyword in text for keyword in ("大空间", "家用", "带娃", "家庭")):
        info["space_preference"] = "family"

    if any(keyword in text for keyword in ("家用", "通勤", "商务")):
        if "家用" in text:
            info["use_case"] = "family"
        elif "通勤" in text:
            info["use_case"] = "commute"
        elif "商务" in text:
            info["use_case"] = "business"

    for city in _CITY_NAMES:
        if city in text:
            info["city"] = city
            break

    cities_in_text = [city for city in _CITY_NAMES if city in text]
    if "从" in text and ("到" in text or "去" in text) and len(cities_in_text) >= 2:
        info["origin"] = cities_in_text[0]
        info["destination"] = cities_in_text[1]

    for hint in _TIME_HINTS:
        if hint in text:
            info["time_slot"] = hint
            break

    vin_match = _VIN_RE.search(text.upper())
    if vin_match:
        info["vin"] = vin_match.group(0)

    if "不需要道路救援" in text or "不用道路救援" in text:
        info["roadside_preference"] = "declined"
    elif "道路救援" in text:
        info["roadside_preference"] = "requested"

    damage_keywords = ("右后门", "左后门", "右前门", "左前门", "前保险杠", "后保险杠", "前杠", "后杠", "车门", "轮毂")
    for keyword in damage_keywords:
        if keyword in text:
            info["damage_area"] = f"{keyword}刮擦" if "刮" in text and "刮擦" not in keyword else keyword
            break
    if "单方" in text:
        info["accident_type"] = "single_party"
    elif "双方" in text or "对方" in text:
        info["accident_type"] = "two_party"
    if "没有人员伤亡" in text or "无人伤" in text:
        info["injury"] = "none"
    elif "人员伤亡" in text or "受伤" in text:
        info["injury"] = "injury_reported"
    if "走保险" in text or "报保险" in text or "理赔" in text:
        info["claim_preference"] = "claim"
    elif "自费" in text:
        info["claim_preference"] = "self_pay"

    repair_count_match = re.search(r"(?:维修|修了|去了|去4S店维修)?\s*(\d+|一|二|三|四|五)\s*次", text)
    if repair_count_match:
        raw_count = repair_count_match.group(1)
        number_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
        info["repair_count"] = number_map.get(raw_count, int(raw_count) if raw_count.isdigit() else 0)
    if any(keyword in text for keyword in ("未解决", "还没好", "修不好", "仍未解决")):
        info["unresolved"] = True
    if "异响" in text:
        info["complaint_issue"] = "车门异响" if "车门" in text else "异响"
    work_orders = re.findall(r"\bSV\d+\b", text.upper())
    if work_orders:
        info["work_orders"] = work_orders

    models = [entity["text"] for entity in entities if entity["label"] == "vehicle_model"]
    if "Model Y" in text or "model y" in text.lower():
        models.append("Model Y")
    if models:
        info.setdefault("candidate_models", [])
        for model in models:
            if model not in info["candidate_models"]:
                info["candidate_models"].append(model)
        info["selected_model"] = models[-1]

    components = [entity["text"] for entity in entities if entity["label"] == "component"]
    faults = [entity["text"] for entity in entities if entity["label"] == "fault"]
    if components:
        info["components"] = sorted(set(info.get("components", []) + components))
    if faults:
        info["faults"] = sorted(set(info.get("faults", []) + faults))

    info["raw_entities"] = entities
    return info


def purchase_missing_fields(info: dict[str, Any]) -> list[str]:
    questions = []
    if "budget" not in info:
        questions.append("请告诉我您的预算范围，例如 20万 或 25万以内。")
    if "charging_condition" not in info:
        questions.append("家里或公司有固定充电桩吗？")
    if "use_case" not in info:
        questions.append("主要是通勤、家用，还是兼顾长途出行？")
    if "space_preference" not in info:
        questions.append("对空间有没有明确偏好，例如 SUV、五座或大空间？")
    return questions


def purchase_booking_missing_fields(info: dict[str, Any]) -> list[str]:
    questions = []
    if "selected_model" not in info:
        questions.append("我已经给出候选车型了，您想预约试驾哪一款？")
    if "city" not in info:
        questions.append("您希望在哪个城市预约试驾？")
    if "time_slot" not in info:
        questions.append("您希望预约哪个时间段试驾？")
    return questions


def aftersales_missing_fields(info: dict[str, Any]) -> list[str]:
    questions = []
    if "vin" not in info:
        questions.append("请提供车辆 VIN 码或车牌号。")
    if "city" not in info:
        questions.append("请告诉我您所在的城市，方便查询服务中心。")
    if "time_slot" not in info:
        questions.append("您方便到店或处理的时间段是什么时候？")
    return questions


def charging_missing_fields(info: dict[str, Any]) -> list[str]:
    questions = []
    if "origin" not in info:
        questions.append("请告诉我出发城市，例如上海。")
    if "destination" not in info:
        questions.append("请告诉我目的地城市，例如成都。")
    if "selected_model" not in info:
        questions.append("请告诉我车型，例如 Model Y 长续航。")
    return questions


def insurance_missing_fields(info: dict[str, Any]) -> list[str]:
    questions = []
    if "vin" not in info:
        questions.append("请提供车辆 VIN 或车牌号，便于生成保险报案记录。")
    if "city" not in info:
        questions.append("请告诉我事故所在城市。")
    if "time_slot" not in info:
        questions.append("请告诉我事故发生时间。")
    if "damage_area" not in info:
        questions.append("请描述受损位置，例如右后门刮擦。")
    if "accident_type" not in info:
        questions.append("这是单方事故还是双方事故？")
    if "injury" not in info:
        questions.append("是否有人员伤亡？")
    return questions


def complaint_missing_fields(info: dict[str, Any]) -> list[str]:
    questions = []
    if "vin" not in info:
        questions.append("请提供车辆 VIN 或车牌号，便于关联历史维修记录。")
    if "city" not in info:
        questions.append("请告诉我所在城市，便于确定投诉受理和服务中心区域。")
    if "complaint_issue" not in info:
        questions.append("请说明要投诉的问题，例如车门异响、同一故障多次未修复。")
    if "repair_count" not in info:
        questions.append("请告诉我同一问题已经维修过几次。")
    if "unresolved" not in info:
        questions.append("目前问题是否仍未解决？")
    return questions
