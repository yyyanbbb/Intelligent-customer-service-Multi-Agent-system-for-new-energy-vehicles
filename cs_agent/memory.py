"""
跨会话用户记忆，仿 Mem0 思路的轻量实现。
每轮对话自动从实体和规则中提取预算/偏好/关注车型，存到本地 JSON。
memory_as_context() 把记忆格式化成一段文字注入 system prompt。
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime

_MEM_DIR = Path(__file__).parent / "knowledge" / "memory"
_MEM_DIR.mkdir(parents=True, exist_ok=True)


_PROFILE_KEYS = ("budget", "interested_models", "interested_brands", "usage_scenario", "family_size")


def _mem_path(session_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)[:64] or "default"
    return _MEM_DIR / f"{safe}.json"


def load_memory(session_id: str) -> dict:
    p = _mem_path(session_id)
    if not p.exists():
        return {
            "session_id": session_id,
            "profile": {k: None for k in _PROFILE_KEYS},
            "interested_models": [],
            "interested_brands": [],
            "recent_intents": [],
            "facts": [],
            "updated_at": "",
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return load_memory("default")


def save_memory(mem: dict) -> None:
    mem["updated_at"] = datetime.now().isoformat(timespec="seconds")
    p = _mem_path(mem.get("session_id", "default"))
    p.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")


def update_from_turn(session_id: str, query: str, intent: str, entities: list[dict]) -> dict:
    """每轮对话后增量更新记忆，保留最近 20 个车型/品牌。"""
    mem = load_memory(session_id)

    # recent intents (最多 10 条)
    mem["recent_intents"] = (mem.get("recent_intents", []) + [intent])[-10:]

    # 从实体累计偏好
    for e in entities:
        if e["label"] == "vehicle_model" and e["text"] not in mem["interested_models"]:
            mem["interested_models"].append(e["text"])
        if e["label"] == "brand" and e["text"] not in mem["interested_brands"]:
            mem["interested_brands"].append(e["text"])
        if e["label"] == "budget":
            mem["profile"]["budget"] = e["text"]

    # 规则提取使用场景
    scenario_map = {
        "家用": ["家用", "接送孩子", "老婆", "家庭"],
        "商务": ["商务", "接客户", "公司"],
        "越野": ["越野", "穿越", "野外"],
        "性能": ["性能", "赛道", "驾控"],
    }
    for s, kws in scenario_map.items():
        if any(kw in query for kw in kws):
            mem["profile"]["usage_scenario"] = s

    mem["interested_models"] = mem["interested_models"][-20:]
    mem["interested_brands"] = mem["interested_brands"][-20:]
    save_memory(mem)
    return mem


def memory_as_context(session_id: str) -> str:
    """把记忆拼成一段简短文字，供 system prompt 注入。"""
    mem = load_memory(session_id)
    parts = []
    if mem["profile"].get("budget"):
        parts.append(f"用户预算：{mem['profile']['budget']}")
    if mem["profile"].get("usage_scenario"):
        parts.append(f"用车场景：{mem['profile']['usage_scenario']}")
    if mem["interested_models"]:
        parts.append(f"关注车型：{', '.join(mem['interested_models'][-5:])}")
    if mem["interested_brands"]:
        parts.append(f"关注品牌：{', '.join(mem['interested_brands'][-5:])}")
    return "；".join(parts) if parts else ""


def clear_memory(session_id: str) -> None:
    p = _mem_path(session_id)
    if p.exists():
        p.unlink()
