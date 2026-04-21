"""
工单生成工具：将故障实体转为结构化服务工单。
"""
from __future__ import annotations
from datetime import datetime
import uuid


def create_ticket(
    description: str,
    components: list[str],
    faults: list[str],
    vehicle_model: str = "",
    contact: str = "",
) -> dict:
    """生成售后服务工单，返回工单字典。"""
    is_urgent = any(
        kw in description for kw in ["刹车", "制动", "起火", "冒烟", "失灵", "漏气", "安全", "报警"]
    )
    ticket = {
        "ticket_id": f"WO{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "vehicle_model": vehicle_model,
        "description": description,
        "components": components,
        "faults": faults,
        "priority": "urgent" if is_urgent else "normal",
        "status": "pending",
        "contact": contact,
        "estimated_response_hours": 2 if is_urgent else 24,
    }
    return ticket
