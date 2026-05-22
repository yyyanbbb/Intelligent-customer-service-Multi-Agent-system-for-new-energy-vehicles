from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class MobilityProvider(Protocol):
    name: str

    def search_charging_stations(self, city: str, radius_km: int) -> list[dict[str, Any]]:
        ...

    def subsidy_policy(self, city: str, model_id: str) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class StaticMobilityProvider:
    name: str = "static-mobility-provider"

    _stations = (
        {
            "city": "上海",
            "name": "上海虹桥超充站",
            "address": "申长路 900 号",
            "power_kw": 250,
            "distance_km": 4.2,
            "open_hours": "24h",
        },
        {
            "city": "上海",
            "name": "上海浦东新能源补能中心",
            "address": "沪南路 1188 号",
            "power_kw": 180,
            "distance_km": 7.6,
            "open_hours": "07:00-23:00",
        },
        {
            "city": "杭州",
            "name": "杭州滨江快充站",
            "address": "江晖路 900 号",
            "power_kw": 160,
            "distance_km": 5.3,
            "open_hours": "24h",
        },
        {
            "city": "深圳",
            "name": "深圳南山超充站",
            "address": "深南大道 10001 号",
            "power_kw": 250,
            "distance_km": 6.8,
            "open_hours": "24h",
        },
    )

    _subsidies = {
        "上海": {
            "eligible": True,
            "estimated_amount": 10000,
            "policy_items": ["置换补贴最高 10000 元", "新能源专用牌照便利政策"],
        },
        "杭州": {
            "eligible": True,
            "estimated_amount": 6000,
            "policy_items": ["区级新能源消费补贴最高 6000 元"],
        },
        "深圳": {
            "eligible": True,
            "estimated_amount": 8000,
            "policy_items": ["以旧换新补贴最高 8000 元"],
        },
    }

    def search_charging_stations(self, city: str, radius_km: int) -> list[dict[str, Any]]:
        stations = [
            {**station, "provider": self.name}
            for station in self._stations
            if station["city"] == city and station["distance_km"] <= radius_km
        ]
        return sorted(stations, key=lambda station: (-station["power_kw"], station["distance_km"]))

    def subsidy_policy(self, city: str, model_id: str) -> dict[str, Any]:
        policy = self._subsidies.get(
            city,
            {
                "eligible": False,
                "estimated_amount": 0,
                "policy_items": ["暂未命中本地补贴规则，建议以当地商务部门公告为准"],
            },
        )
        return {
            "city": city,
            "model_id": model_id,
            "provider": self.name,
            **policy,
        }


def get_default_mobility_provider() -> MobilityProvider:
    return StaticMobilityProvider()
