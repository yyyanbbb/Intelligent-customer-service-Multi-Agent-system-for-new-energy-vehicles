"""
电池领域 JSON Schema 定义
定义了 5 大类实体的结构化抽取规范，
涵盖字段类型、必填项、数值范围、枚举值等约束。
"""
from __future__ import annotations

BATTERY_ENTITY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "BatteryEntity",
    "description": "电池领域实体基类",
    "type": "object",
    "properties": {
        "entity_type": {
            "type": "string",
            "description": "实体类型",
            "enum": [
                "BatteryModel",
                "TestCondition",
                "PerformanceMetric",
                "DegradationCurve",
                "TemperatureThreshold",
            ],
        },
        "name": {
            "type": "string",
            "description": "实体名称",
            "minLength": 1,
        },
        "attributes": {
            "type": "object",
            "description": "实体属性",
        },
        "source": {
            "type": "string",
            "description": "信息来源",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "置信度",
        },
    },
    "required": ["entity_type", "name"],
}


BATTERY_MODELS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "BatteryModel",
    "type": "object",
    "properties": {
        "entity_type": {"const": "BatteryModel"},
        "name": {"type": "string", "description": "电池型号名称"},
        "attributes": {
            "type": "object",
            "properties": {
                "capacity_wh": {
                    "type": "number",
                    "description": "标称容量（Wh）",
                    "minimum": 1000,
                    "maximum": 200000,
                },
                "voltage_nominal": {
                    "type": "number",
                    "description": "标称电压（V）",
                    "minimum": 200,
                    "maximum": 500,
                },
                "chemistry": {
                    "type": "string",
                    "description": "电池化学体系",
                    "enum": ["NCM", "NCA", "LFP", "LMO", "NCM811", "NCA622", "unknown"],
                },
                "weight_kg": {
                    "type": "number",
                    "description": "重量（kg）",
                    "minimum": 1,
                    "maximum": 2000,
                },
                "cycle_life": {
                    "type": "integer",
                    "description": "标称循环寿命（次）",
                    "minimum": 0,
                    "maximum": 10000,
                },
                "max_charge_rate": {
                    "type": "number",
                    "description": "最大充电倍率（C）",
                    "minimum": 0,
                    "maximum": 10,
                },
            },
            "required": ["capacity_wh", "voltage_nominal"],
        },
        "source": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["entity_type", "name", "attributes"],
}


TEST_CONDITIONS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TestCondition",
    "type": "object",
    "properties": {
        "entity_type": {"const": "TestCondition"},
        "name": {"type": "string", "description": "测试条件名称"},
        "attributes": {
            "type": "object",
            "properties": {
                "temperature_c": {
                    "type": "number",
                    "description": "测试温度（℃）",
                    "minimum": -40,
                    "maximum": 80,
                },
                "charge_rate": {
                    "type": "number",
                    "description": "充电倍率（C）",
                    "minimum": 0,
                    "maximum": 10,
                },
                "discharge_rate": {
                    "type": "number",
                    "description": "放电倍率（C）",
                    "minimum": 0,
                    "maximum": 10,
                },
                "soc_level": {
                    "type": "number",
                    "description": "SOC 初始状态（%）",
                    "minimum": 0,
                    "maximum": 100,
                },
                "test_duration_h": {
                    "type": "number",
                    "description": "测试时长（小时）",
                    "minimum": 0,
                },
                "test_standard": {
                    "type": "string",
                    "description": "测试标准（如 IEC62660）",
                },
            },
            "required": ["temperature_c", "charge_rate", "discharge_rate"],
        },
        "source": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["entity_type", "name", "attributes"],
}


PERFORMANCE_METRICS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PerformanceMetric",
    "type": "object",
    "properties": {
        "entity_type": {"const": "PerformanceMetric"},
        "name": {"type": "string", "description": "指标名称"},
        "attributes": {
            "type": "object",
            "properties": {
                "metric_name": {
                    "type": "string",
                    "enum": [
                        "soh", "soc", "energy_density_wh_kg",
                        "power_density_w_kg", "round_trip_efficiency",
                        "self_discharge_rate", "capacity_wh",
                        "internal_resistance_mohm", "open_circuit_voltage",
                    ],
                },
                "value": {
                    "type": "number",
                    "description": "指标数值",
                },
                "unit": {
                    "type": "string",
                    "description": "单位",
                },
                "timestamp": {
                    "type": "string",
                    "description": "测试时间戳",
                },
                "test_id": {
                    "type": "string",
                    "description": "关联测试 ID",
                },
            },
            "required": ["metric_name", "value", "unit"],
        },
        "source": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["entity_type", "name", "attributes"],
}


DEGRADATION_CURVE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DegradationCurve",
    "type": "object",
    "properties": {
        "entity_type": {"const": "DegradationCurve"},
        "name": {"type": "string", "description": "衰减曲线名称"},
        "attributes": {
            "type": "object",
            "properties": {
                "cycle_count": {
                    "type": "integer",
                    "description": "循环次数",
                    "minimum": 0,
                    "maximum": 100000,
                },
                "capacity_retention": {
                    "type": "number",
                    "description": "容量保持率（%）",
                    "minimum": 0,
                    "maximum": 100,
                },
                "soh": {
                    "type": "number",
                    "description": "健康状态（%）",
                    "minimum": 0,
                    "maximum": 100,
                },
                "temperature_c": {
                    "type": "number",
                    "description": "测试温度（℃）",
                },
                "test_id": {
                    "type": "string",
                    "description": "关联测试 ID",
                },
            },
            "required": ["cycle_count", "capacity_retention"],
        },
        "source": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["entity_type", "name", "attributes"],
}


TEMPERATURE_THRESHOLD_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TemperatureThreshold",
    "type": "object",
    "properties": {
        "entity_type": {"const": "TemperatureThreshold"},
        "name": {"type": "string", "description": "阈值名称"},
        "attributes": {
            "type": "object",
            "properties": {
                "threshold_type": {
                    "type": "string",
                    "description": "阈值类型",
                    "enum": [
                        "max_charge_temp",
                        "max_discharge_temp",
                        "min_temp",
                        "critical_high_temp",
                        "critical_low_temp",
                        "optimal_temp",
                    ],
                },
                "value_c": {
                    "type": "number",
                    "description": "温度阈值（℃）",
                    "minimum": -50,
                    "maximum": 150,
                },
                "condition": {
                    "type": "string",
                    "description": "触发条件描述",
                },
                "warning_level": {
                    "type": "string",
                    "enum": ["info", "warning", "critical", "emergency"],
                },
            },
            "required": ["threshold_type", "value_c", "condition"],
        },
        "source": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["entity_type", "name", "attributes"],
}


RELATION_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Relation",
    "type": "object",
    "properties": {
        "source_name": {"type": "string", "description": "源实体名称"},
        "target_name": {"type": "string", "description": "目标实体名称"},
        "relation_type": {
            "type": "string",
            "enum": [
                "battery_used_in_test",
                "test_generates_metric",
                "metric_belongs_to_battery",
                "curve_records_degradation",
                "threshold_protects_component",
                "condition_affects_performance",
                "related_to",
            ],
            "description": "关系类型",
        },
        "attributes": {"type": "object"},
    },
    "required": ["source_name", "target_name", "relation_type"],
}


ALL_SCHEMAS = {
    "BatteryModel": BATTERY_MODELS_SCHEMA,
    "TestCondition": TEST_CONDITIONS_SCHEMA,
    "PerformanceMetric": PERFORMANCE_METRICS_SCHEMA,
    "DegradationCurve": DEGRADATION_CURVE_SCHEMA,
    "TemperatureThreshold": TEMPERATURE_THRESHOLD_SCHEMA,
}


def get_schema(entity_type: str) -> dict | None:
    return ALL_SCHEMAS.get(entity_type)
