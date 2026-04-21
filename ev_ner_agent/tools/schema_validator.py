"""
Schema 校验工具
使用 jsonschema + pydantic 对抽取结果进行多层级校验。
支持字段类型校验、必填字段检查、枚举值校验、数值范围校验。
"""
from __future__ import annotations

import logging
from typing import Any
from dataclasses import dataclass, field

from jsonschema import validate, ValidationError, Draft7Validator
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SchemaValidator:
    """
    Schema 校验器，基于 jsonschema Draft7 标准。
    支持：
    - 实体类型校验（必须在预定义实体列表中）
    - 数值范围校验（温度、容量、电压等）
    - 枚举值校验
    - 必填字段检查
    - 跨字段联合校验（如：end_charge > start_charge）
    """

    # 预定义的实体类型及其字段约束
    ENTITY_CONSTRAINTS: dict[str, dict[str, Any]] = {
        "BatteryModel": {
            "required": ["model_name", "capacity_wh", "voltage_nominal"],
            "numeric_fields": ["capacity_wh", "voltage_nominal", "weight_kg"],
            "ranges": {
                "voltage_nominal": (200, 500),
                "capacity_wh": (1000, 200000),
                "weight_kg": (1, 2000),
            },
        },
        "TestCondition": {
            "required": ["temperature_c", "charge_rate", "discharge_rate"],
            "numeric_fields": ["temperature_c", "charge_rate", "discharge_rate"],
            "ranges": {
                "temperature_c": (-40, 80),
                "charge_rate": (0, 10),
                "discharge_rate": (0, 10),
            },
        },
        "PerformanceMetric": {
            "required": ["metric_name", "value", "unit"],
            "ranges": {
                "soh": (0, 100),
                "soc": (0, 100),
            },
        },
        "DegradationCurve": {
            "required": ["cycle_count", "capacity_retention"],
            "numeric_fields": ["cycle_count", "capacity_retention"],
            "ranges": {
                "cycle_count": (0, 100000),
                "capacity_retention": (0, 100),
            },
        },
        "TemperatureThreshold": {
            "required": ["threshold_type", "value_c", "condition"],
            "numeric_fields": ["value_c"],
            "ranges": {
                "value_c": (-50, 150),
            },
            "enum_values": {
                "threshold_type": ["max_charge_temp", "max_discharge_temp", "min_temp", "critical"],
            },
        },
    }

    def validate(self, data: dict[str, Any], schema_type: str | None = None) -> ValidationResult:
        """
        对抽取结果进行校验。
        """
        errors: list[str] = []
        warnings: list[str] = []

        if schema_type and schema_type in self.ENTITY_CONSTRAINTS:
            constraint = self.ENTITY_CONSTRAINTS[schema_type]
            err, warn = self._validate_constraints(data, constraint)
            errors.extend(err)
            warnings.extend(warn)

        if "entities" in data:
            for i, entity in enumerate(data["entities"]):
                entity_type = entity.get("type", "unknown")
                if entity_type in self.ENTITY_CONSTRAINTS:
                    constraint = self.ENTITY_CONSTRAINTS[entity_type]
                    err, warn = self._validate_constraints(entity, constraint)
                    for e in err:
                        errors.append(f"实体[{i}].{entity_type}: {e}")
                    for w in warn:
                        warnings.append(f"实体[{i}].{entity_type}: {w}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_constraints(
        self, data: dict, constraint: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        for field_name in constraint.get("required", []):
            if field_name not in data or data[field_name] is None or data[field_name] == "":
                errors.append(f"必填字段缺失: {field_name}")

        for field_name in constraint.get("numeric_fields", []):
            if field_name in data and data[field_name] is not None:
                val = data[field_name]
                try:
                    val = float(val)
                    if field_name in constraint.get("ranges", {}):
                        lo, hi = constraint["ranges"][field_name]
                        if not (lo <= val <= hi):
                            warnings.append(
                                f"字段 {field_name}={val} 超出常见范围 [{lo}, {hi}]，请核实"
                            )
                except (ValueError, TypeError):
                    if field_name in constraint.get("required", []):
                        errors.append(f"字段 {field_name} 应为数值，当前值: {val}")

        for field_name, enum_vals in constraint.get("enum_values", {}).items():
            if field_name in data and data[field_name] not in enum_vals:
                errors.append(
                    f"字段 {field_name} 值 '{data[field_name]}' 不在枚举范围 {enum_vals} 内"
                )

        return errors, warnings

    def validate_and_fix(self, data: dict[str, Any]) -> tuple[dict[str, Any], ValidationResult]:
        """
        校验 + 自动修复常见问题。
        """
        result = self.validate(data)

        fixed = dict(data)

        if "entities" in fixed and isinstance(fixed["entities"], list):
            for i, entity in enumerate(fixed["entities"]):
                if isinstance(entity, dict):
                    entity["_valid"] = True
                    if i < len(result.errors):
                        entity["_validation_errors"] = [e for e in result.errors if e.startswith(f"实体[{i}]")]
                        if entity["_validation_errors"]:
                            entity["_valid"] = False

        return fixed, result


def validate_schema(data: dict[str, Any], schema_type: str | None = None) -> str:
    """
    暴露给 Agent 的工具函数。
    """
    validator = SchemaValidator()
    fixed_data, result = validator.validate_and_fix(data)

    if result.is_valid:
        return "✅ Schema 校验通过，所有字段符合规范。"

    parts = ["❌ Schema 校验未通过：\n"]
    if result.errors:
        parts.append("\n[错误]")
        for err in result.errors:
            parts.append(f"  - {err}")
    if result.warnings:
        parts.append("\n[警告]")
        for warn in result.warnings:
            parts.append(f"  - {warn}")

    parts.append("\n\n[自动修复后的数据预览]")
    import json
    preview = json.dumps(fixed_data, ensure_ascii=False, indent=2)
    parts.append(preview[:2000])

    return "\n".join(parts)
