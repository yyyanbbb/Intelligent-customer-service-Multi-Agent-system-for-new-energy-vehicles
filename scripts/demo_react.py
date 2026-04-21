"""
Demo 脚本：展示 ReAct Agent 的完整工作流程
不依赖真实模型，用 mock 方式模拟 Agent 的思考过程。

用法：
  python scripts/demo_react.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ev_ner_agent.agent.react_loop import StepRecord, ExtractionResult
from ev_ner_agent.tools import TOOL_SCHEMAS, execute_tool


def print_step(step: StepRecord) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Step {step.step} | 耗时: {step.elapsed:.3f}s")
    print(f"{'=' * 60}")
    print(f"  Thought: {step.thought}")
    if step.action:
        args_str = json.dumps(step.tool_args or {}, ensure_ascii=False)[:100]
        print(f"  Action:  {step.action}({args_str}...)")
    if step.observation:
        print(f"  Result:  {step.observation[:300]}...")
    if step.error:
        print(f"  Error:   {step.error}")


def demo_extraction_flow():
    """
    演示一个完整的抽取流程（mock 方式）。
    模拟 Agent 如何逐步调用工具完成电池报告抽取。
    """
    print("\n" + "=" * 60)
    print("  EV NER Agent — ReAct 流程演示")
    print("  场景：从电池诊断报告中抽取实体和关系")
    print("=" * 60)

    print("""
场景描述：
  用户上传了一份电池诊断报告（diagnostic_report.pdf），
  要求从中抽取所有电池型号、测试条件、温度阈值。

模拟的 ReAct 执行流程如下：
  Step 1: Agent 判断需要先读取文档内容
  Step 2: Agent 提取 PDF 文本
  Step 3: Agent 分析文档类型
  Step 4: Agent 提取表格数据
  Step 5: Agent 抽取 BatteryModel 实体
  Step 6: Agent 抽取 TestCondition 实体
  Step 7: Agent 查询知识图谱（去重检查）
  Step 8: Agent 抽取 TemperatureThreshold 实体
  Step 9: Agent 写入知识图谱
  Step 10: Agent 输出结构化结果
""")

    # 模拟一个典型的 step history
    mock_steps = [
        StepRecord(
            step=1,
            thought="我需要先提取 PDF 文档的文本内容，了解文档的结构和内容类型",
            action="extract_pdf_text",
            tool_args={"path": "data/diagnostic_report.pdf"},
            observation="成功提取 15 段文本，文档类型: diagnostic_report，总页数: 8",
            error=None,
            elapsed=0.45,
        ),
        StepRecord(
            step=2,
            thought="从第1页的文本来看，这是一个电池诊断报告，包含容量测试数据。我需要提取表格数据来获取具体的数值信息",
            action="extract_tables",
            tool_args={"path": "data/diagnostic_report.pdf", "pages": [2, 3, 5]},
            observation="成功提取 3 张表格：表1(容量测试)、表2(SOH衰减)、表3(温度阈值)",
            error=None,
            elapsed=0.82,
        ),
        StepRecord(
            step=3,
            thought="文档类型已确认为 diagnostic_report，包含电池型号和测试条件数据",
            action=None,
            tool_args=None,
            observation="类型预判为 diagnostic_report",
            error=None,
            elapsed=0.01,
        ),
        StepRecord(
            step=4,
            thought="现在我需要查询知识图谱，检查是否已有相关的电池型号记录",
            action="search_knowledge_graph",
            tool_args={"query": "battery", "entity_type": "BatteryModel"},
            observation="图谱中未找到相关实体，需要新建",
            error=None,
            elapsed=0.05,
        ),
        StepRecord(
            step=5,
            thought="从表格数据中识别到电池型号 NCM-80000，容量 80000Wh，电压 400V，化学体系 NCM",
            action="validate_schema",
            tool_args={
                "data": {
                    "entity_type": "BatteryModel",
                    "name": "NCM-80000",
                    "attributes": {"capacity_wh": 80000, "voltage_nominal": 400, "chemistry": "NCM"}
                },
                "schema_type": "BatteryModel",
            },
            observation="Schema 校验通过",
            error=None,
            elapsed=0.03,
        ),
        StepRecord(
            step=6,
            thought="检测到重复调用 search_knowledge_graph，应该换一个策略，直接写入图谱",
            action="write_to_graph",
            tool_args={
                "entities": [
                    {
                        "entity_type": "BatteryModel",
                        "name": "NCM-80000",
                        "attributes": {"capacity_wh": 80000, "voltage_nominal": 400, "chemistry": "NCM", "weight_kg": 450},
                    },
                    {
                        "entity_type": "TestCondition",
                        "name": "25C标准测试",
                        "attributes": {"temperature_c": 25, "charge_rate": 1.0, "discharge_rate": 1.0},
                    },
                ],
                "relations": [
                    {
                        "source_name": "NCM-80000",
                        "target_name": "25C标准测试",
                        "relation_type": "battery_used_in_test",
                    }
                ],
                "source": "diagnostic_report.pdf",
            },
            observation="写入 2 个实体，1 条关系。图谱统计: 2 entities, 1 relations",
            error=None,
            elapsed=0.08,
        ),
        StepRecord(
            step=7,
            thought="所有主要实体已抽取完毕，现在输出最终结构化结果",
            action=None,
            tool_args=None,
            observation="抽取到 5 个实体，4 条关系",
            error=None,
            elapsed=0.0,
        ),
    ]

    for step in mock_steps:
        print_step(step)

    print("\n" + "=" * 60)
    print("  最终抽取结果")
    print("=" * 60)

    result = ExtractionResult()
    result.entities = [
        {
            "entity_type": "BatteryModel",
            "name": "NCM-80000",
            "attributes": {"capacity_wh": 80000, "voltage_nominal": 400, "chemistry": "NCM", "weight_kg": 450},
            "source": "diagnostic_report.pdf",
            "confidence": 0.95,
        },
        {
            "entity_type": "BatteryModel",
            "name": "LFP-60000",
            "attributes": {"capacity_wh": 60000, "voltage_nominal": 350, "chemistry": "LFP", "weight_kg": 380},
            "source": "diagnostic_report.pdf",
            "confidence": 0.92,
        },
        {
            "entity_type": "TestCondition",
            "name": "25C标准测试",
            "attributes": {"temperature_c": 25, "charge_rate": 1.0, "discharge_rate": 1.0},
            "source": "diagnostic_report.pdf",
            "confidence": 0.98,
        },
        {
            "entity_type": "PerformanceMetric",
            "name": "SOH_20240115",
            "attributes": {"metric_name": "soh", "value": 87.5, "unit": "%", "timestamp": "2024-01-15"},
            "source": "diagnostic_report.pdf",
            "confidence": 0.90,
        },
        {
            "entity_type": "TemperatureThreshold",
            "name": "充电温度上限",
            "attributes": {"threshold_type": "max_charge_temp", "value_c": 55, "condition": "充电时电池温度不得超过", "warning_level": "warning"},
            "source": "diagnostic_report.pdf",
            "confidence": 0.97,
        },
    ]
    result.relations = [
        {"source_name": "NCM-80000", "target_name": "25C标准测试", "relation_type": "battery_used_in_test"},
        {"source_name": "25C标准测试", "target_name": "SOH_20240115", "relation_type": "test_generates_metric"},
        {"source_name": "SOH_20240115", "target_name": "NCM-80000", "relation_type": "metric_belongs_to_battery"},
        {"source_name": "充电温度上限", "target_name": "NCM-80000", "relation_type": "threshold_protects_component"},
    ]
    result.summary = "从诊断报告中抽取到 5 个实体（2个电池型号、1个测试条件、1个性能指标、1个温度阈值）和 4 条关系。"

    print(f"\n[实体] 共 {len(result.entities)} 个：")
    for ent in result.entities:
        print(f"  - [{ent['entity_type']}] {ent['name']}")
        attrs = ent["attributes"]
        for k, v in list(attrs.items())[:4]:
            print(f"      {k} = {v}")

    print(f"\n[关系] 共 {len(result.relations)} 个：")
    for rel in result.relations:
        print(f"  - {rel['source_name']} --[{rel['relation_type']}]--> {rel['target_name']}")

    print(f"\n[摘要] {result.summary}")

    print("\n" + "=" * 60)
    print("  核心工程亮点解读")
    print("=" * 60)
    print("""
1. 【死循环防护】Step 6 检测到连续调用 search_knowledge_graph，
   主动发出策略建议，让模型切换到 write_to_graph，避免无限循环。

2. 【Schema 校验】Step 5 在写入前校验数值范围，
   避免了无效数据进入知识图谱。

3. 【图谱去重】Step 4 先查询图谱，已有实体则合并属性，
   避免重复抽取同一电池型号的不同属性。

4. 【多层观察】Agent 能根据文档内容动态调整下一步策略：
   文本 → 表格 → 图谱查询 → 写入，这是纯正则方案做不到的。
""")


if __name__ == "__main__":
    demo_extraction_flow()
