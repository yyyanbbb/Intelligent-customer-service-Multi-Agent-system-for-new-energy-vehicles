"""
Tools 模块 — 暴露给 Agent 的工具集及其 OpenAI tool calling 定义
每个工具都有两部分：
1. OpenAI tool calling 格式的 schema（供模型理解工具签名）
2. 实际执行的 Python 函数
"""
from ev_ner_agent.tools.pdf_extractor import extract_pdf_text
from ev_ner_agent.tools.table_extractor import extract_tables
from ev_ner_agent.tools.schema_validator import validate_schema
from ev_ner_agent.tools.kg_searcher import search_knowledge_graph, write_to_graph


# ---------------------------------------------------------------------------
# OpenAI Tool Calling Schema（符合 tool_use 规范）
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "extract_pdf_text",
            "description": "从 PDF 文件中提取文本内容并进行清洗分段。可按页范围提取，并自动预判文档类型（诊断报告/用户手册/维修记录）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "PDF 文件的绝对路径或相对路径",
                    },
                    "start_page": {
                        "type": "integer",
                        "description": "起始页码（从 1 开始），不传则从首页开始",
                    },
                    "end_page": {
                        "type": "integer",
                        "description": "结束页码，不传则到最后一页",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_tables",
            "description": "从 PDF 文件中提取表格数据，转化为结构化列表格式。适合提取电池测试报告中的性能数据表、循环寿命表等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "PDF 文件的路径",
                    },
                    "pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "指定要提取表格的页码列表，不传则提取全部页",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_schema",
            "description": "校验抽取出的数据是否符合预定义的 JSON Schema 规范。包括字段类型、必填字段、数值范围、枚举值校验。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "要校验的数据对象（JSON 格式）",
                    },
                    "schema_type": {
                        "type": "string",
                        "description": "实体类型（如 BatteryModel、TestCondition、PerformanceMetric、DegradationCurve），不传则校验全部字段",
                    },
                },
                "required": ["data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_graph",
            "description": "在本地知识图谱中检索相关实体。支持按关键词搜索和按实体类型过滤，返回关联实体及其关系。可用于跨文档去重和上下文补全。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词（实体名称或属性值）",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "按实体类型过滤（如 BatteryModel、TestCondition 等），不传则搜索所有类型",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_to_graph",
            "description": "将抽取出的实体和关系批量写入知识图谱。写入时会自动去重和合并相似实体。支持构建结构化知识图谱用于后续查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "实体列表，每个实体包含 entity_type、name、attributes 等字段",
                    },
                    "relations": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "关系列表，每个关系包含 source_name、target_name、relation_type",
                    },
                    "source": {
                        "type": "string",
                        "description": "数据来源文档名称",
                    },
                },
                "required": ["entities"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 工具函数映射表（name -> callable）
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, callable] = {
    "extract_pdf_text": extract_pdf_text,
    "extract_tables": extract_tables,
    "validate_schema": validate_schema,
    "search_knowledge_graph": search_knowledge_graph,
    "write_to_graph": write_to_graph,
}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    根据工具名称和参数执行对应的工具函数。
    所有工具的返回结果都是 str，Agent 会将其作为 observation 继续推理。
    """
    if tool_name not in TOOL_FUNCTIONS:
        return f"❌ 未知工具: {tool_name}，可用工具: {list(TOOL_FUNCTIONS.keys())}"

    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**arguments)
        return str(result)
    except TypeError as e:
        return f"❌ 工具 {tool_name} 参数错误: {e}，请检查参数格式"
    except FileNotFoundError:
        return f"❌ 工具 {tool_name} 执行失败：文件未找到，请检查路径是否正确"
    except Exception as e:
        return f"❌ 工具 {tool_name} 执行失败: {e}"


__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_FUNCTIONS",
    "execute_tool",
    "extract_pdf_text",
    "extract_tables",
    "validate_schema",
    "search_knowledge_graph",
    "write_to_graph",
]
