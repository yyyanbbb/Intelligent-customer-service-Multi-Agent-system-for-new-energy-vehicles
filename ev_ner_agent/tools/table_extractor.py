"""
表格抽取工具
使用 tabula-py 从 PDF 中提取表格数据，转化为结构化列表。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    PDF 表格抽取器，将表格转为 JSON 列表格式。
    """

    def extract_tables(self, path: str, pages: list[int] | None = None) -> dict:
        """
        从 PDF 中提取表格。

        参数：
        - path: PDF 文件路径
        - pages: 指定页码列表（None 表示全部页）

        返回：
        - tables: 表格列表，每个表格包含 page、rows、col_count
        """
        try:
            import tabula
        except ImportError:
            return {
                "error": "tabula-py 未安装，请运行: pip install tabula-py",
                "tables": [],
            }

        try:
            dfs = tabula.read_pdf(
                path,
                pages=pages if pages else "all",
                lattice=True,
                stream=True,
                guess=True,
                format="JSON",
            )
        except Exception as e:
            logger.warning(f"tabula 提取失败，尝试备用模式: {e}")
            try:
                dfs = tabula.read_pdf(
                    path,
                    pages=pages if pages else "all",
                    guess=True,
                    format="JSON",
                )
            except Exception as e2:
                return {"error": f"表格提取失败: {e2}", "tables": []}

        tables = []
        for i, df_data in enumerate(dfs):
            if not df_data:
                continue
            rows = df_data if isinstance(df_data, list) else df_data.to_dict("records")
            if isinstance(df_data, list) and len(df_data) > 0 and isinstance(df_data[0], dict):
                headers = list(df_data[0].keys())
            elif hasattr(df_data, "columns"):
                headers = list(df_data.columns)
            else:
                headers = []

            tables.append({
                "table_index": i,
                "headers": headers,
                "rows": rows[:100],
                "row_count": len(rows),
            })

        logger.info(f"从 {path} 中提取到 {len(tables)} 张表格")
        return {"tables": tables, "table_count": len(tables)}


def extract_tables(path: str, pages: list[int] | None = None) -> str:
    """
    暴露给 Agent 的工具函数。
    """
    extractor = TableExtractor()
    result = extractor.extract_tables(path, pages)

    if "error" in result:
        return f"表格提取失败: {result['error']}"

    tables = result.get("tables", [])
    if not tables:
        return "未从 PDF 中提取到任何表格。"

    parts = [f"共提取到 {len(tables)} 张表格：\n"]
    for tbl in tables[:10]:
        parts.append(f"\n--- 表格 {tbl['table_index'] + 1} (行数={tbl['row_count']}) ---")
        if tbl["headers"]:
            parts.append("表头: " + " | ".join(str(h) for h in tbl["headers"]))
        for row in tbl["rows"][:15]:
            if isinstance(row, dict):
                vals = [str(row.get(h, "")) for h in tbl["headers"]]
                parts.append(" | ".join(vals))
            else:
                parts.append(str(row))
        if tbl["row_count"] > 15:
            parts.append(f"... (共 {tbl['row_count']} 行，以上仅展示前 15 行)")

    return "\n".join(parts)
