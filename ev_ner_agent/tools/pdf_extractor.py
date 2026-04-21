"""PDF 解析工具。用 PyMuPDF 提取文本，按段落 + 长度切成 chunks，并预判文档类型（诊断报告/用户手册/维修记录）。"""
from __future__ import annotations

import re
import logging
from typing import Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _get_fitz():
    """延迟导入 PyMuPDF，避免顶层 import 失败。"""
    try:
        import fitz
        return fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) 未安装，请运行: pip install PyMuPDF"
        )

DOC_TYPE_PATTERNS = {
    "diagnostic_report": [
        r"(?:电池| Battery | BMS).*?(?:诊断|诊断报告|report)",
        r"(?:SOC|SOH|容量).*?\d+",
        r"(?:温度|温度阈值)",
    ],
    "user_manual": [
        r"(?:用户手册|使用说明|操作指南)",
        r"(?:警告|注意|CAUTION|WARNING).*?℃",
    ],
    "maintenance_record": [
        r"(?:维修|保养|维护).*?(?:记录|报告)",
        r"(?:日期|时间).*?\d{4}[-/]\d{2}[-/]\d{2}",
    ],
}


@dataclass
class DocumentChunk:
    page_num: int
    content: str
    doc_type: str


class PDFExtractor:
    """
    PDF 解析器，支持：
    - 全文档文本提取 + 清洗
    - 按页范围提取
    - 文档类型预判（诊断报告 / 用户手册 / 维修记录）
    - 文本分段（按段落 + 长度截断）
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract(
        self,
        path: str,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> dict:
        """
        提取 PDF 内容，返回结构化结果。

        返回字段：
        - page_count: 总页数
        - doc_type: 预判的文档类型
        - chunks: 分段后的文本块列表
        - metadata: 文档元信息（标题、作者等）
        """
        try:
            doc = _get_fitz().open(path)
        except Exception as e:
            logger.error(f"无法打开 PDF 文件: {path}, 错误: {e}")
            return {"error": f"无法打开文件: {e}", "chunks": []}

        total_pages = len(doc)
        start = start_page if start_page is not None else 0
        end = end_page if end_page is not None else total_pages

        all_text = []
        page_texts = []

        for i in range(start, min(end, total_pages)):
            page = doc[i]
            text = page.get_text("text")
            text = self._clean_text(text)
            if text.strip():
                all_text.append(f"[第 {i + 1} 页]\n{text}")
                page_texts.append((i + 1, text))

        doc.close()

        chunks = self._make_chunks(page_texts)
        doc_type = self._detect_doc_type("\n".join(all_text))

        logger.info(f"PDF 提取完成: {path}, 页数={total_pages}, 类型={doc_type}, 分段数={len(chunks)}")

        return {
            "page_count": total_pages,
            "doc_type": doc_type,
            "chunks": chunks,
            "metadata": {
                "title": self._extract_title(all_text),
                "page_range": f"{start + 1}-{min(end, total_pages)}",
            },
        }

    def _clean_text(self, text: str) -> str:
        """清洗 PDF 提取的原始文本。"""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"([。！？；\]】」』])\s*\n\s*([（\(【「『])", r"\1\2", text)
        text = re.sub(r"([。！？；\]】」』])\s*\n", r"\1\n", text)
        text = text.strip()
        return text

    def _make_chunks(self, page_texts: list[tuple[int, str]]) -> list[DocumentChunk]:
        """将文本按段落和长度切分为 chunks。"""
        chunks = []
        current = ""
        current_page = 1

        for page_num, text in page_texts:
            paragraphs = re.split(r"\n{1,2}", text)
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) + 1 <= self.chunk_size:
                    current += ("\n" if current else "") + para
                    current_page = page_num
                else:
                    if current.strip():
                        chunks.append(DocumentChunk(current_page, current.strip(), ""))
                    overlap_text = para[-self.chunk_overlap:] if len(para) > self.chunk_overlap else para
                    current = overlap_text
                    current_page = page_num

        if current.strip():
            chunks.append(DocumentChunk(current_page, current.strip(), ""))

        for chunk in chunks:
            chunk.doc_type = self._detect_doc_type(chunk.content)

        return [
            {
                "page_num": c.page_num,
                "content": c.content,
                "doc_type": c.doc_type,
            }
            for c in chunks
        ]

    def _detect_doc_type(self, text: str) -> str:
        """基于关键词匹配预判文档类型。"""
        scores: dict[str, float] = {
            "diagnostic_report": 0.0,
            "user_manual": 0.0,
            "maintenance_record": 0.0,
        }
        for dtype, patterns in DOC_TYPE_PATTERNS.items():
            for pat in patterns:
                matches = re.findall(pat, text, re.IGNORECASE)
                scores[dtype] += len(matches) * 0.5

        if max(scores.values()) == 0:
            return "unknown"
        return max(scores, key=scores.get)  # type: ignore

    def _extract_title(self, all_text: list[str]) -> str:
        """从首行提取文档标题。"""
        if not all_text:
            return "Unknown"
        first_lines = "\n".join(all_text[:3])
        title_match = re.search(r"^[^@\n]+", first_lines)
        return title_match.group(0).strip()[:100] if title_match else "Unknown"


def extract_pdf_text(path: str, start_page: int | None = None, end_page: int | None = None) -> str:
    """
    暴露给 Agent 的工具函数。
    符合 OpenAI tool calling 的参数规范。
    """
    extractor = PDFExtractor()
    result = extractor.extract(path, start_page, end_page)
    if "error" in result:
        return f"PDF 解析失败: {result['error']}"

    chunks = result["chunks"]
    if not chunks:
        return "PDF 中未提取到有效文本内容。"

    summary_parts = [
        f"文档类型: {result['doc_type']}",
        f"总页数: {result['page_count']}",
        f"分段数: {len(chunks)}",
        "",
        "=== 文本内容分段 ===",
    ]
    for i, chunk in enumerate(chunks[:20], 1):
        summary_parts.append(f"\n--- 段落 {i} (第 {chunk['page_num']} 页, 类型={chunk['doc_type']}) ---")
        summary_parts.append(chunk["content"][:800])

    if len(chunks) > 20:
        summary_parts.append(f"\n... (共 {len(chunks)} 段，以上仅展示前 20 段)")

    return "\n".join(summary_parts)
