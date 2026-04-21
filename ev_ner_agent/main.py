"""
EV NER Agent 主入口。
命令行：python -m ev_ner_agent.main --query "提取电池型号" --doc data/sample.pdf
Python API：from ev_ner_agent.main import run_extraction
"""
from __future__ import annotations

import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Any

from ev_ner_agent.model_client import create_client
from ev_ner_agent.agent import ReActAgent
from ev_ner_agent.tools.kg_searcher import get_knowledge_graph, search_knowledge_graph
from ev_ner_agent.tools.pdf_extractor import PDFExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_doc_context(doc_paths: list[str]) -> str:
    """构建文档上下文，将 PDF 路径列表转为字符串描述。"""
    if not doc_paths:
        return ""
    parts = ["用户提供了以下文档：\n"]
    for path in doc_paths:
        parts.append(f"  - {path}")
    parts.append("\n请先用 extract_pdf_text 工具提取第一个文档的内容。")
    return "\n".join(parts)


def run_extraction(
    query: str,
    doc_paths: list[str] | None = None,
    model: str = "qwen2.5:7b",
    provider: str = "ollama",
    base_url: str | None = None,
    max_steps: int = 15,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    output_path: str | None = None,
    show_history: bool = True,
) -> dict[str, Any]:
    """
    运行抽取流程的 Python API。

    参数：
    - query: 用户的抽取指令
    - doc_paths: PDF 文件路径列表
    - model: 模型名称（默认 qwen2.5:7b）
    - provider: 部署方式（ollama / vllm / lmstudio）
    - base_url: API base URL（不传则用 provider 默认值）
    - max_steps: 最大 Agent 步数
    - max_tokens: 最大输出 token 数
    - temperature: 生成温度
    - output_path: 结果输出文件路径
    - show_history: 是否打印 Agent 推理过程

    返回：
    - 包含 entities、relations、summary、step_history 的字典
    """
    logger.info(f"开始抽取 | query={query!r} | model={model} | provider={provider}")
    logger.info(f"文档列表: {doc_paths}")

    # 初始化模型客户端
    if base_url:
        client = create_client(provider=provider, model=model, base_url=base_url)
    else:
        client = create_client(provider=provider, model=model)

    # 预检查文档路径
    doc_context = ""
    if doc_paths:
        valid_paths = []
        for path in doc_paths:
            p = Path(path)
            if p.exists():
                valid_paths.append(str(p.absolute()))
                doc_context += build_doc_context(valid_paths)
            else:
                logger.warning(f"文档不存在: {path}，跳过")

    # 查知识图谱获取上下文
    graph_context = ""
    if query:
        graph_result = search_knowledge_graph(query)
        graph_context = graph_result[:1500]

    # 运行 ReAct Agent
    agent = ReActAgent(
        model_client=client,
        max_steps=max_steps,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = agent.run(
        user_query=query,
        doc_context=doc_context,
        graph_context=graph_context,
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("  抽取结果")
    print("=" * 60)

    if result.get("entities"):
        print(f"\n[实体] 共 {len(result['entities'])} 个：")
        for ent in result["entities"]:
            print(f"  - [{ent.get('entity_type', '?')}] {ent.get('name', '?')}")
            attrs = ent.get("attributes", {})
            if attrs:
                for k, v in list(attrs.items())[:4]:
                    print(f"      {k} = {v}")

    if result.get("relations"):
        print(f"\n[关系] 共 {len(result['relations'])} 个：")
        for rel in result["relations"]:
            print(f"  - {rel.get('source_name', '?')} --[{rel.get('relation_type', '?')}]--> {rel.get('target_name', '?')}")

    if result.get("summary"):
        print(f"\n[摘要]\n{result['summary']}")

    print("\n" + "-" * 60)
    meta = result.get("metadata", {})
    print(f"[元信息]")
    print(f"  总步数: {meta.get('total_steps', 0)}")
    print(f"  总耗时: {meta.get('total_time', 0):.3f}s")
    print(f"  错误数: {meta.get('error_count', 0)}")
    print(f"  最终状态: {meta.get('final_state', 'unknown')}")

    if show_history:
        print("\n" + "-" * 60)
        print("[推理过程]")
        for step in result.get("step_history", []):
            print(f"\n  Step {step['step']} ({step['elapsed']:.2f}s)")
            print(f"    Thought: {step['thought'][:100]}")
            if step["action"]:
                print(f"    Action: {step['action']}({json.dumps(step['tool_args'], ensure_ascii=False)[:80]})")
            if step["observation"]:
                print(f"    Observation: {step['observation'][:150]}")
            if step["error"]:
                print(f"    Error: {step['error']}")

    # 保存结果
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存至: {output_path}")

    return result


def interactive_mode():
    """交互式模式，持续接收用户查询。"""
    print("\n" + "=" * 60)
    print("  EV NER Agent — 交互式抽取模式")
    print("  输入 'exit' 或 'quit' 退出")
    print("  输入 'graph' 查看当前知识图谱状态")
    print("  输入 'reset' 清空知识图谱")
    print("  输入 'load <path>' 加载文档")
    print("  输入 'model <name>' 设置模型名称")
    print("  输入 'provider <name>' 设置部署方式（ollama/vllm/lmstudio）")
    print("=" * 60 + "\n")

    doc_paths: list[str] = []
    model_name: str = "qwen2.5:7b"
    provider: str = "ollama"
    base_url: str | None = None

    while True:
        try:
            user_input = input("\n[用户] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("exit", "quit", "q"):
            print("再见！")
            break

        if cmd == "graph":
            kg = get_knowledge_graph()
            stats = kg.get_stats()
            print(f"\n[知识图谱状态] {stats}")
            continue

        if cmd == "reset":
            import ev_ner_agent.tools.kg_searcher as kg_module
            kg_module._global_kg = None
            doc_paths.clear()
            print("[知识图谱已重置，文档列表已清空]")
            continue

        if user_input.lower().startswith("load "):
            path = user_input[5:].strip()
            doc_paths.append(path)
            print(f"[已加载文档] {path} (共 {len(doc_paths)} 个)")
            continue

        if user_input.lower().startswith("model "):
            model_name = user_input[6:].strip()
            print(f"[模型已设置为] {model_name}")
            continue

        if user_input.lower().startswith("provider "):
            provider = user_input[9:].strip()
            print(f"[部署方式已设置为] {provider}")
            continue

        run_extraction(
            query=user_input,
            doc_paths=doc_paths if doc_paths else None,
            model=model_name,
            provider=provider,
            base_url=base_url,
            show_history=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="EV NER Agent — 面向新能源汽车领域的多源异构文档智能抽取系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本用法
  python -m ev_ner_agent.main --query "提取所有电池型号和测试条件" --doc data/battery_report.pdf

  # 交互模式
  python -m ev_ner_agent.main --interactive

  # 使用 vLLM 后端
  python -m ev_ner_agent.main --query "提取温度阈值" --provider vllm --model hermes-3-8b

  # 指定 API 地址
  python -m ev_ner_agent.main --query "提取衰减曲线" --base-url http://localhost:8000/v1

  # 指定模型和参数
  python -m ev_ner_agent.main --query "提取所有实体" --model qwen2.5:14b --max-steps 20 --max-tokens 8192
        """,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="抽取指令，如：'提取所有电池型号和测试条件'",
    )
    parser.add_argument(
        "--doc", "-d",
        type=str,
        action="append",
        dest="docs",
        help="PDF 文档路径，可多次指定",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen2.5:7b",
        help="模型名称（默认: qwen2.5:7b）",
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="ollama",
        choices=["ollama", "vllm", "lmstudio"],
        help="模型部署方式（默认: ollama）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL（覆盖 provider 默认值）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="最大 Agent 步数（默认: 15）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="最大输出 token 数（默认: 4096）",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.1,
        help="生成温度（默认: 0.1）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="结果输出文件路径（JSON 格式）",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="不显示推理过程",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互式模式",
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if not args.query:
        parser.print_help()
        print("\n错误：请提供 --query 参数，或使用 --interactive 进入交互模式。")
        sys.exit(1)

    run_extraction(
        query=args.query,
        doc_paths=args.docs,
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_path=args.output,
        show_history=not args.no_history,
    )


if __name__ == "__main__":
    main()
