"""
cs_agent CLI 入口。
用法：
  python -m cs_agent.main                   # 交互式对话
  python -m cs_agent.main --query "G6续航多少"
  python -m cs_agent.main --build-index     # 重建 FAISS 索引
  python -m cs_agent.main --eval            # 运行评估
  python -m cs_agent.main --ui              # 启动 Gradio
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _interactive():
    from cs_agent.graph import chat
    print("\n新能源汽车智能客服 — 交互模式")
    print("输入 exit 退出 | 输入 eval 运行评估 | 输入 ui 启动界面\n")
    history = []
    session_id = "cli_session"
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！"); break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("再见！"); break
        if q.lower() == "eval":
            from cs_agent.evaluation.run_eval import run_eval
            run_eval(); continue
        if q.lower() == "ui":
            from cs_agent.app import launch_ui
            launch_ui(server_name="0.0.0.0", server_port=7860); break

        result = chat(q, session_id=session_id, history=history)
        history = result["messages"]
        print(f"\n小P：{result['answer']}")
        if result["intent"]:
            print(f"    [意图:{result['intent']} | 实体:{[e['text'] for e in result['entities']]}]")
        if result["ticket_id"]:
            print(f"    [工单:{result['ticket_id']}]")
        print()


def main():
    parser = argparse.ArgumentParser(description="新能源汽车智能客服 Multi-Agent 系统")
    parser.add_argument("--query", "-q", type=str, default="", help="单次查询")
    parser.add_argument("--build-index", action="store_true", help="重建 FAISS 向量索引")
    parser.add_argument("--eval", action="store_true", help="运行评估")
    parser.add_argument("--ui", action="store_true", help="启动 Gradio Web UI")
    args = parser.parse_args()

    if args.build_index:
        from cs_agent.tools.rag_tool import rebuild_index
        print("正在重建 FAISS 索引...")
        rebuild_index()
        print("索引构建完成！")
        return

    if args.eval:
        from cs_agent.evaluation.run_eval import run_eval
        run_eval()
        return

    if args.ui:
        from cs_agent.app import launch_ui
        launch_ui(server_name="0.0.0.0", server_port=7860)
        return

    if args.query:
        from cs_agent.graph import chat
        result = chat(args.query)
        print(f"\n问：{args.query}")
        print(f"答：{result['answer']}")
        print(f"意图：{result['intent']} | 实体：{[e['text'] for e in result['entities']]}")
        if result["ticket_id"]:
            print(f"工单：{result['ticket_id']}")
        return

    _interactive()


if __name__ == "__main__":
    main()
