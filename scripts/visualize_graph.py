"""
一键输出 LangGraph Agent 图结构（Mermaid 格式）。
用法：python scripts/visualize_graph.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs_agent.graph import get_graph_mermaid

if __name__ == "__main__":
    mermaid = get_graph_mermaid()
    print(mermaid)

    out = Path(__file__).parent.parent / "cs_agent" / "knowledge" / "graph.mmd"
    out.write_text(mermaid, encoding="utf-8")
    print(f"\n已保存到 {out}")
