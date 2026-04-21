"""
MCP Server：将 cs_agent 核心工具暴露为 Model Context Protocol server。
Claude Code / Cursor / 任意 MCP 客户端可直接调用。

启动：
    python -m cs_agent.mcp_server
    # 或通过 .mcp.json 配置后自动启动

工具列表：
  - ask_ev_agent       完整 multi-agent 对话（路由+RAG+记忆）
  - rag_search         直接调用混合 RAG 检索
  - extract_entities   车辆领域 NER
  - create_ticket      生成售后工单
  - get_memory         获取用户会话记忆
  - list_recent_traces 查看最近调用 trace
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


TOOLS = [
    _make_tool(
        "ask_ev_agent",
        "向新能源汽车智能客服发送问题，自动路由到购车/售后/车型问答节点，返回回答和结构化分析",
        {
            "query": {"type": "string", "description": "用户问题"},
            "session_id": {"type": "string", "description": "会话ID，用于多轮记忆，默认 'mcp_user'"},
        },
        ["query"],
    ),
    _make_tool(
        "rag_search",
        "在 181+ 车型知识库中混合检索（BM25+向量+重排），返回最相关的知识片段",
        {
            "query": {"type": "string", "description": "检索查询"},
            "top_k": {"type": "integer", "description": "返回条数，默认 5", "default": 5},
        },
        ["query"],
    ),
    _make_tool(
        "extract_entities",
        "从文本中抽取新能源汽车领域实体（车型/品牌/部件/故障/功能/预算）",
        {"text": {"type": "string", "description": "待抽取文本"}},
        ["text"],
    ),
    _make_tool(
        "create_ticket",
        "创建售后服务工单",
        {
            "description": {"type": "string", "description": "故障描述"},
            "vehicle_model": {"type": "string", "description": "车型名"},
            "components": {"type": "array", "items": {"type": "string"}, "description": "涉及部件"},
            "faults": {"type": "array", "items": {"type": "string"}, "description": "故障类型"},
        },
        ["description"],
    ),
    _make_tool(
        "get_memory",
        "获取指定会话的长期记忆（用户偏好、预算、关注车型等）",
        {"session_id": {"type": "string", "description": "会话ID"}},
        ["session_id"],
    ),
    _make_tool(
        "list_recent_traces",
        "查看最近 N 条 agent 调用 trace，用于调试和可观测性",
        {"n": {"type": "integer", "description": "条数，默认 10", "default": 10}},
        [],
    ),
]


def _handle_ask_ev_agent(args: dict) -> dict:
    from cs_agent.graph import chat
    result = chat(
        query=args["query"],
        session_id=args.get("session_id", "mcp_user"),
    )
    return {
        "answer": result["answer"],
        "intent": result["intent"],
        "entities": result["entities"],
        "sources": result["sources"],
        "ticket_id": result.get("ticket_id", ""),
        "structured": result.get("structured", {}),
        "cache_hit": result.get("cache_hit", False),
        "backend": result.get("backend", ""),
        "elapsed_ms": result.get("elapsed_ms", 0),
    }


def _handle_rag_search(args: dict) -> dict:
    from cs_agent.tools.hybrid_rag import hybrid_retrieve
    hits = hybrid_retrieve(args["query"], top_k=int(args.get("top_k", 5)))
    return {"results": [{"content": h["content"], "source": h["source"], "score": h.get("rerank_score", h.get("score", 0))} for h in hits]}


def _handle_extract_entities(args: dict) -> dict:
    from cs_agent.tools.ner_tool import extract_entities
    return {"entities": extract_entities(args["text"])}


def _handle_create_ticket(args: dict) -> dict:
    from cs_agent.tools.ticket_tool import create_ticket
    return create_ticket(
        description=args["description"],
        vehicle_model=args.get("vehicle_model", ""),
        components=args.get("components", []),
        faults=args.get("faults", []),
    )


def _handle_get_memory(args: dict) -> dict:
    from cs_agent.memory import load_memory
    return load_memory(args["session_id"])


def _handle_list_traces(args: dict) -> dict:
    from cs_agent.observability import get_recent_traces
    return {"traces": get_recent_traces(int(args.get("n", 10)))}


_HANDLERS = {
    "ask_ev_agent": _handle_ask_ev_agent,
    "rag_search": _handle_rag_search,
    "extract_entities": _handle_extract_entities,
    "create_ticket": _handle_create_ticket,
    "get_memory": _handle_get_memory,
    "list_recent_traces": _handle_list_traces,
}


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle_request(req: dict) -> dict | None:
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ev-cs-agent", "version": "2.0.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        tool_name = req.get("params", {}).get("name", "")
        tool_args = req.get("params", {}).get("arguments", {})
        handler = _HANDLERS.get(tool_name)
        if not handler:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
        try:
            result = handler(tool_args)
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]},
            }
        except Exception as e:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}}

    if method == "notifications/initialized":
        return None

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main() -> None:
    """stdio transport MCP server 主循环。"""
    import io
    # Ensure UTF-8 I/O on all platforms (especially Windows with GBK default)
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle_request(req)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()


if __name__ == "__main__":
    main()
