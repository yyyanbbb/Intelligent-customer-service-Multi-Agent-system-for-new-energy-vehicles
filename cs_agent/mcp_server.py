"""
MCP server for both the legacy cs_agent flow and the new task_agent runtime.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
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
        "Run the legacy customer-service agent flow and return its answer.",
        {
            "query": {"type": "string", "description": "User question"},
            "session_id": {"type": "string", "description": "Session ID", "default": "mcp_user"},
        },
        ["query"],
    ),
    _make_tool(
        "rag_search",
        "Search the EV knowledge base with the legacy hybrid RAG stack.",
        {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Result count", "default": 5},
        },
        ["query"],
    ),
    _make_tool(
        "extract_entities",
        "Extract EV-domain entities such as models, brands, faults, and features.",
        {"text": {"type": "string", "description": "Input text"}},
        ["text"],
    ),
    _make_tool(
        "create_ticket",
        "Create a legacy aftersales ticket payload.",
        {
            "description": {"type": "string", "description": "Issue description"},
            "vehicle_model": {"type": "string", "description": "Vehicle model"},
            "components": {"type": "array", "items": {"type": "string"}, "description": "Components"},
            "faults": {"type": "array", "items": {"type": "string"}, "description": "Fault types"},
        },
        ["description"],
    ),
    _make_tool(
        "start_task",
        "Start a new task-agent workflow and return task_id, state, and current result.",
        {
            "query": {"type": "string", "description": "Task or goal"},
            "session_id": {"type": "string", "description": "Session ID", "default": "task_mcp_user"},
        },
        ["query"],
    ),
    _make_tool(
        "continue_task",
        "Continue an existing task by providing more user input.",
        {
            "task_id": {"type": "string", "description": "Task ID"},
            "user_input": {"type": "string", "description": "New user input"},
        },
        ["task_id", "user_input"],
    ),
    _make_tool(
        "confirm_task_action",
        "Approve or reject a pending task action such as booking or ticket submission.",
        {
            "task_id": {"type": "string", "description": "Task ID"},
            "confirmation_id": {"type": "string", "description": "Pending confirmation ID"},
            "approved": {"type": "boolean", "description": "Whether to approve the action"},
        },
        ["task_id", "confirmation_id", "approved"],
    ),
    _make_tool(
        "get_task_status",
        "Get the latest state, pending questions, pending confirmations, and result for a task.",
        {"task_id": {"type": "string", "description": "Task ID"}},
        ["task_id"],
    ),
    _make_tool(
        "get_memory",
        "Get stored user memory from the legacy cs_agent subsystem.",
        {"session_id": {"type": "string", "description": "Session ID"}},
        ["session_id"],
    ),
    _make_tool(
        "list_recent_traces",
        "List recent legacy agent traces for debugging.",
        {"n": {"type": "integer", "description": "Trace count", "default": 10}},
        [],
    ),
]


def _handle_ask_ev_agent(args: dict) -> dict:
    dependency_error = _legacy_dependency_error()
    if dependency_error:
        return _legacy_agent_fallback(args["query"], dependency_error)
    from cs_agent.graph import chat

    try:
        result = chat(query=args["query"], session_id=args.get("session_id", "mcp_user"))
    except Exception as exc:
        return _legacy_agent_fallback(args["query"], f"{type(exc).__name__}: {exc}")
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


def _legacy_dependency_error() -> str:
    try:
        if importlib.util.find_spec("langgraph.checkpoint.sqlite") is None:
            return "missing dependency: langgraph.checkpoint.sqlite"
    except ModuleNotFoundError as exc:
        return f"missing dependency: {exc.name}"
    return ""


def _legacy_agent_fallback(query: str, reason: str) -> dict:
    from task_agent.parsing import classify_task

    intent = classify_task(query)
    return {
        "answer": "当前旧版客服图不可用。任务型 Agent 工具仍可用；请优先调用 start_task/continue_task/confirm_task_action 完成闭环任务。",
        "intent": intent,
        "entities": [],
        "sources": [],
        "ticket_id": "",
        "structured": {},
        "cache_hit": False,
        "backend": "fallback",
        "elapsed_ms": 0,
        "error_code": "legacy_unavailable",
        "error": reason,
    }


def _handle_rag_search(args: dict) -> dict:
    from cs_agent.tools.hybrid_rag import hybrid_retrieve

    hits = hybrid_retrieve(args["query"], top_k=int(args.get("top_k", 5)))
    return {"results": [{"content": hit["content"], "source": hit["source"], "score": hit.get("rerank_score", hit.get("score", 0))} for hit in hits]}


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


def _handle_start_task(args: dict) -> dict:
    from task_agent.service import start_task

    return start_task(args["query"], session_id=args.get("session_id", "task_mcp_user"))


def _handle_continue_task(args: dict) -> dict:
    from task_agent.service import continue_task

    return continue_task(args["task_id"], user_input=args["user_input"])


def _handle_confirm_task_action(args: dict) -> dict:
    from task_agent.service import confirm_task_action

    return confirm_task_action(
        task_id=args["task_id"],
        confirmation_id=args["confirmation_id"],
        approved=bool(args["approved"]),
    )


def _handle_get_task_status(args: dict) -> dict:
    from task_agent.service import get_task_status

    return get_task_status(args["task_id"])


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
    "start_task": _handle_start_task,
    "continue_task": _handle_continue_task,
    "confirm_task_action": _handle_confirm_task_action,
    "get_task_status": _handle_get_task_status,
    "get_memory": _handle_get_memory,
    "list_recent_traces": _handle_list_traces,
}


def _handle_request(req: dict) -> dict | None:
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ev-task-agent", "version": "3.0.0"},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        tool_name = req.get("params", {}).get("name", "")
        tool_args = req.get("params", {}).get("arguments", {})
        handler = _HANDLERS.get(tool_name)
        if handler is None:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
        try:
            result = handler(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]},
            }
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(exc)}}

    if method == "notifications/initialized":
        return None

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main() -> None:
    import io

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
