from __future__ import annotations
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from cs_agent.graph import get_graph, resume_after_safety, get_graph_mermaid
from cs_agent.state import CSState
from cs_agent.memory import clear_memory
from cs_agent.observability import get_recent_traces, cache_clear
from cs_agent.llm_client import get_active_backend

_session_messages: dict[str, list] = {}

# Pre-initialize graph at import time (before Gradio's async event loop starts)
# so AsyncSqliteSaver can be set up with asyncio.run() without conflict.
_graph = get_graph()

_INTENT_LABELS = {
    "vehicle_qa": "车型问答",
    "aftersales": "售后服务",
    "purchase": "购车咨询",
    "charging": "充电问题",
    "order_tracking": "订单交付",
    "complaint": "投诉反馈",
    "account": "账户服务",
    "insurance": "保险理赔",
    "test_drive": "试驾预约",
    "navigation": "导航/OTA",
    "roadside": "道路救援",
    "chitchat": "闲聊",
    "": "—",
}
_INTENT_ICONS = {
    "vehicle_qa": "◈",
    "aftersales": "◉",
    "purchase": "◎",
    "charging": "⚡",
    "order_tracking": "📦",
    "complaint": "⚠",
    "account": "👤",
    "insurance": "🛡",
    "test_drive": "🚗",
    "navigation": "🗺",
    "roadside": "🚨",
    "chitchat": "○",
    "": "·",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@300;400;600&family=DM+Mono:wght@300;400&display=swap');
:root {
    --bg: #f8f7f4; --surface: #ffffff; --surface-alt: #f2f1ee;
    --border: #e2e0db; --border-soft: #ede9e3;
    --text-primary: #1a1917; --text-secondary: #6b6760; --text-muted: #a09d97;
    --accent: #1a9e6e; --accent-light: #e8f5f0; --accent-dim: #a8d8c8;
    --danger: #c44b4b; --danger-light: #fdf0f0;
    --cache: #7c5cbf; --cache-light: #f0ecfa;
    --mono: 'DM Mono','JetBrains Mono','Fira Code',monospace;
    --serif: 'Noto Serif SC','Source Han Serif SC',Georgia,serif;
}
* { box-sizing: border-box; }
body { background: var(--bg) !important; font-family: 'PingFang SC','Hiragino Sans GB','Microsoft YaHei UI',sans-serif; color: var(--text-primary); }
.gradio-container { max-width: 1280px !important; padding: 0 !important; background: transparent !important; }
#app-header { padding: 32px 40px 20px; border-bottom: 1px solid var(--border); background: var(--surface); }
#app-header h1 { font-family: var(--serif); font-size: 22px; font-weight: 400; letter-spacing: .08em; color: var(--text-primary); margin: 0 0 4px; }
#app-header p { font-size: 11px; color: var(--text-muted); letter-spacing: .15em; text-transform: uppercase; margin: 0; font-family: var(--mono); }
#main-row { display: flex; gap: 0; min-height: 640px; background: var(--surface); border-bottom: 1px solid var(--border); }
#chat-col { flex: 1; border-right: 1px solid var(--border); display: flex; flex-direction: column; }
#chatbot { height: 480px !important; border-radius: 0 !important; border: none !important; background: var(--bg) !important; font-size: 14px; }
#chatbot .message-wrap { padding: 20px 28px !important; }
#chatbot .user-message { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 2px 12px 12px 12px !important; font-size: 13.5px; line-height: 1.7; box-shadow: none !important; }
#chatbot .bot-message { background: var(--accent-light) !important; border: 1px solid var(--accent-dim) !important; border-radius: 12px 2px 12px 12px !important; font-size: 13.5px; line-height: 1.7; box-shadow: none !important; }
#input-area { padding: 16px 20px; border-top: 1px solid var(--border); background: var(--surface); }
#msg-input textarea { border: 1px solid var(--border) !important; border-radius: 6px !important; background: var(--bg) !important; font-size: 13.5px !important; padding: 10px 14px !important; transition: border-color .2s; resize: none !important; }
#msg-input textarea:focus { border-color: var(--accent) !important; outline: none !important; box-shadow: 0 0 0 3px rgba(26,158,110,.08) !important; }
#send-btn { background: var(--accent) !important; border: none !important; border-radius: 6px !important; color: white !important; font-size: 13px !important; font-weight: 500; padding: 10px 18px !important; letter-spacing: .04em; transition: opacity .15s,transform .1s; white-space: nowrap; height: 40px; }
#send-btn:hover { opacity: .88; transform: translateY(-1px); }
#controls-row { padding: 8px 20px 12px; border-top: 1px solid var(--border-soft); background: var(--surface); display: flex; align-items: center; gap: 12px; }
#clear-btn { background: transparent !important; border: 1px solid var(--border) !important; color: var(--text-secondary) !important; font-size: 12px !important; border-radius: 4px !important; padding: 5px 12px !important; }
#clear-btn:hover { border-color: var(--danger) !important; color: var(--danger) !important; background: var(--danger-light) !important; }
#session-box input { border: 1px solid var(--border) !important; border-radius: 4px !important; background: var(--bg) !important; font-size: 12px !important; font-family: var(--mono) !important; height: 30px !important; }
#examples-container { padding: 12px 20px; background: var(--bg); border-top: 1px solid var(--border-soft); }
.examples button { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 4px !important; color: var(--text-secondary) !important; font-size: 12px !important; padding: 5px 11px !important; }
.examples button:hover { border-color: var(--accent) !important; color: var(--accent) !important; background: var(--accent-light) !important; }
#info-panel { width: 340px; min-width: 300px; max-width: 360px; padding: 24px 20px; background: var(--surface); overflow-y: auto; }
#info-panel-title { font-family: var(--mono); font-size: 10px; color: var(--text-muted); letter-spacing: .2em; text-transform: uppercase; padding-bottom: 14px; border-bottom: 1px solid var(--border-soft); margin-bottom: 18px; }
.info-section-label { font-size: 10px; font-family: var(--mono); color: var(--text-muted); letter-spacing: .15em; text-transform: uppercase; margin-bottom: 6px; }
#intent-display p { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 500; background: var(--accent-light); color: var(--accent); border: 1px solid var(--accent-dim); padding: 4px 12px; border-radius: 3px; font-family: var(--mono); }
#cache-badge p { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; background: var(--cache-light); color: var(--cache); border: 1px solid #c9b8ef; padding: 3px 10px; border-radius: 3px; font-family: var(--mono); }
#entities-out table { width: 100%; border-collapse: collapse; font-size: 12px; }
#entities-out td, #entities-out th { padding: 5px 8px; border-bottom: 1px solid var(--border-soft); text-align: left; }
#entities-out th { font-family: var(--mono); font-size: 10px; color: var(--text-muted); font-weight: 400; text-transform: uppercase; }
#entities-out code { background: var(--surface-alt,#f2f1ee); border: 1px solid var(--border-soft); padding: 1px 5px; border-radius: 3px; font-family: var(--mono); font-size: 11.5px; }
#sources-out ul { padding-left: 0; margin: 0; list-style: none; }
#sources-out li { font-family: var(--mono); font-size: 11.5px; color: var(--text-secondary); padding: 3px 0; border-bottom: 1px solid var(--border-soft); }
#sources-out code { color: var(--accent); background: var(--accent-light); padding: 1px 5px; border-radius: 2px; font-family: var(--mono); font-size: 11px; }
#ticket-out code { font-family: var(--mono); font-size: 11.5px; color: var(--danger); background: var(--danger-light); border: 1px solid #f0d0d0; padding: 2px 7px; border-radius: 3px; }
#struct-out { font-size: 12px; font-family: var(--mono); background: var(--surface-alt,#f2f1ee); border-radius: 4px; padding: 10px 12px; border: 1px solid var(--border-soft); white-space: pre-wrap; }
#trace-out { font-size: 11px; font-family: var(--mono); color: var(--text-muted); background: var(--bg); border-radius: 4px; padding: 8px 10px; border: 1px solid var(--border-soft); white-space: pre-wrap; max-height: 120px; overflow-y: auto; }
#status-bar { padding: 8px 40px; background: var(--surface-alt,#f2f1ee); border-top: 1px solid var(--border); display: flex; align-items: center; gap: 16px; font-family: var(--mono); font-size: 10.5px; color: var(--text-muted); }
#status-bar .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); display: inline-block; animation: pulse-dot 2.5s ease-in-out infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)} }
::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
footer{display:none!important}.contain{padding:0!important}.form{background:transparent!important;border:none!important}
hr{border:none;border-top:1px solid var(--border-soft);margin:12px 0}
#chatbot .message{animation:slide-in .25s ease-out}
@keyframes slide-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
"""


def _format_structured(structured: dict, intent: str) -> str:
    if not structured:
        return ""
    if intent == "purchase":
        lines = []
        if structured.get("primary_pick"):
            lines.append(f"首推：{structured['primary_pick']}")
        if structured.get("alternatives"):
            lines.append(f"备选：{', '.join(structured['alternatives'])}")
        if structured.get("price_range"):
            lines.append(f"价格：{structured['price_range']}")
        if structured.get("next_action"):
            lines.append(f"建议：{structured['next_action']}")
        return "\n".join(lines)
    if intent == "aftersales":
        lines = []
        sev_map = {"critical": "🔴 严重", "urgent": "🟠 紧急", "normal": "🟡 一般", "info": "⚪ 提示"}
        lines.append(sev_map.get(structured.get("severity", "normal"), ""))
        if structured.get("safety_warning"):
            lines.append(f"⚠️ {structured['safety_warning']}")
        if structured.get("immediate_actions"):
            lines.append("立即：" + "、".join(structured["immediate_actions"]))
        return "\n".join(l for l in lines if l)
    return ""


async def _respond(message: str, history: list, session_id: str, backend: str):
    if not message.strip():
        yield history, "", "", "", "", "", "", "", gr.update(visible=False)
        return

    sid = session_id or "default"
    prev_messages = _session_messages.get(sid, [])

    # Show user message + loading placeholder immediately
    history = list(history) + [{"role": "user", "content": message}]
    history.append({"role": "assistant", "content": "⏳ 思考中..."})
    yield history, "", "", "", "", "", "", "", gr.update(visible=False)

    os.environ["LLM_BACKEND"] = backend.lower() if backend else "ollama"

    graph = _graph
    config = {"configurable": {"thread_id": sid}}

    init_state: CSState = {
        "messages": prev_messages,
        "query": message,
        "intent": "",
        "intent_confidence": 0.0,
        "entities": [],
        "retrieved_chunks": [],
        "retrieval_trace": [],
        "ticket_id": "",
        "answer": "",
        "structured": {},
        "sources": [],
        "session_id": sid,
        "step_count": 0,
        "memory_context": "",
        "backend": backend.lower() if backend else "ollama",
        "cache_hit": False,
        "elapsed_ms": 0,
    }

    displayed = ""
    final_result: dict = {}

    # Real token streaming via astream_events
    async for event in graph.astream_events(init_state, config=config, version="v2"):  # type: ignore[arg-type]
        etype = event.get("event", "")

        if etype == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk is None:
                continue
            content = chunk.content
            token = content if isinstance(content, str) else ""
            if not token:
                continue
            displayed += token
            history[-1] = {"role": "assistant", "content": displayed}
            yield history, "", "", "", "", "", "", "", gr.update(visible=False)

        elif etype == "on_chain_end" and event.get("name") == "LangGraph":
            output = event["data"].get("output", {})
            if isinstance(output, dict):
                final_result = output

    # After stream completes: get state snapshot to check for safety interrupt
    state_snapshot = graph.get_state(config)  # type: ignore[arg-type]
    is_interrupted = bool(state_snapshot.next)

    _session_messages[sid] = final_result.get("messages", prev_messages)

    # Ensure chatbot shows the final answer (may differ from streamed tokens if cache hit)
    raw_answer = final_result.get("answer", displayed) or displayed
    if not raw_answer:
        raw_answer = "抱歉，暂时无法回答，请稍后再试。"
    history[-1] = {"role": "assistant", "content": raw_answer}

    if is_interrupted:
        yield history, "◉ 售后服务", "", "", "", "", "🔴 严重 — 等待确认", "", gr.update(visible=True)
        return

    intent_key = final_result.get("intent", "")
    icon = _INTENT_ICONS.get(intent_key, "·")
    label = _INTENT_LABELS.get(intent_key, intent_key)
    intent_display = f"{icon} {label}" if label and label != "—" else ""

    cache_md = "⚡ 语义缓存命中" if final_result.get("cache_hit") else ""

    entities_md = ""
    if final_result.get("entities"):
        rows = "\n".join(
            f"| `{e['text']}` | {e['label']} |" for e in final_result["entities"][:8]
        )
        entities_md = f"| 实体 | 类型 |\n|---|---|\n{rows}"

    sources_md = ""
    if final_result.get("sources"):
        sources_md = "\n".join(f"- `{s}`" for s in final_result["sources"][:5])

    ticket_md = f"**工单** `{final_result['ticket_id']}`" if final_result.get("ticket_id") else ""

    struct_md = _format_structured(final_result.get("structured", {}), intent_key)

    trace_md = ""
    if final_result.get("retrieval_trace"):
        trace_md = "\n".join(final_result["retrieval_trace"][:6])
    elapsed = final_result.get("elapsed_ms", 0)
    bk = final_result.get("backend", backend)
    if elapsed:
        trace_md = (trace_md + f"\n⏱ {elapsed}ms · {bk}").strip()

    yield history, intent_display, cache_md, entities_md, sources_md, ticket_md, struct_md, trace_md, gr.update(visible=False)


def _confirm_safety(approved: bool, session_id: str, history: list):
    """Human-in-the-Loop 确认回调。"""
    sid = session_id or "default"
    result = resume_after_safety(sid, approved=approved)
    msg = result.get("answer", "")
    if result.get("ticket_id"):
        msg += f"\n\n📋 工单已提交：`{result['ticket_id']}`"
    elif not approved:
        msg = "已取消工单提交。如需帮助请联系人工客服。"
    history = list(history)
    if history and isinstance(history[-1], dict) and history[-1].get("role") == "assistant":
        history[-1] = {"role": "assistant", "content": msg}
    return history, gr.update(visible=False)


def _clear(session_id: str):
    sid = session_id or "default"
    _session_messages.pop(sid, None)
    clear_memory(sid)
    cache_clear()
    return [], "", "", "", "", "", "", "", gr.update(visible=False)


with gr.Blocks(title="新能源汽车智能客服", css=CSS) as demo:

    gr.HTML("""
    <div id="app-header">
        <h1>新能源汽车智能客服</h1>
        <p>Multi-Agent · LangGraph · Hybrid RAG · Self-RAG · Checkpointing · 长期记忆 · 181 车型</p>
    </div>
    """)

    with gr.Tabs():
      with gr.TabItem("💬 对话"):
        with gr.Row(elem_id="main-row"):
            with gr.Column(elem_id="chat-col", scale=7):
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label="",
                    show_label=False, height=480,
                    type="messages",
                )

                # Human-in-the-Loop 安全确认行（默认隐藏）
                with gr.Row(visible=False) as safety_row:
                    gr.HTML("<b>⚠️ 检测到安全风险，是否提交服务工单？</b>")
                    confirm_btn = gr.Button("✅ 确认提交工单", variant="primary", scale=1)
                    cancel_btn = gr.Button("❌ 取消", variant="secondary", scale=1)

                with gr.Row(elem_id="input-area"):
                    msg_box = gr.Textbox(
                        placeholder="输入您的问题，如：问界M9和理想L9怎么选？",
                        show_label=False, elem_id="msg-input", scale=9, lines=1, max_lines=4,
                    )
                    send_btn = gr.Button("发送", elem_id="send-btn", variant="primary", scale=1)

                with gr.Row(elem_id="controls-row"):
                    clear_btn = gr.Button("清空对话", elem_id="clear-btn", size="sm", scale=1)
                    session_box = gr.Textbox(value="user_001", label="会话 ID", elem_id="session-box", scale=2)
                    backend_radio = gr.Radio(
                        choices=["ollama", "deepseek"],
                        value=get_active_backend(),
                        label="LLM 后端",
                        scale=2,
                    )

                with gr.Column(elem_id="examples-container"):
                    gr.Examples(
                        examples=[
                            ["问界M9和理想L9哪个更适合家用？"],
                            ["小米SU7 Ultra和特斯拉Model 3性能对比"],
                            ["仰望U8和蔚来ET9，100万预算怎么选？"],
                            ["极氪007续航多少，快充多久充满？"],
                            ["我的空调不制冷了，怎么处理？"],
                            ["XNGP城市版哪些车型支持？"],
                            ["首任车主有哪些终身权益？"],
                            ["30万预算，家用5座SUV，推荐哪款？"],
                            ["我的刹车失灵了！"],
                        ],
                        inputs=msg_box,
                        label="示例问题",
                    )

            with gr.Column(elem_id="info-panel", scale=3, min_width=300):
                gr.HTML('<div id="info-panel-title">ANALYSIS</div>')

                gr.HTML('<div class="info-section-label">INTENT</div>')
                intent_display = gr.Markdown(elem_id="intent-display")

                cache_badge = gr.Markdown(elem_id="cache-badge")

                gr.HTML('<hr/><div class="info-section-label">ENTITIES</div>')
                entities_out = gr.Markdown(elem_id="entities-out")

                gr.HTML('<hr/><div class="info-section-label">SOURCES</div>')
                sources_out = gr.Markdown(elem_id="sources-out")

                gr.HTML('<hr/><div class="info-section-label">TICKET / STRUCTURED</div>')
                ticket_out = gr.Markdown(elem_id="ticket-out")
                struct_out = gr.Markdown(elem_id="struct-out")

                gr.HTML('<hr/><div class="info-section-label">RETRIEVAL TRACE</div>')
                trace_out = gr.Markdown(elem_id="trace-out")

        _backend_val = get_active_backend()
        gr.HTML(f"""
        <div id="status-bar">
            <span class="dot"></span>
            <span>系统就绪</span><span>·</span>
            <span>181+ 车型</span><span>·</span>
            <span>12类意图路由</span><span>·</span>
            <span>Hybrid RAG + Self-RAG</span><span>·</span>
            <span>SqliteSaver 持久化</span><span>·</span>
            <span>Human-in-the-Loop</span><span>·</span>
            <span>后端: {_backend_val}</span>
        </div>
        """)

      with gr.TabItem("🗺️ 图结构"):
        gr.HTML("<p style='padding:12px;font-size:12px;color:#888'>LangGraph Agent 路由拓扑（Mermaid）</p>")
        mermaid_out = gr.Code(language="markdown", label="Mermaid")
        viz_btn = gr.Button("刷新图结构", variant="secondary")
        viz_btn.click(fn=get_graph_mermaid, outputs=mermaid_out)

    _outputs = [chatbot, intent_display, cache_badge, entities_out, sources_out, ticket_out, struct_out, trace_out, safety_row]

    send_btn.click(
        _respond,
        inputs=[msg_box, chatbot, session_box, backend_radio],
        outputs=_outputs,
    ).then(lambda: "", outputs=msg_box)

    msg_box.submit(
        _respond,
        inputs=[msg_box, chatbot, session_box, backend_radio],
        outputs=_outputs,
    ).then(lambda: "", outputs=msg_box)

    clear_btn.click(
        _clear,
        inputs=[session_box],
        outputs=_outputs,
    )

    confirm_btn.click(
        lambda sid, hist: _confirm_safety(True, sid, hist),
        inputs=[session_box, chatbot],
        outputs=[chatbot, safety_row],
    )
    cancel_btn.click(
        lambda sid, hist: _confirm_safety(False, sid, hist),
        inputs=[session_box, chatbot],
        outputs=[chatbot, safety_row],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)


def launch_ui(server_name: str = "127.0.0.1", server_port: int = 7860, **kwargs):
    """External entry point."""
    demo.launch(server_name=server_name, server_port=server_port, show_error=True, **kwargs)
