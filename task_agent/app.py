from __future__ import annotations

import json

import gradio as gr

from task_agent.service import TaskService


service = TaskService()

CSS = """
:root {
  --bg: #f7f5ef;
  --panel: #ffffff;
  --muted: #6b6861;
  --ink: #171614;
  --line: #ddd8cc;
  --accent: #0d8f66;
  --warn: #b96932;
}
body { background: linear-gradient(160deg, #f4efe2 0%, #f9f8f3 55%, #eef5f2 100%) !important; }
.gradio-container { max-width: 1280px !important; }
#header { padding: 28px 8px 8px; }
#header h1 { margin: 0; color: var(--ink); font-size: 28px; }
#header p { margin: 6px 0 0; color: var(--muted); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }
#summary-card, #result-card, #tool-card { border: 1px solid var(--line); border-radius: 14px; background: var(--panel); }
"""


def _render_state(state: dict) -> tuple[str, str, str, str, str, str]:
    result = state.get("result", {})
    plan_lines = []
    for step in state.get("plan", []):
        marker = "x" if step.get("status") == "completed" else " "
        plan_lines.append(f"- [{marker}] {step.get('title')} ({step.get('owner_agent')})")
    plan_md = "\n".join(plan_lines) or "- 暂无计划"

    questions = "\n".join(f"- {item}" for item in state.get("pending_questions", [])) or "- 无"
    confirmations = "\n".join(
        f"- `{item['confirmation_id']}` {item['prompt']}" for item in state.get("pending_confirmations", [])
    ) or "- 无"
    completed = "\n".join(
        f"- `{item['tool_name']}` {item['summary']}" for item in state.get("completed_actions", [])
    ) or "- 无"

    summary = (
        f"**Task ID** `{state.get('task_id', '')}`\n\n"
        f"**Goal** {state.get('goal', '')}\n\n"
        f"**Status** `{state.get('task_status', '')}`\n\n"
        f"**Active Agent** `{state.get('active_agent', '')}`"
    )
    result_md = f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```" if result else "暂无结果"
    tool_md = f"```json\n{json.dumps(state.get('tool_outputs', {}), ensure_ascii=False, indent=2)}\n```"
    return summary, plan_md, questions, confirmations, completed, tool_md + "\n\n" + result_md


def _submit(message: str, task_id: str, session_id: str):
    if not message.strip():
        return task_id, *(_render_state({})), ""
    if task_id:
        state = service.continue_task(task_id, message)
    else:
        state = service.start_task(message, session_id=session_id or "task-ui")
        task_id = state["task_id"]
    summary, plan_md, questions, confirmations, completed, result_md = _render_state(state)
    return task_id, summary, plan_md, questions, confirmations, completed, result_md, ""


def _confirm(task_id: str, approve: bool):
    if not task_id:
        return task_id, *(_render_state({})), ""
    state = service.get_task_status(task_id)
    pending = state.get("pending_confirmations", [])
    if not pending:
        return task_id, *(_render_state(state)), ""
    updated = service.confirm_task_action(task_id, pending[0]["confirmation_id"], approve)
    summary, plan_md, questions, confirmations, completed, result_md = _render_state(updated)
    return task_id, summary, plan_md, questions, confirmations, completed, result_md, ""


with gr.Blocks(title="EV Task Agent", css=CSS) as demo:
    gr.HTML(
        """
        <div id="header">
          <h1>EV Task Agent</h1>
          <p>Supervisor + PlannerAgent + PurchaseAgent + AftersalesAgent</p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=5):
            session_id = gr.Textbox(label="Session ID", value="task-ui")
            task_id = gr.Textbox(label="Task ID", interactive=False)
            user_input = gr.Textbox(label="输入任务或补充信息", lines=3, placeholder="例如：预算25万，家里有充电桩，每天通勤60公里，帮我选车并预约试驾")
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary")
                approve_btn = gr.Button("确认待执行动作")
                reject_btn = gr.Button("拒绝待执行动作")
            result_panel = gr.Markdown(elem_id="result-card")
        with gr.Column(scale=4):
            summary_panel = gr.Markdown(elem_id="summary-card")
            plan_panel = gr.Markdown(label="当前计划")
            question_panel = gr.Markdown(label="待补充信息")
            confirm_panel = gr.Markdown(label="待确认动作")
            action_panel = gr.Markdown(label="已完成动作")
            tool_panel = gr.Markdown(elem_id="tool-card")

    submit_btn.click(
        _submit,
        inputs=[user_input, task_id, session_id],
        outputs=[task_id, summary_panel, plan_panel, question_panel, confirm_panel, action_panel, tool_panel, user_input],
    )
    user_input.submit(
        _submit,
        inputs=[user_input, task_id, session_id],
        outputs=[task_id, summary_panel, plan_panel, question_panel, confirm_panel, action_panel, tool_panel, user_input],
    )
    approve_btn.click(
        lambda task: _confirm(task, True),
        inputs=[task_id],
        outputs=[task_id, summary_panel, plan_panel, question_panel, confirm_panel, action_panel, tool_panel, user_input],
    )
    reject_btn.click(
        lambda task: _confirm(task, False),
        inputs=[task_id],
        outputs=[task_id, summary_panel, plan_panel, question_panel, confirm_panel, action_panel, tool_panel, user_input],
    )


def launch_ui(server_name: str = "127.0.0.1", server_port: int = 7861, **kwargs):
    demo.queue().launch(server_name=server_name, server_port=server_port, **kwargs)
