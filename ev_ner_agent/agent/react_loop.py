"""
手写 ReAct 调度循环，不依赖 LangChain 框架。
Thought → Action → Observation 循环，有三层死循环防护：
max_steps 硬截断、连续同一工具检测、连续相同 tool_calls 检测。
工具失败时把错误观察返回给模型，让它自己决定换哪个工具。
"""
from __future__ import annotations

import json
import re
import time
import logging
from typing import Any
from dataclasses import dataclass, field

from ev_ner_agent.model_client import ModelClient, ModelResponse
from ev_ner_agent.agent.prompt_templates import (
    SYSTEM_PROMPT,
    build_user_prompt,
    REFINE_PROMPT,
    FINAL_SUMMARY_PROMPT,
)
import re

from ev_ner_agent.agent.json_parser import parse_extraction_result
from ev_ner_agent.agent.error_handler import (
    ErrorType,
    ErrorRecord,
    ErrorRecoveryPolicy,
)
from ev_ner_agent.tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    step: int
    thought: str
    action: str | None
    tool_args: dict | None
    observation: str | None
    error: str | None
    elapsed: float


@dataclass
class ExtractionResult:
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    summary: str = ""
    metadata: dict = field(default_factory=dict)

    def merge(self, other: "ExtractionResult") -> None:
        """合并两轮抽取结果，去重。"""
        seen_entities = {(e.get("name", ""), e.get("entity_type", "")) for e in self.entities}
        for entity in other.entities:
            key = (entity.get("name", ""), entity.get("entity_type", ""))
            if key not in seen_entities:
                self.entities.append(entity)
                seen_entities.add(key)

        seen_relations = {
            (r.get("source_name", ""), r.get("relation_type", ""), r.get("target_name", ""))
            for r in self.relations
        }
        for relation in other.relations:
            key = (
                relation.get("source_name", ""),
                relation.get("relation_type", ""),
                relation.get("target_name", ""),
            )
            if key not in seen_relations:
                self.relations.append(relation)
                seen_relations.add(key)

        if other.summary:
            self.summary += ("\n" if self.summary else "") + other.summary


class ReActAgent:
    """
    手写 ReAct 调度器。

    设计要点：
    - 不依赖 LangChain / AutoGPT 等框架
    - 完整的状态机管理（idle → thinking → acting → observing → done）
    - 所有中间状态可审计（step_history）
    - 支持工具执行失败后的模型自我修正
    """

    def __init__(
        self,
        model_client: ModelClient,
        max_steps: int = 15,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ):
        self.model = model_client
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.step_history: list[StepRecord] = []
        self.error_policy = ErrorRecoveryPolicy(max_retries=2)
        self.state = "idle"

    def run(self, user_query: str, doc_context: str = "", graph_context: str = "") -> dict:
        """
        运行 ReAct Agent 主循环。

        参数：
        - user_query: 用户的问题或抽取指令
        - doc_context: 文档上下文（如已知的文件路径列表）
        - graph_context: 知识图谱上下文

        返回：
        - 包含 entities、relations、summary、step_history 的字典
        """
        self.state = "running"
        self.step_history.clear()
        self.error_policy.clear_history()

        user_prompt = build_user_prompt(user_query, doc_context, graph_context)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        accumulated_result = ExtractionResult()
        step_count = 0

        while step_count < self.max_steps:
            step_count += 1
            step_start = time.time()

            # ---------- Step 1: 调用模型 ----------
            self.state = "thinking"
            response: ModelResponse = self.model.chat(
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # API 请求失败，降级重试
            if response.content is None and not response.has_tool_calls:
                error_record = ErrorRecord(
                    step=step_count,
                    error_type=ErrorType.MODEL_API_ERROR,
                    message=str(response.raw.get("error", "未知错误")),
                )
                self.error_policy.record_error(error_record)

                if self.error_policy.should_retry(ErrorType.MODEL_API_ERROR):
                    # 降级：减少 max_tokens 后重试
                    self.max_tokens = max(512, self.max_tokens // 2)
                    recovery_msg = (
                        f"模型请求失败: {error_record.message}。"
                        f"请尝试减少输出内容，输出更简洁的结果。"
                    )
                    messages.append({"role": "user", "content": recovery_msg})
                    continue
                else:
                    return self._make_final_response(
                        accumulated_result,
                        error=f"模型持续请求失败: {error_record.message}"
                    )

            assistant_msg = response.raw["choices"][0]["message"]
            messages.append(assistant_msg)

            # 解析模型输出
            self.state = "acting"

            # 如果模型返回了文本（可能是 JSON 格式的最终结果）
            if response.content and not response.has_tool_calls:
                text = response.content.strip()

                # 尝试解析为结构化 JSON
                parsed = parse_extraction_result(text)

                if parsed and isinstance(parsed, dict):
                    # 成功解析到 JSON，尝试从中提取 entities 和 relations
                    entities = parsed.get("entities", [])
                    relations = parsed.get("relations", [])
                    summary = parsed.get("summary", text[:500])

                    if entities or relations:
                        step_record = StepRecord(
                            step=step_count,
                            thought="模型直接输出结构化 JSON",
                            action=None,
                            tool_args=None,
                            observation=f"抽取到 {len(entities)} 个实体，{len(relations)} 个关系",
                            error=None,
                            elapsed=time.time() - step_start,
                        )
                        self.step_history.append(step_record)

                        accumulated_result.merge(ExtractionResult(
                            entities=entities,
                            relations=relations,
                            summary=summary,
                        ))
                        self.state = "done"
                        break
                    else:
                        # JSON 里没有 entities，但可能是最终文字回答
                        accumulated_result.summary = text[:2000]
                        self.state = "done"
                        break
                else:
                    # 模型输出了文字但不是 JSON，当作最终答案
                    accumulated_result.summary = text[:2000]
                    self.state = "done"
                    break

            # 解析 tool_calls
            tool_call = response.first_tool_call
            if not tool_call:
                self.state = "done"
                break

            # 解析 tool_call 参数
            try:
                tool_args = json.loads(tool_call["arguments"])
            except json.JSONDecodeError:
                tool_args = {}
            tool_name = tool_call["name"]

            # 死循环检测：连续两步调用相同工具+参数则中断
            if step_count >= 2:
                prev_record = self.step_history[-1]
                same_name = tool_name == prev_record.action
                same_args = tool_args == prev_record.tool_args
                same_tool_twice = (
                    step_count >= 3
                    and self.step_history[-1].action == self.step_history[-2].action
                )
                if same_name and same_args:
                    logger.warning(f"检测到死循环: 连续两步调用相同工具+相同参数 {tool_name}")
                    step_record = StepRecord(
                        step=step_count,
                        thought=f"检测到死循环：连续调用 {tool_name}，参数相同",
                        action=tool_name,
                        tool_args=tool_args,
                        observation="死循环截断，强制输出当前结果",
                        error="DEAD_LOOP_DETECTED",
                        elapsed=time.time() - step_start,
                    )
                    self.step_history.append(step_record)
                    self.state = "done"
                    break
                if same_tool_twice and same_name:
                    # 同一工具连续调用两次，检查是否该换策略
                    logger.warning(f"检测到重复工具调用: {tool_name}，通知模型换策略")
                    messages.append({
                        "role": "user",
                        "content": f"[策略建议] 你已连续调用 {tool_name} 两次，请评估当前策略是否有效。如果文档中找不到相关信息，请换一个工具或直接输出已抽取的结果。",
                    })

            # 执行工具
            self.state = "observing"

            success, observation = self._execute_tool_safely(tool_name, tool_args)

            # 提取 thought
            thought = self._extract_thought(assistant_msg)

            step_record = StepRecord(
                step=step_count,
                thought=thought,
                action=tool_name,
                tool_args=tool_args,
                observation=observation[:1000] if observation else None,
                error=None if success else observation[:200],
                elapsed=time.time() - step_start,
            )
            self.step_history.append(step_record)

            # 追加 tool 返回结果到消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": observation,
            })

            # 如果工具执行失败且不可重试，给模型错误反馈让它换策略
            if not success:
                error_type = ErrorType.TOOL_EXECUTION_ERROR
                self.error_policy.record_error(ErrorRecord(
                    step=step_count,
                    error_type=error_type,
                    message=observation,
                    tool_name=tool_name,
                ))
                if not self.error_policy.should_retry(error_type):
                    recovery = self.error_policy.get_recovery_suggestion(error_type)
                    messages.append({
                        "role": "user",
                        "content": f"[错误反馈] {observation}\n\n{recovery}",
                    })

            # 如果达到最大步数
            if step_count >= self.max_steps:
                self.error_policy.record_error(ErrorRecord(
                    step=step_count,
                    error_type=ErrorType.MAX_STEPS_EXCEEDED,
                    message="达到最大步数限制",
                ))
                self.state = "done"
                break

        return self._make_final_response(accumulated_result)

    def _execute_tool_safely(self, tool_name: str, tool_args: dict) -> tuple[bool, str]:
        """安全执行工具，带超时和错误处理。"""
        if tool_name not in [s["function"]["name"] for s in TOOL_SCHEMAS]:
            return False, f"未知工具: {tool_name}"

        try:
            result = execute_tool(tool_name, tool_args)
            return True, str(result)
        except Exception as e:
            logger.error(f"工具 {tool_name} 执行异常: {e}")
            return False, f"工具执行异常: {e}"

    def _extract_thought(self, assistant_msg: dict) -> str:
        """从模型输出中提取 thought 部分（简单策略）。"""
        content = assistant_msg.get("content", "")

        # 尝试从 content 中提取 thought
        patterns = [
            r"(?:thought|思考)[:：]\s*(.+?)(?:\n(?:action|行动|observation)|$)",
            r"(?:让我|我先|首先|接下来)[\[：:](.+?)[】\]]",
        ]
        for pat in patterns:
            match = re.search(pat, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()[:200]

        # 如果找不到 thought，返回前 200 字
        return content[:200] if content else "（无 thought）"

    def _make_final_response(
        self,
        result: ExtractionResult,
        error: str | None = None,
    ) -> dict:
        """构造最终响应。"""
        return {
            "entities": result.entities,
            "relations": result.relations,
            "summary": result.summary,
            "step_history": [
                {
                    "step": r.step,
                    "thought": r.thought,
                    "action": r.action,
                    "tool_args": r.tool_args,
                    "observation": r.observation,
                    "error": r.error,
                    "elapsed": round(r.elapsed, 3),
                }
                for r in self.step_history
            ],
            "metadata": {
                "total_steps": len(self.step_history),
                "total_time": round(sum(r.elapsed for r in self.step_history), 3),
                "error_count": sum(1 for r in self.step_history if r.error),
                "final_state": self.state,
                "error": error,
            },
        }
