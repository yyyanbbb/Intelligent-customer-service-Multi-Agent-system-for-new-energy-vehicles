"""
LLM 调用层，封装 Ollama 和 DeepSeek 两个后端。
get_chat_model() 返回 LangChain BaseChatModel，支持 bind_tools / with_structured_output。
llm_chat / llm_generate 是兼容旧节点代码的简化接口，内部都走 LangChain。
一个后端挂掉会自动切到另一个。
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import Iterator, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.language_models import BaseChatModel

# Load .env
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")

DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


# ============================================================
#  ChatModel factory (cached)
# ============================================================
@lru_cache(maxsize=8)
def _get_ollama_model(temperature: float = 0.3, max_tokens: int = 1024) -> BaseChatModel:
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE,
        temperature=temperature,
        num_predict=max_tokens,
        reasoning=False,  # disable think for qwen3
    )


@lru_cache(maxsize=8)
def _get_deepseek_model(temperature: float = 0.3, max_tokens: int = 1024) -> BaseChatModel:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置")
    from langchain_deepseek import ChatDeepSeek
    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_chat_model(
    backend: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> BaseChatModel:
    """返回对应后端的 BaseChatModel，走 lru_cache 不会重复初始化。"""
    backend = (backend or DEFAULT_BACKEND).lower()
    if backend == "deepseek":
        return _get_deepseek_model(temperature, max_tokens)
    return _get_ollama_model(temperature, max_tokens)


# ============================================================
#  Message format helpers
# ============================================================
def _to_lc_messages(messages: list[dict]) -> list[BaseMessage]:
    """dict messages → LangChain BaseMessage list."""
    result: list[BaseMessage] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


def _tool_calls_to_dict(tool_calls: list) -> list[dict]:
    """LangChain tool_calls → OpenAI-style dict (for backward compat)."""
    out = []
    for tc in tool_calls:
        out.append({
            "id": tc.get("id", ""),
            "function": {
                "name": tc.get("name", ""),
                "arguments": tc.get("args", {}),
            },
            "args": tc.get("args", {}),
        })
    return out


# ============================================================
#  Public API (backward compatible)
# ============================================================
def llm_chat(
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.3,
    tools: list[dict] | None = None,
    stream: bool = False,
    backend: str | None = None,
) -> Any:
    """
    通用对话接口。
    stream=True 返回 Iterator[str]；触发 tool_call 返回 {"content": str, "tool_calls": [...]}；
    其余情况返回字符串。双后端都挂时返回错误提示字符串。
    """
    backend = (backend or DEFAULT_BACKEND).lower()
    try:
        model = get_chat_model(backend, temperature=temperature, max_tokens=max_tokens)
        if tools:
            model = model.bind_tools(tools)
        lc_msgs = _to_lc_messages(messages)

        if stream:
            return _stream_iter(model, lc_msgs)

        resp = model.invoke(lc_msgs)
        if getattr(resp, "tool_calls", None):
            return {
                "content": resp.content if isinstance(resp.content, str) else "",
                "tool_calls": _tool_calls_to_dict(resp.tool_calls),
            }
        return resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:
        logger.warning(f"{backend} 调用失败: {e}，尝试回退")
        fallback = "deepseek" if backend == "ollama" else "ollama"
        try:
            model = get_chat_model(fallback, temperature=temperature, max_tokens=max_tokens)
            if tools:
                model = model.bind_tools(tools)
            lc_msgs = _to_lc_messages(messages)
            if stream:
                return _stream_iter(model, lc_msgs)
            resp = model.invoke(lc_msgs)
            if getattr(resp, "tool_calls", None):
                return {
                    "content": resp.content if isinstance(resp.content, str) else "",
                    "tool_calls": _tool_calls_to_dict(resp.tool_calls),
                }
            return resp.content if isinstance(resp.content, str) else str(resp.content)
        except Exception as e2:
            logger.error(f"双后端均失败: {e2}")
            return f"[LLM 双后端均不可用: {e2}]"


def _stream_iter(model: BaseChatModel, lc_msgs: list[BaseMessage]) -> Iterator[str]:
    for chunk in model.stream(lc_msgs):
        if hasattr(chunk, "content") and chunk.content:
            if isinstance(chunk.content, str):
                yield chunk.content
            else:
                yield str(chunk.content)


def llm_generate(
    prompt: str,
    system: str = "",
    max_tokens: int = 1024,
    stream: bool = False,
    backend: str | None = None,
) -> Any:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return llm_chat(messages, max_tokens=max_tokens, stream=stream, backend=backend)


def get_active_backend() -> str:
    return DEFAULT_BACKEND
