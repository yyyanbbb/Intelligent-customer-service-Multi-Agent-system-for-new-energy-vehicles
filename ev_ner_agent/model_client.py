"""Ollama / vLLM OpenAI-compatible API 的统一封装，对上层屏蔽底层差异。"""
from __future__ import annotations

import json
import time
import logging
from typing import Any
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    content: str | None
    tool_calls: list[dict] | None
    raw: dict

    @property
    def has_tool_calls(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0

    @property
    def first_tool_call(self) -> dict | None:
        return self.tool_calls[0] if self.has_tool_calls else None


class ModelClient:
    """
    统一模型客户端，兼容 Ollama / vLLM / LM Studio 等 OpenAI-compatible API。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        api_key: str | None = None,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "ollama"
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
    ) -> ModelResponse:
        """
        发送 chat 请求到模型，返回 ModelResponse。

        如果 tools 不为空，模型返回 tool_calls；否则返回纯文本。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if stop:
            payload["stop"] = stop

        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"模型请求失败: {e}")
            return ModelResponse(
                content=None,
                tool_calls=None,
                raw={"error": str(e)},
            )

        elapsed = time.time() - start
        logger.info(f"模型推理耗时: {elapsed:.2f}s | model={self.model}")

        raw = resp.json()
        choice = raw["choices"][0]
        msg = choice["message"]

        if "tool_calls" in msg and msg["tool_calls"]:
            tool_calls = []
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": fn["name"],
                    "arguments": fn["arguments"],
                })
            return ModelResponse(
                content=None,
                tool_calls=tool_calls,
                raw=raw,
            )
        else:
            return ModelResponse(
                content=msg.get("content"),
                tool_calls=None,
                raw=raw,
            )

    def chat_with_text(self, prompt: str, system: str | None = None) -> str:
        """简化的纯文本补全接口，用于快速测试。"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.chat(messages)
        return resp.content or ""

    def get_model_info(self) -> dict[str, Any]:
        """查询模型信息（仅 Ollama 支持）。"""
        try:
            resp = requests.get(
                f"{self.base_url.rreplace('/v1', '')}/api/show",
                params={"name": self.model},
                timeout=10,
            )
            return resp.json()
        except Exception:
            return {}


def create_client(
    provider: str = "ollama",
    model: str = "qwen2.5:7b",
    **kwargs,
) -> ModelClient:
    """
    工厂函数，根据 provider 类型创建对应的 ModelClient。
    """
    if provider == "ollama":
        base_url = kwargs.get("base_url", "http://localhost:11434/v1")
        return ModelClient(base_url=base_url, model=model, api_key="ollama", **kwargs)
    elif provider == "vllm":
        base_url = kwargs.get("base_url", "http://localhost:8000/v1")
        return ModelClient(base_url=base_url, model=model, api_key="EMPTY", **kwargs)
    elif provider == "lmstudio":
        base_url = kwargs.get("base_url", "http://localhost:1234/v1")
        return ModelClient(base_url=base_url, model=model, api_key="lm-studio", **kwargs)
    else:
        raise ValueError(f"不支持的 provider: {provider}")
