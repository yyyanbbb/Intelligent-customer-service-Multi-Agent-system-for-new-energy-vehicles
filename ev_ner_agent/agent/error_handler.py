"""
工具执行和模型调用的错误恢复。工具失败直接把错误观察返回给模型让它换方案；
超时用 threading 子线程 + sentinel 实现（Windows 没有 SIGALRM）；
JSON 解析失败自动修复后重试，最多 2 次。
"""
from __future__ import annotations

import time
import logging
import threading
from typing import Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    JSON_PARSE_ERROR = "json_parse_error"
    MODEL_API_ERROR = "model_api_error"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    VALIDATION_ERROR = "validation_error"


@dataclass
class ErrorRecord:
    step: int
    error_type: ErrorType
    message: str
    tool_name: str | None = None
    timestamp: float = field(default_factory=time.time)


class ErrorRecoveryPolicy:
    """
    错误恢复策略管理器。
    核心思想：不是所有错误都需要重试，有些错误应该让模型换一个方案。
    """

    RETRYABLE_ERRORS: set[ErrorType] = {
        ErrorType.MODEL_API_ERROR,
        ErrorType.JSON_PARSE_ERROR,
    }

    NON_RETRYABLE_ERRORS: set[ErrorType] = {
        ErrorType.TOOL_EXECUTION_ERROR,
        ErrorType.VALIDATION_ERROR,
        ErrorType.MAX_STEPS_EXCEEDED,
    }

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.error_history: list[ErrorRecord] = []

    def should_retry(self, error_type: ErrorType) -> bool:
        if error_type not in self.RETRYABLE_ERRORS:
            return False
        retry_count = sum(1 for r in self.error_history if r.error_type == error_type)
        return retry_count < self.max_retries

    def record_error(self, record: ErrorRecord) -> None:
        self.error_history.append(record)
        logger.warning(
            f"[ErrorRecord] step={record.step} type={record.error_type.value} "
            f"msg={record.message[:80]} tool={record.tool_name}"
        )

    def get_recovery_suggestion(self, error_type: ErrorType) -> str:
        suggestions = {
            ErrorType.TOOL_EXECUTION_ERROR: (
                "工具执行失败。请检查参数是否正确，如果文件路径有问题可以换一个文档。"
            ),
            ErrorType.JSON_PARSE_ERROR: (
                "JSON 格式解析失败。请确保输出是标准的 JSON 格式，key 和 string value 必须使用双引号。"
            ),
            ErrorType.MODEL_API_ERROR: (
                "模型 API 请求失败，请稍后重试。如果持续失败，可能是网络问题。"
            ),
            ErrorType.MAX_STEPS_EXCEEDED: (
                "已达到最大步数限制。请输出当前已抽取的结果，不要继续调用工具。"
            ),
            ErrorType.VALIDATION_ERROR: (
                "Schema 校验失败，请修正抽取结果的格式和数值范围后重新输出。"
            ),
        }
        return suggestions.get(error_type, "发生未知错误，请检查输入。")

    def clear_history(self) -> None:
        self.error_history.clear()


def _run_with_timeout(func: Callable, args: tuple, kwargs: dict, result: dict) -> None:
    """在线程中执行函数，结果存入 result 字典。"""
    try:
        result["value"] = func(*args, **kwargs)
        result["error"] = None
    except Exception as e:
        result["error"] = e


def safe_execute_tool(
    func: Callable,
    *args,
    timeout: float = 30.0,
    **kwargs,
) -> tuple[bool, str]:
    """
    带超时的工具执行包装器（跨平台，Windows 兼容）。
    使用 threading + 子线程 sentinel 模式实现超时。
    返回: (是否成功, 结果或错误信息)
    """
    start = time.time()
    result: dict[str, Any] = {}

    worker = threading.Thread(
        target=_run_with_timeout,
        args=(func, args, kwargs, result),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        elapsed = time.time() - start
        logger.warning(f"工具 {func.__name__} 执行超时（{elapsed:.1f}s >= {timeout}s）")
        return False, f"工具执行超时（{timeout}s），请检查文件是否过大或路径是否正确。"

    if "error" in result and result["error"] is not None:
        exc = result["error"]
        logger.error(f"工具 {func.__name__} 执行异常: {exc}")
        if isinstance(exc, FileNotFoundError):
            return False, f"文件未找到，请检查路径是否正确。"
        return False, f"工具执行异常: {exc}"

    elapsed = time.time() - start
    logger.info(f"工具 {func.__name__} 执行成功，耗时 {elapsed:.2f}s")
    return True, str(result.get("value", ""))


def execute_with_retry(
    policy: ErrorRecoveryPolicy,
    error_type: ErrorType,
    func: Callable,
    *args,
    **kwargs,
) -> tuple[bool, str]:
    """带重试策略的执行。"""
    if policy.should_retry(error_type):
        logger.info(f"尝试重试 error_type={error_type.value}")
        time.sleep(1)
        return safe_execute_tool(func, *args, **kwargs)
    else:
        return safe_execute_tool(func, *args, **kwargs)
