"""
模型输出的 JSON 解析和修复。
先直接 json.loads，失败后依次尝试：提取 markdown 代码块、提取 {...} 块、正则修复单引号和尾部逗号。
还是不行就返回错误信息让调用方决定是否重试。
"""
from __future__ import annotations

import re
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    """JSON 解析失败的异常。"""
    def __init__(self, raw_text: str, reason: str):
        self.raw_text = raw_text
        self.reason = reason
        super().__init__(f"JSON 解析失败: {reason}")


def parse_json_response(text: str) -> dict[str, Any] | list | None:
    """
    解析模型返回的文本，提取其中的 JSON 数据。

    尝试以下策略（按优先级）：
    1. 直接解析
    2. 提取 ```json ... ``` 代码块
    3. 提取 ``` ... ``` 普通代码块
    4. 提取 {...} 或 [...] 块
    5. 修复后重试
    """
    text = text.strip()

    # 策略 1：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略 2：提取 ```json ... ``` 块
    code_block_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL
    )
    if code_block_match:
        json_text = code_block_match.group(1).strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # 策略 3：提取 ``` ... ``` 块（不指定语言）
    plain_block_match = re.search(r"```\s*\n?(.*?)\n?```", text, re.DOTALL)
    if plain_block_match:
        json_text = plain_block_match.group(1).strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # 策略 4：提取 {...} 或 [...] 块
    for pattern in [
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]",
    ]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_text = match.group(0)
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                continue

    return None


def fix_common_json_issues(text: str) -> str:
    """
    修复常见的 JSON 格式问题：

    1. 单引号 → 双引号（但需要处理引号内的内容）
    2. 尾部逗号 → 移除
    3. 注释（// 或 /* */）→ 移除
    4. unquoted keys → 加上引号
    5. trailing noise（markdown 文字等）→ 截断到最后一个 } 或 ]
    """
    original = text

    # 移除行注释
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    # 移除块注释
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # 移除尾部逗号（, 后面紧跟 ] 或 }）
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    # 处理单引号：将包裹字符串的单引号替换为双引号
    # 这是简化处理，复杂情况交给 json.loads 失败后的重试
    fixed = []
    i = 0
    in_string = False
    string_char = None
    while i < len(text):
        ch = text[i]
        if not in_string:
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                fixed.append('"')
            elif ch in ("'",):
                # 可能是 key 或 value 的引号
                # 简单策略：如果前后是冒号或逗号或括号，当作 key 处理
                fixed.append('"')
                string_char = ch
                in_string = True
            else:
                fixed.append(ch)
        else:
            if ch == string_char and (i == 0 or text[i - 1] != "\\"):
                in_string = False
                fixed.append('"')
                string_char = None
            elif ch == "'" and string_char == '"':
                fixed.append(ch)
            else:
                fixed.append(ch)
        i += 1

    text = "".join(fixed)

    # 截断到最后一个有效 JSON 结构
    last_brace = text.rfind("}")
    last_bracket = text.rfind("]")
    cut_point = max(last_brace, last_bracket)
    if cut_point > 0:
        text = text[:cut_point + 1]
        if text.startswith("{"):
            text = "{" + text[1:]
            text = "{" + text
        if not (text.startswith("{") or text.startswith("[")):
            for j, ch in enumerate(text):
                if ch in ("{", "["):
                    text = text[j:]
                    break

    if text != original:
        logger.debug(f"JSON 修复: 原始长度={len(original)}, 修复后={len(text)}")

    return text


def extract_json_with_fallback(text: str) -> tuple[dict[str, Any] | list | None, str]:
    """
    带修复的 JSON 提取。

    返回: (解析结果, 状态描述)
    """
    # 第一次尝试
    result = parse_json_response(text)
    if result is not None:
        return result, "直接解析成功"

    # 第二次尝试：修复后解析
    fixed = fix_common_json_issues(text)
    result = parse_json_response(fixed)
    if result is not None:
        return result, "修复格式后解析成功"

    return None, f"解析失败，原始文本前100字符: {text[:100]!r}"


def parse_extraction_result(text: str) -> dict[str, Any] | None:
    """
    专门解析抽取结果的 JSON。
    期望格式包含 entities 和 relations 字段。
    """
    result, status = extract_json_with_fallback(text)
    logger.debug(f"JSON 解析状态: {status}")

    if result is None:
        return None

    # 如果是列表（多个结果），取第一个字典元素
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                result = item
                break
        else:
            return None

    if not isinstance(result, dict):
        return None

    return result
