"""
LLM utility functions: completion wrapper, JSON parsing, token tracking.
"""

import asyncio
import json
import logging
import re
from typing import Any, List, Optional

import config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-session LLM state (replaces mutable globals)
# ---------------------------------------------------------------------------

_llm_state: dict = {
    "json_mode_supported": True,
    "total_tokens": 0,
    "total_calls": 0,
    "last_model": None,
}


def get_llm_usage_stats() -> dict:
    """Return a snapshot of current LLM session usage statistics."""
    return dict(_llm_state)


async def get_llm_completion(
    client,
    messages: List[dict],
    temperature: float = 0.7,
    json_mode: bool = True,
) -> Any:
    """
    Robust wrapper for LLM completions with:
    - Automatic JSON mode negotiation (disables if model doesn't support it)
    - Model-change detection to reset json_mode flag
    - Token usage tracking
    """
    global _llm_state
    model_name = cfg.get_llm_model_name()

    # Reset json_mode flag when the user switches models
    if _llm_state["last_model"] is not None and _llm_state["last_model"] != model_name:
        logger.info(
            "Model changed (%s → %s): resetting JSON mode flag.",
            _llm_state["last_model"],
            model_name,
        )
        _llm_state["json_mode_supported"] = True
    _llm_state["last_model"] = model_name

    if not _llm_state["json_mode_supported"]:
        json_mode = False

    try:
        kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await asyncio.to_thread(client.chat.completions.create, **kwargs)

        if hasattr(response, "usage") and response.usage:
            _llm_state["total_tokens"] += getattr(response.usage, "total_tokens", 0) or 0
        _llm_state["total_calls"] += 1
        return response

    except Exception as e:
        error_str = str(e).lower()
        if (
            json_mode
            and ("response_format" in error_str or "json_object" in error_str)
            and ("400" in str(e) or "invalid" in error_str)
        ):
            logger.info("LLM json_object mode not supported. Switching to text mode.")
            _llm_state["json_mode_supported"] = False
            kwargs.pop("response_format", None)
            response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
            if hasattr(response, "usage") and response.usage:
                _llm_state["total_tokens"] += getattr(response.usage, "total_tokens", 0) or 0
            _llm_state["total_calls"] += 1
            return response
        raise


def parse_json_response(content: str) -> Any:
    """
    Robustly parse JSON from LLM responses:
    1. Direct parse
    2. Strip markdown fences (```json ... ```)
    3. Regex fallback to find JSON block
    """
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    if "```json" in content:
        parts = content.split("```json")
        if len(parts) > 1:
            content = parts[1].split("```")[0].strip()
    elif "```" in content:
        parts = content.split("```")
        if len(parts) > 1:
            content = parts[1].split("```")[0].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r"(\[.*\]|\{.*\})", content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except (json.JSONDecodeError, Exception):
        pass

    raise json.JSONDecodeError("Could not find valid JSON", content, 0)


def ensure_str(value: Any) -> str:
    """Ensure a value is a string, joining lists if necessary."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)
