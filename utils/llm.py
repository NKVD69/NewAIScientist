"""
LLM utility functions: completion wrapper, JSON parsing, token tracking.
"""

import asyncio
import json
import logging
import random
import re
from typing import Any

import config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-session LLM state (replaces mutable globals)
# ---------------------------------------------------------------------------

_llm_state: dict = {
    "json_mode_supported": True,
    "total_tokens": 0,
    "total_calls": 0,
    "total_retries": 0,
    "last_model": None,
}


def get_llm_usage_stats() -> dict:
    """Return a snapshot of current LLM session usage statistics."""
    return dict(_llm_state)


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

# Default retry policy. Can be overridden per-call via the `retry_*` kwargs.
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 30.0  # seconds


def _is_retryable_error(exc: BaseException) -> bool:
    """
    Heuristic: decide whether an exception thrown by the OpenAI client is
    transient (worth retrying) or permanent (auth/validation/etc.).

    We avoid hard-imports of openai exception classes so this stays usable
    when the package isn't installed; instead we sniff class names and
    error messages.
    """
    name = type(exc).__name__
    msg = str(exc).lower()

    # Explicit non-retryable signals first.
    non_retryable = (
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
        "UnprocessableEntityError",
    )
    if name in non_retryable:
        return False
    if any(tok in msg for tok in ("invalid_api_key", "permission denied", "401", "403", "404")):
        return False

    # Retryable signals.
    retryable_names = (
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "Timeout",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "ReadTimeout",
    )
    if name in retryable_names:
        return True
    if any(tok in msg for tok in (
        "timeout", "timed out", "connection", "rate limit",
        "429", "500", "502", "503", "504",
    )):
        return True

    # When in doubt, do not retry: better to surface unknown errors than
    # to mask bugs by spinning on them.
    return False


def _backoff_delay(attempt: int,
                   base: float = DEFAULT_BASE_DELAY,
                   cap: float = DEFAULT_MAX_DELAY) -> float:
    """Exponential backoff with full jitter."""
    expo = min(cap, base * (2 ** attempt))
    return random.uniform(0, expo)


async def _call_with_retry(client, kwargs: dict,
                           max_retries: int,
                           base_delay: float,
                           max_delay: float):
    """Call client.chat.completions.create with bounded retry on transient failures."""
    attempt = 0
    while True:
        try:
            return await asyncio.to_thread(client.chat.completions.create, **kwargs)
        except Exception as exc:  # noqa: BLE001 — broad on purpose, classified below
            if attempt >= max_retries or not _is_retryable_error(exc):
                raise
            delay = _backoff_delay(attempt, base_delay, max_delay)
            _llm_state["total_retries"] += 1
            logger.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt + 1, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)
            attempt += 1


async def get_llm_completion(
    client,
    messages: list[dict],
    temperature: float = 0.7,
    json_mode: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> Any:
    """
    Robust wrapper for LLM completions with:
    - Automatic JSON mode negotiation (disables if model doesn't support it)
    - Model-change detection to reset json_mode flag
    - Token usage tracking
    - Exponential backoff with jitter on transient failures (network,
      timeouts, 429/5xx). Permanent errors (auth, bad request) fail fast.
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

    kwargs: dict = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await _call_with_retry(
            client, kwargs, max_retries, base_delay, max_delay,
        )
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
            response = await _call_with_retry(
                client, kwargs, max_retries, base_delay, max_delay,
            )
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
