"""
tests/test_utils.py
Unit tests for utility functions (JSON parsing, LLM state, safety checker).
Fully offline — no LLM or network required.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from utils.llm import ensure_str, get_llm_usage_stats, parse_json_response
from utils.safety import DANGEROUS_MODULES, check_code_safety

# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json_object(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_plain_json_array(self):
        result = parse_json_response('[{"a": 1}, {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_json_with_markdown_fence(self):
        content = '```json\n{"key": "value"}\n```'
        result = parse_json_response(content)
        assert result == {"key": "value"}

    def test_json_with_plain_fence(self):
        content = '```\n{"key": 42}\n```'
        result = parse_json_response(content)
        assert result == {"key": 42}

    def test_json_embedded_in_preamble(self):
        content = 'Here is the result:\n{"answer": true}'
        result = parse_json_response(content)
        assert result["answer"] is True

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_json_response("this is not json at all")

    def test_nested_json(self):
        raw = '{"hypotheses": [{"title": "H1"}, {"title": "H2"}]}'
        result = parse_json_response(raw)
        assert len(result["hypotheses"]) == 2


# ---------------------------------------------------------------------------
# ensure_str
# ---------------------------------------------------------------------------

class TestEnsureStr:
    def test_string_passthrough(self):
        assert ensure_str("hello") == "hello"

    def test_none_returns_empty(self):
        assert ensure_str(None) == ""

    def test_list_joins(self):
        assert ensure_str(["a", "b", "c"]) == "a b c"

    def test_int_converts(self):
        result = ensure_str(42)
        assert result == "42"

    def test_empty_list(self):
        assert ensure_str([]) == ""


# ---------------------------------------------------------------------------
# check_code_safety
# ---------------------------------------------------------------------------

class TestCheckCodeSafety:
    def test_safe_code_passes(self):
        code = """
import numpy as np
import scipy.stats as st
x = np.random.randn(100)
print(np.mean(x))
"""
        is_safe, reason = check_code_safety(code)
        assert is_safe, f"Expected safe, got: {reason}"

    def test_blocked_os_import(self):
        code = "import os\nos.remove('data.csv')"
        is_safe, reason = check_code_safety(code)
        assert not is_safe
        assert "os" in reason

    def test_blocked_subprocess_import(self):
        code = "import subprocess\nsubprocess.run(['rm', '-rf', '/'])"
        is_safe, reason = check_code_safety(code)
        assert not is_safe
        assert "subprocess" in reason

    def test_blocked_from_import(self):
        code = "from pathlib import Path\nPath('/etc/passwd').read_text()"
        is_safe, reason = check_code_safety(code)
        assert not is_safe
        assert "pathlib" in reason

    def test_blocked_socket(self):
        code = "import socket\ns = socket.socket()"
        is_safe, reason = check_code_safety(code)
        assert not is_safe

    def test_syntax_error_caught(self):
        code = "def broken(:\n    pass"
        is_safe, reason = check_code_safety(code)
        assert not is_safe
        assert "Syntax error" in reason

    def test_dotted_import_blocked(self):
        code = "import os.path"
        is_safe, reason = check_code_safety(code)
        assert not is_safe  # os is the top-level module

    def test_dangerous_modules_set_non_empty(self):
        assert len(DANGEROUS_MODULES) > 0
        assert "os" in DANGEROUS_MODULES
        assert "subprocess" in DANGEROUS_MODULES


# ---------------------------------------------------------------------------
# get_llm_usage_stats
# ---------------------------------------------------------------------------

class TestLLMUsageStats:
    def test_returns_dict(self):
        stats = get_llm_usage_stats()
        assert isinstance(stats, dict)

    def test_has_expected_keys(self):
        stats = get_llm_usage_stats()
        assert "total_tokens" in stats
        assert "total_calls" in stats
        assert "json_mode_supported" in stats

    def test_returns_snapshot_not_reference(self):
        """Mutating the returned dict should not affect internal state."""
        stats = get_llm_usage_stats()
        stats["total_tokens"] = 999999
        stats2 = get_llm_usage_stats()
        assert stats2["total_tokens"] != 999999
