"""
agents/literature.py — LiteratureAgent wrapper.

This module re-exports LiteratureAgent from co_scientist.py to provide
a clean import path under the `agents` package.
For full implementation, see co_scientist.LiteratureAgent.
"""
# Lazy import to avoid circular dependencies during package loading.
from __future__ import annotations

from co_scientist import LiteratureAgent  # noqa: F401

__all__ = ["LiteratureAgent"]
