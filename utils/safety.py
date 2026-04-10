"""
AST-based code safety utilities for the ExperimentAgent.
"""

import ast
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

DANGEROUS_MODULES: frozenset = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "requests",
    "urllib",
    "ftplib",
    "smtplib",
    "ctypes",
    "winreg",
    "multiprocessing",
    "threading",
    "signal",
    "fcntl",
    "pty",
})


def check_code_safety(code: str) -> Tuple[bool, str]:
    """
    AST-based safety check for LLM-generated experimental code.

    Returns
    -------
    (is_safe, reason)
        is_safe: True if code passes all checks.
        reason:  Human-readable explanation when is_safe is False.

    Security model
    --------------
    * Blocks imports of OS-level and network modules.
    * Does NOT try to sandbox execution completely (use a proper sandbox for that).
    * Intended as a first-pass filter to stop obviously dangerous code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error in generated code: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in DANGEROUS_MODULES:
                    return False, f"Blocked import: '{alias.name}' (restricted module)"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod in DANGEROUS_MODULES:
                    return False, f"Blocked import from: '{node.module}' (restricted module)"

    return True, "OK"
