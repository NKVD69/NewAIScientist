"""
AST-based *quality* filter for LLM-generated experiment code.

⚠ THIS IS NOT A SECURITY BOUNDARY. ⚠

This module inspects only ``Import`` / ``ImportFrom`` nodes against a
blocklist. That is structurally insufficient against a code generator:
Python offers unbounded ways to reach a capability without naming it in an
import statement. All of the following pass this filter unchanged, and were
verified to do so against the shipped implementation::

    importlib.import_module("os")        __import__("os").system("id")
    import http.client                   open("/etc/passwd", "w")
    exec(compile("import os", ...))      ().__class__.__mro__[1].__subclasses__()
    import asyncio                       import pickle
    while True: pass                     bytearray(10**10)

The security boundary is :mod:`utils.sandbox_runner`, which delegates
isolation to the kernel (no network, read-only rootfs, memory/PID/CPU caps,
all capabilities dropped). This filter is retained only because it is cheap
and catches obviously off-task generations before we pay for a container
start.

Callers MUST NOT treat a ``True`` return as authorisation to execute code
outside a sandbox.
"""

import ast
import logging

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


def check_code_safety(code: str) -> tuple[bool, str]:
    """
    AST-based safety check for LLM-generated experimental code.

    Returns
    -------
    (is_safe, reason)
        is_safe: True if code passes all checks.
        reason:  Human-readable explanation when is_safe is False.

    Scope
    -----
    * Rejects code that fails to parse (a real, cheap win).
    * Flags imports of OS-level and network modules, which in a correctly
      sandboxed run indicate the generator misunderstood the task rather
      than that an attack is underway.

    NOT a security control. See the module docstring: ten trivial bypasses
    are documented there. Execution safety comes from
    ``utils.sandbox_runner.run_sandboxed``, never from this function.
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
