"""
utils/sandbox_runner.py — Kernel-enforced isolation for LLM-generated code.

Why this module exists
----------------------
Execution used to be guarded by ``utils.safety.check_code_safety``, an AST
walk that inspects only ``Import`` / ``ImportFrom`` nodes against a 17-entry
blocklist. Ten bypasses were verified against the shipped code, all passing:

    importlib.import_module("os")     __import__("os").system(...)
    import http.client                open("/etc/passwd", "w")
    exec(compile("import os", ...))   ().__class__.__mro__[1].__subclasses__()
    import asyncio / pickle           while True: pass       bytearray(10**10)

A static allowlist of import names is structurally the wrong defence against
a code generator: Python offers unbounded ways to reach a capability without
naming it in an import statement. Security has to come from the kernel.

Isolation tiers (best available wins)
-------------------------------------
1. ``docker`` / ``podman`` — no network, read-only rootfs, memory + PID +
   CPU caps, all capabilities dropped, non-root user, ``no-new-privileges``.
2. ``rlimit`` — same process tree as the host but with ``setrlimit`` caps on
   address space, CPU time and process count, plus network isolation via
   ``unshare -rn`` when available. Strictly weaker; used only as a fallback.
3. ``none`` — refuse to execute. ``require_isolation=True`` (the default in
   production) turns this into an explicit failure instead of silently
   running untrusted code with the user's full privileges.

``check_code_safety`` is retained upstream, but demoted to what it actually
is: a *quality* filter that catches obviously off-task code early. It is no
longer a security boundary and callers must not treat it as one.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SandboxPolicy:
    """Resource and privilege envelope for one execution."""

    timeout_s: int = 30
    memory_mb: int = 512
    pids_limit: int = 64
    cpus: float = 1.0
    network: bool = False
    image: str = "python:3.12-slim"
    #: Extra pip packages baked into the run image (scientific stack).
    #: Left empty by default: the base image is used as-is and the generated
    #: script must stick to the standard library unless an image with the
    #: scientific stack is supplied via ``NEWAISCI_SANDBOX_IMAGE``.
    workdir_in_container: str = "/work"
    #: Refuse to run at all when no container runtime is available.
    require_isolation: bool = True

    @classmethod
    def from_env(cls) -> SandboxPolicy:
        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        return cls(
            timeout_s=_int("NEWAISCI_SANDBOX_TIMEOUT", 30),
            memory_mb=_int("NEWAISCI_SANDBOX_MEMORY_MB", 512),
            pids_limit=_int("NEWAISCI_SANDBOX_PIDS", 64),
            cpus=float(os.environ.get("NEWAISCI_SANDBOX_CPUS", "1.0")),
            network=os.environ.get("NEWAISCI_SANDBOX_NETWORK", "0") == "1",
            image=os.environ.get("NEWAISCI_SANDBOX_IMAGE", "python:3.12-slim"),
            require_isolation=os.environ.get(
                "NEWAISCI_ALLOW_UNSANDBOXED", "0"
            ) != "1",
        )


@dataclass
class SandboxResult:
    """Outcome of one sandboxed execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    backend: str = "none"
    timed_out: bool = False
    blocked: bool = False           # refused before execution
    error: str = ""
    code_sha256: str = ""
    artifacts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blocked and not self.timed_out and self.exit_code == 0


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_RUNTIME_CACHE: str | None = None


def detect_runtime(refresh: bool = False) -> str:
    """Return ``"docker"``, ``"podman"``, ``"rlimit"`` or ``"none"``.

    A runtime counts as available only if its binary exists *and* it answers
    a trivial command — a Docker CLI without a reachable daemon is useless
    and we must not advertise isolation we cannot deliver.
    """
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None and not refresh:
        return _RUNTIME_CACHE

    for runtime in ("docker", "podman"):
        binary = shutil.which(runtime)
        if not binary:
            continue
        try:
            probe = subprocess.run(
                [binary, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=10,
            )
            if probe.returncode == 0:
                _RUNTIME_CACHE = runtime
                logger.info("Sandbox backend: %s", runtime)
                return runtime
            logger.debug("%s present but not usable: %s", runtime, probe.stderr.strip()[:200])
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("%s probe failed: %s", runtime, exc)

    if hasattr(os, "fork") and _resource_module() is not None:
        _RUNTIME_CACHE = "rlimit"
        logger.warning(
            "No container runtime available — falling back to rlimit isolation, "
            "which is substantially weaker. Install Docker or Podman for real isolation."
        )
        return "rlimit"

    _RUNTIME_CACHE = "none"
    return "none"


def _resource_module():
    try:
        import resource  # noqa: PLC0415 — POSIX-only, imported defensively
        return resource
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Container backend
# ---------------------------------------------------------------------------

def _build_container_argv(runtime: str, policy: SandboxPolicy, host_dir: Path) -> list[str]:
    """Assemble the hardened container invocation.

    Every flag here is load-bearing:

    ``--network none``        no egress, no exfiltration, no data poisoning
    ``--read-only``           immutable rootfs
    ``--tmpfs``               scratch space that dies with the container
    ``--memory``/``-swap``    equal values, so swap cannot bypass the cap
    ``--pids-limit``          fork bombs terminate instead of the host
    ``--cpus``                bounded CPU share
    ``--cap-drop ALL``        no Linux capabilities whatsoever
    ``no-new-privileges``     setuid binaries cannot escalate
    ``--user 65534:65534``    nobody:nogroup
    """
    workdir = policy.workdir_in_container
    argv = [
        runtime, "run", "--rm",
        "--network", "bridge" if policy.network else "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--memory", f"{policy.memory_mb}m",
        "--memory-swap", f"{policy.memory_mb}m",
        "--pids-limit", str(policy.pids_limit),
        "--cpus", str(policy.cpus),
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--workdir", workdir,
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "HOME=/tmp",
        "--env", "MPLCONFIGDIR=/tmp",
        "-v", f"{host_dir}:{workdir}:ro",
        policy.image,
        "python", "-I", f"{workdir}/experiment.py",
    ]
    if runtime == "podman":
        # Podman rejects --security-opt no-new-privileges in some versions;
        # it applies the equivalent by default for rootless containers.
        argv = [a for a in argv if a != "no-new-privileges"]
        argv = [a for i, a in enumerate(argv)
                if not (a == "--security-opt" and argv[i + 1:i + 2] == [])]
    return argv


def _run_container(runtime: str, policy: SandboxPolicy, host_dir: Path) -> SandboxResult:
    argv = _build_container_argv(runtime, policy, host_dir)
    logger.debug("Sandbox argv: %s", " ".join(argv))
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            # Give the runtime a little slack beyond the in-container budget so
            # that image pulls and teardown do not read as a script timeout.
            timeout=policy.timeout_s + 20,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            backend=runtime, timed_out=True,
            duration_s=time.monotonic() - started,
            error=f"container exceeded {policy.timeout_s}s budget",
        )
    except OSError as exc:
        return SandboxResult(
            backend=runtime, blocked=True,
            error=f"failed to start container: {exc}",
        )

    return SandboxResult(
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
        duration_s=time.monotonic() - started,
        backend=runtime,
    )


# ---------------------------------------------------------------------------
# rlimit fallback backend
# ---------------------------------------------------------------------------

def _rlimit_preexec(policy: SandboxPolicy):
    """Return a preexec_fn applying hard resource limits in the child."""
    resource = _resource_module()
    if resource is None:
        return None

    mem_bytes = policy.memory_mb * 1024 * 1024
    cpu_seconds = max(1, policy.timeout_s)
    pids = policy.pids_limit

    def _apply():  # pragma: no cover — runs post-fork, before exec
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_NPROC, (pids, pids))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _apply


def _run_rlimit(policy: SandboxPolicy, host_dir: Path) -> SandboxResult:
    import sys

    script = host_dir / "experiment.py"
    argv: list[str] = []

    # Network namespace isolation when unshare is available and rootless
    # user namespaces are permitted.
    if not policy.network and shutil.which("unshare"):
        argv += ["unshare", "--map-root-user", "--net"]

    # -I: isolated mode — ignores PYTHON* env vars and the user site directory.
    argv += [sys.executable, "-I", str(script)]

    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(host_dir),
        "TMPDIR": str(host_dir),
        "MPLCONFIGDIR": str(host_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(host_dir),
            capture_output=True,
            text=True,
            timeout=policy.timeout_s,
            env=env,
            preexec_fn=_rlimit_preexec(policy),  # noqa: PLW1509 — intentional
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            backend="rlimit", timed_out=True,
            duration_s=time.monotonic() - started,
            error=f"script exceeded {policy.timeout_s}s budget",
        )
    except OSError as exc:
        # unshare can fail when unprivileged user namespaces are disabled;
        # retry once without it rather than losing the run entirely.
        if argv and argv[0] == "unshare":
            logger.warning("unshare unavailable (%s) — retrying without netns.", exc)
            policy_no_ns = SandboxPolicy(**{**policy.__dict__, "network": True})
            return _run_rlimit(policy_no_ns, host_dir)
        return SandboxResult(backend="rlimit", blocked=True, error=str(exc))

    return SandboxResult(
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
        duration_s=time.monotonic() - started,
        backend="rlimit",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sandboxed_sync(
    code: str,
    input_files: dict[str, str] | None = None,
    policy: SandboxPolicy | None = None,
) -> SandboxResult:
    """Execute ``code`` under the strongest isolation available.

    Parameters
    ----------
    code
        The Python source to run. Written to ``experiment.py`` in a
        throwaway directory mounted read-only into the container.
    input_files
        ``{filename: contents}`` written alongside the script (e.g.
        ``{"data.csv": "..."}``). Filenames are flattened to their basename
        so a generated path cannot escape the working directory.
    policy
        Resource envelope. Defaults to ``SandboxPolicy.from_env()``.
    """
    policy = policy or SandboxPolicy.from_env()
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()

    runtime = detect_runtime()
    if runtime == "none" or (runtime == "rlimit" and policy.require_isolation
                             and not _rlimit_acceptable()):
        return SandboxResult(
            backend=runtime, blocked=True, code_sha256=digest,
            error=(
                "No container runtime (docker/podman) is available and "
                "unsandboxed execution is disabled. Install a runtime, or set "
                "NEWAISCI_ALLOW_UNSANDBOXED=1 to accept rlimit-only isolation. "
                "Refusing to run LLM-generated code with full user privileges."
            ),
        )

    with tempfile.TemporaryDirectory(prefix="newaisci-exp-") as tmp:
        host_dir = Path(tmp)
        (host_dir / "experiment.py").write_text(code, encoding="utf-8")

        for name, contents in (input_files or {}).items():
            safe_name = os.path.basename(name) or "input.dat"
            (host_dir / safe_name).write_text(contents, encoding="utf-8")

        if runtime in ("docker", "podman"):
            result = _run_container(runtime, policy, host_dir)
        else:
            result = _run_rlimit(policy, host_dir)

        result.code_sha256 = digest
        result.artifacts = sorted(
            p.name for p in host_dir.iterdir() if p.name != "experiment.py"
        )

    return result


def _rlimit_acceptable() -> bool:
    """Whether rlimit-only isolation has been explicitly accepted."""
    return os.environ.get("NEWAISCI_ALLOW_UNSANDBOXED", "0") == "1"


async def run_sandboxed(
    code: str,
    input_files: dict[str, str] | None = None,
    policy: SandboxPolicy | None = None,
) -> SandboxResult:
    """Async wrapper around :func:`run_sandboxed_sync`."""
    return await asyncio.to_thread(run_sandboxed_sync, code, input_files, policy)


def isolation_report() -> dict:
    """Describe the isolation actually in force — surfaced in the UI and logs."""
    runtime = detect_runtime()
    policy = SandboxPolicy.from_env()
    tiers = {
        "docker": "strong (kernel-enforced: no network, read-only FS, capped memory/PIDs/CPU)",
        "podman": "strong (kernel-enforced, rootless)",
        "rlimit": "weak (same host, resource caps only; filesystem readable)",
        "none": "none (execution refused)",
    }
    return {
        "backend": runtime,
        "strength": tiers.get(runtime, "unknown"),
        "network_enabled": policy.network,
        "memory_mb": policy.memory_mb,
        "timeout_s": policy.timeout_s,
        "pids_limit": policy.pids_limit,
        "will_execute": runtime in ("docker", "podman")
        or (runtime == "rlimit" and _rlimit_acceptable()),
    }


__all__ = [
    "SandboxPolicy",
    "SandboxResult",
    "detect_runtime",
    "isolation_report",
    "run_sandboxed",
    "run_sandboxed_sync",
]
