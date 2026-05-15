"""
api/security.py
Security helpers for the FastAPI server.

Centralises three concerns:

1. **API-key auth.** ``require_api_key`` is a FastAPI dependency that
   compares the request's ``X-API-Key`` header against the keys listed
   in the ``API_KEYS`` environment variable (comma-separated). If
   ``API_KEYS`` is unset *and* ``ALLOW_UNAUTHENTICATED=1``, the server
   runs in dev mode (no auth, with a loud warning). Otherwise every
   request is rejected with 401.

2. **CORS allowlist.** ``get_cors_origins()`` reads the
   ``CORS_ALLOWED_ORIGINS`` environment variable (comma-separated) and
   falls back to local React/Vite dev ports. The wildcard ``*`` is
   forbidden combined with credentials per the CORS spec.

3. **Safe path resolution.** ``safe_path_within`` resolves a
   user-supplied path and refuses anything that escapes the configured
   root directory (path-traversal defence). ``sanitise_filename``
   strips path components and rejects unsafe characters / extensions.

All helpers are pure (no network, no global state). They log warnings
when the dev-mode escape hatches are taken so issues are visible.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API-key authentication
# ---------------------------------------------------------------------------

_DEV_WARNING_EMITTED = False


def _load_api_keys() -> set[str]:
    raw = os.environ.get("API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def _dev_mode_enabled() -> bool:
    return os.environ.get("ALLOW_UNAUTHENTICATED", "").strip() == "1"


def _emit_dev_warning_once() -> None:
    global _DEV_WARNING_EMITTED
    if not _DEV_WARNING_EMITTED:
        logger.warning(
            "API_KEYS is empty and ALLOW_UNAUTHENTICATED=1 — running "
            "WITHOUT authentication. Do not deploy this to a public host."
        )
        _DEV_WARNING_EMITTED = True


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency: verifies the ``X-API-Key`` header.

    Returns the validated key so handlers can audit it if needed.
    Raises ``HTTPException(401)`` otherwise.
    """
    valid_keys = _load_api_keys()

    if not valid_keys:
        if _dev_mode_enabled():
            _emit_dev_warning_once()
            return "dev"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Server has no API_KEYS configured. Set the API_KEYS env var "
                "to a comma-separated list, or ALLOW_UNAUTHENTICATED=1 for "
                "development only."
            ),
        )

    if not x_api_key or x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
    return x_api_key


# ---------------------------------------------------------------------------
# CORS allowlist
# ---------------------------------------------------------------------------

_DEFAULT_DEV_ORIGINS = (
    "http://localhost:3000",   # CRA / Next.js default
    "http://localhost:5173",   # Vite default
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
)


def get_cors_origins() -> list[str]:
    """Read the allowed CORS origins from CORS_ALLOWED_ORIGINS env var.

    Refuses to return ``["*"]`` since the server uses ``allow_credentials``
    and the CORS spec forbids that combination. ``*`` collapses to the
    dev allowlist with a warning.
    """
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return list(_DEFAULT_DEV_ORIGINS)
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if "*" in origins:
        logger.warning(
            "CORS_ALLOWED_ORIGINS contained '*'; falling back to dev "
            "allowlist because wildcard + credentials is unsafe."
        )
        return list(_DEFAULT_DEV_ORIGINS)
    return origins


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

_FORBIDDEN_NAME_CHARS = set('\x00\r\n\t/\\:*?"<>|')


def sanitise_filename(name: str, allowed_extensions: Iterable[str]) -> str:
    """Return a safe basename, or raise ``ValueError``.

    Steps:
      - Strip any path components (``Path(name).name`` only).
      - Reject if any forbidden char remains (slashes, NUL, control chars).
      - Reject if the extension is not in ``allowed_extensions``
        (compared case-insensitive, with or without leading dot).
    """
    if not name:
        raise ValueError("empty filename")

    base = Path(name).name  # strips any directory traversal
    if not base or base in (".", ".."):
        raise ValueError("invalid filename")

    if any(c in _FORBIDDEN_NAME_CHARS for c in base):
        raise ValueError(f"filename {base!r} contains forbidden characters")

    allowed = {f".{e.lstrip('.').lower()}" for e in allowed_extensions}
    ext = Path(base).suffix.lower()
    if ext not in allowed:
        raise ValueError(
            f"extension {ext!r} not allowed; expected one of {sorted(allowed)}"
        )
    return base


def safe_path_within(path: str | os.PathLike, root: str | os.PathLike) -> Path:
    """Resolve *path* and require it to live inside *root*.

    Raises ``ValueError`` if the resolved path escapes the root. The root
    itself is created if it does not exist (so callers can use freshly
    configured upload dirs).
    """
    root_p = Path(root).resolve()
    root_p.mkdir(parents=True, exist_ok=True)
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"could not resolve path {path!r}: {exc}") from exc

    try:
        resolved.relative_to(root_p)
    except ValueError as exc:
        raise ValueError(
            f"path {str(resolved)!r} is outside the allowed root {str(root_p)!r}"
        ) from exc
    return resolved


__all__ = [
    "get_cors_origins",
    "require_api_key",
    "safe_path_within",
    "sanitise_filename",
]
