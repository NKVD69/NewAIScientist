"""
Centralized Configuration Module for AI Co-Scientist
Provides a single source of truth for all environment-based settings.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

# Fix Windows console encoding for Unicode (emoji in agent print() calls)
if sys.platform == "win32":
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

# =============================================================================
# LLM Configuration
# =============================================================================

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LLM_MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_LLM_API_KEY = "lm-studio"
DEFAULT_ENTREZ_EMAIL = "ai-scientist@example.com"
DEFAULT_NCBI_API_KEY = ""


def get_llm_base_url() -> str:
    """Get the LLM API base URL from environment."""
    return os.environ.get("OPENAI_API_BASE", DEFAULT_LLM_BASE_URL)


def get_llm_model_name() -> str:
    """Get the LLM model name from environment."""
    return os.environ.get("OPENAI_MODEL_NAME", DEFAULT_LLM_MODEL_NAME)


def get_llm_api_key() -> str:
    """Get the LLM API key from environment."""
    return os.environ.get("OPENAI_API_KEY", DEFAULT_LLM_API_KEY)


def get_entrez_email() -> str:
    """Get the Entrez email for NCBI API access."""
    return os.environ.get("ENTREZ_EMAIL", DEFAULT_ENTREZ_EMAIL)


def get_ncbi_api_key() -> str:
    """Get the NCBI API key from environment."""
    return os.environ.get("NCBI_API_KEY", DEFAULT_NCBI_API_KEY)


def get_openai_client():
    """Return a pre-configured OpenAI client."""
    try:
        import openai
        return openai.OpenAI(
            base_url=get_llm_base_url(),
            api_key=get_llm_api_key()
        )
    except ImportError:
        logger.error("openai package not installed")
        return None


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(level: int = logging.INFO):
    """Configure structured logging for the entire application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
