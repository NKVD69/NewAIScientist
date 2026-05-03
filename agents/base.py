"""
Base class for all NewAIScientist agents.
"""
import logging

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Abstract base for all specialized agents.

    Provides:
    - Standardized name attribute (set as class attribute on subclasses)
    - Shared LLM client initialization pattern
    - Logging helpers

    Subclasses should set ``name`` as a class attribute and call
    ``super().__init__(use_local_llm=...)`` from their own ``__init__``.
    """

    name: str = "Base"

    def __init__(self, use_local_llm: bool = True):
        self.llm_client: object | None = None

        if use_local_llm:
            self._init_llm_client()

    def _init_llm_client(self):
        """Initialize the shared LLM client from config."""
        try:
            import openai  # noqa: F401 — imported for side-effect check

            import config as cfg
            self.llm_client = cfg.get_openai_client()
            if self.llm_client is not None:
                logger.info("[%s] LLM client initialized.", self.name)
        except ImportError:
            logger.warning(
                "[%s] openai package not installed. Running in simulation mode.",
                self.name,
            )
        except Exception as exc:
            logger.warning("[%s] Could not connect to LLM: %s", self.name, exc)

    def log(self, msg: str, level: str = "info"):
        """Emit a structured log message prefixed with agent name."""
        log_fn = getattr(logger, level, logger.info)
        log_fn("[%s] %s", self.name, msg)
