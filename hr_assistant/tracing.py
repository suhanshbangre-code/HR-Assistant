"""Langsmith Tracing 

Langsmith tracing needs no wiring in our own ocde - LangChain looks for LANGSMITH_TRACING / LANGSMITH_ENDPOINT / 
LANGSMITH_API_KEY / LANGSMITH_PROJECT directly in the environment (loaded from .env by config.py) and if tracing is turned on,
automatically sends a trace of every LLM call, tool call, and agent step to your LangSmith project.
"""

from hr_assistant import config
from hr_assistant.logger import get_logger
logger = get_logger(__name__)

def check_langsmith_tracing() -> None:
    """Log whether LangSmith tracing is enabled for this run."""
    tracing_on = config.LANGSMITH_TRACING.lower() == "true"

    if tracing_on and config.LANGSMITH_API_KEY:
        logger.info(f"LangSmith tracing ENABLED - project {config.LANGSMITH_PROJECT}, traces at {config.LANGSMITH_ENDPOINT}")
    else:
        logger.info("Lagsmith tracing is OFF (set LANGSMITH_TRACING = true in .env to enable tracing)")