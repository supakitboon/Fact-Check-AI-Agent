import os
from google.adk.models.lite_llm import LiteLlm

_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openrouter/anthropic/claude-3.7-sonnet"


def make_llm(vision: bool = False) -> LiteLlm:
    """Return a configured LiteLlm instance for OpenRouter."""
    env_key = "OPENROUTER_VISION_MODEL" if vision else "OPENROUTER_MODEL"
    return LiteLlm(
        model=os.getenv(env_key, _DEFAULT_MODEL),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base=_OPENROUTER_API_BASE,
    )
