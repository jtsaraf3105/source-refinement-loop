"""Configuration from environment variables.

The model is never hardcoded — every LLM call reads settings.openai_model, so
switching models is a .env change only.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.top_n: int = int(os.getenv("TOP_N_CANDIDATES", "5"))
        # Generous default: a broad search can ask the model to score many
        # candidates, which takes well over the old 40s and tripped timeouts.
        self.openai_timeout: float = float(os.getenv("OPENAI_TIMEOUT", "90"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
