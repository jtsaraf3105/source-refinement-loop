"""The single wrapper around the OpenAI SDK for structured-output calls.

Everything the app consumes from the LLM goes through structured_call, which
reads the model from config, forces the response into a Pydantic schema, and
turns every failure mode into one LLMError the UI can show cleanly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Type, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from config import get_settings

logger = logging.getLogger("sourcing")

T = TypeVar("T", bound=BaseModel)
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class LLMError(Exception):
    """User-safe error. `message` is shown to the recruiter; the original cause
    is logged, never surfaced."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise LLMError("No OpenAI API key configured. Set OPENAI_API_KEY in your .env.")
    # The SDK retries transient network/5xx errors on its own with backoff.
    return OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout, max_retries=1)


def structured_call(system_prompt: str, user_content: str, schema: Type[T]) -> T:
    """Make one structured-output call and return a validated Pydantic object."""
    settings = get_settings()
    try:
        completion = _client().beta.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format=schema,
            temperature=0.2,  # low → more reproducible rankings
        )
    except AuthenticationError:
        logger.exception("OpenAI authentication failed")
        raise LLMError("Invalid OpenAI API key.")
    except RateLimitError:
        logger.exception("OpenAI rate limit / quota hit")
        raise LLMError("The AI service is rate-limited right now. Please wait a moment and retry.")
    except APITimeoutError:
        logger.exception("OpenAI request timed out")
        raise LLMError("The AI service timed out. Please retry.")
    except APIConnectionError:
        logger.exception("Network error reaching OpenAI")
        raise LLMError("Could not reach the AI service. Check your connection and retry.")
    except Exception:
        logger.exception("Unexpected OpenAI error")
        raise LLMError("The AI service returned an unexpected error. Please retry.")

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        logger.warning("Model refused: %s", message.refusal)
        raise LLMError("The AI service declined this request.")

    parsed = message.parsed
    if parsed is None:
        logger.error("Model returned empty/unparseable output")
        raise LLMError("The AI service returned an empty or malformed response. Please retry.")

    try:
        # `parsed` is already the schema type; re-validate defensively.
        return schema.model_validate(parsed.model_dump())
    except ValidationError:
        logger.exception("Structured output failed validation")
        raise LLMError("The AI service returned data in an unexpected shape. Please retry.")
