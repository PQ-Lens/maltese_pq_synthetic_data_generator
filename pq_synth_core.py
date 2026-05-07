"""Compatibility wrapper for the single synthetic PQ generator module.

New code should import from `maltese_pq_synthetic_generator`. This module keeps
older notebook/test imports working while delegating generation to that single
implementation.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

import pandas as pd
import requests

import maltese_pq_synthetic_generator as generator

ANSWER_COLUMNS = generator.ANSWER_COLUMNS
CATEGORY_SPECS = generator.CATEGORY_SPECS
CategorySpec = generator.CategorySpec
ConfiguredModel = generator.ConfiguredModel
DEFAULT_PROVIDER = "gemini"
MPS = generator.MPS
MINISTRIES = generator.MINISTRIES
QUESTION_COLUMNS = generator.QUESTION_COLUMNS
SUPPORTED_PROVIDERS = generator.SUPPORTED_PROVIDERS


def _provider_or_legacy_default(provider: Optional[str]) -> str:
    return provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER


def resolve_provider_and_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve provider/model using the historical Gemini fallback."""

    return generator.resolve_provider_and_model(
        provider=provider,
        model=model,
        default_provider=DEFAULT_PROVIDER,
    )


def call_llm_chat_json(
    messages: Sequence[dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> Any:
    return generator.call_llm_chat_json(
        messages=messages,
        provider=_provider_or_legacy_default(provider),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_set(
    category_id: str,
    n: int = 10,
    temperature: float = 0.7,
    retries: int = 1,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> pd.DataFrame:
    return generator.generate_set(
        category_id=category_id,
        n=n,
        temperature=temperature,
        retries=retries,
        provider=_provider_or_legacy_default(provider),
        model=model,
    )


def run_and_save(
    category_id: str,
    n: int = 10,
    temperature: float = 0.7,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    output_dir: str = "pq_synthetic_outputs",
) -> pd.DataFrame:
    df, _metrics = generator.run_and_save(
        category_id=category_id,
        n=n,
        temperature=temperature,
        provider=_provider_or_legacy_default(provider),
        model=model,
        output_dir=output_dir,
    )
    return df


validate_rows = generator.validate_rows
get_configured_models = generator.get_configured_models
get_configured_model_ids = generator.get_configured_model_ids
resolve_model_choice = generator.resolve_model_choice

__all__ = [
    "ANSWER_COLUMNS",
    "CATEGORY_SPECS",
    "CategorySpec",
    "ConfiguredModel",
    "DEFAULT_PROVIDER",
    "MPS",
    "MINISTRIES",
    "QUESTION_COLUMNS",
    "SUPPORTED_PROVIDERS",
    "call_llm_chat_json",
    "generate_set",
    "get_configured_model_ids",
    "get_configured_models",
    "resolve_model_choice",
    "resolve_provider_and_model",
    "run_and_save",
    "validate_rows",
]
