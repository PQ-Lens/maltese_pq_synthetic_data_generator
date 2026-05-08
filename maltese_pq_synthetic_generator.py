"""Single-file synthetic Maltese Parliamentary Question data generator.

The module can be imported as a Python API or run directly as a CLI. Model
selection is driven by environment variables so a server can configure its
available Ollama, Gemini, and Mistral models once in `.env`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
APPROACH = Literal["zero_shot", "one_shot"]
PROVIDER = Literal["ollama", "gemini", "mistral"]
DEFAULT_MAX_OUTPUT_TOKENS = 8192

SUPPORTED_PROVIDERS: Tuple[str, ...] = ("ollama", "gemini", "mistral")
DEFAULT_PROVIDER = "ollama"

PROVIDER_API_KEY_ENV: Dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

PROVIDER_MODEL_LIST_ENV: Dict[str, str] = {
    "ollama": "OLLAMA_MODELS",
    "gemini": "GEMINI_MODELS",
    "mistral": "MISTRAL_MODELS",
}

PROVIDER_MODEL_ENV: Dict[str, str] = {
    "ollama": "OLLAMA_MODEL",
    "gemini": "GEMINI_MODEL",
    "mistral": "MISTRAL_MODEL",
}

PROVIDER_DEFAULT_MODEL: Dict[str, str] = {
    "ollama": "llama3.1:8b",
    "gemini": "gemini-flash-latest",
    "mistral": "mistral-small-latest",
}

_PROVIDER_OVERRIDE: Optional[str] = None
_MODEL_OVERRIDE: Optional[str] = None
_DEFAULT_N_OVERRIDE: Optional[int] = None


class ProviderRequestError(RuntimeError):
    """Raised when a provider request fails after configured retries."""


@dataclass(frozen=True)
class CategoryFailure:
    """A failed category generation captured during a batch run."""

    category_id: str
    category_name: str
    error: str


class GenerationBatchResult(dict):
    """Dictionary of successful category results with captured failures attached."""

    def __init__(
        self,
        *args: Any,
        failures: Optional[List[CategoryFailure]] = None,
        aggregate_path: Optional[Path] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.failures = failures or []
        self.aggregate_path = aggregate_path


@dataclass(frozen=True)
class ConfiguredModel:
    """A model entry loaded from `.env` or inferred from provider defaults."""

    provider: str
    model: str
    source: str

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def label(self) -> str:
        return f"{self.id} ({self.source})"


@dataclass(frozen=True)
class RuntimeConfig:
    """Effective runtime configuration with secrets excluded."""

    provider: str
    model_name: str
    default_n: int
    default_temperature: float
    default_approach: str
    output_dir: str
    ollama_base_url: str
    max_output_tokens: int
    api_retries: int
    api_retry_backoff: float
    configured_models: List[str]


@dataclass(frozen=True)
class CategorySpec:
    """Metadata describing each output category and its required schema."""

    id: str
    name: str
    kind: str
    columns: List[str]
    instruction: str


class LLMProvider(Protocol):
    def generate_json(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        api_retries: int,
        api_retry_backoff: float,
    ) -> Any:
        ...


QUESTION_COLUMNS = [
    "Date",
    "PQ No.",
    "MP",
    "Ministry (EN)",
    "Title (EN)",
    "Question (EN)",
    "Answer (EN)",
]

ANSWER_COLUMNS = ["Date", "PQ No.", "MP", "Question (EN)", "Answer (EN)"]

CATEGORY_SPECS: Dict[str, CategorySpec] = {
    "A1": CategorySpec(
        "A1",
        "Information-Seeking Questions",
        "question",
        QUESTION_COLUMNS,
        (
            "Generate information-seeking parliamentary questions. The MP is primarily "
            "asking for missing facts such as counts, dates, locations, or status. "
            "Keep it neutral and specific."
        ),
    ),
    "A2": CategorySpec(
        "A2",
        "Assertion / Hidden-Accusation Questions",
        "question",
        QUESTION_COLUMNS,
        (
            "Generate assertion or challenge parliamentary questions with a hidden "
            "accusation or accountability push. The wording should imply criticism "
            "while still being phrased as a question."
        ),
    ),
    "A3": CategorySpec(
        "A3",
        "Request / Directive Questions",
        "question",
        QUESTION_COLUMNS,
        (
            "Generate directive parliamentary questions that ask the Minister or PM "
            "to table documents, provide lists, provide a timeframe, confirm a "
            "contract, publish a breakdown, or take a similar action."
        ),
    ),
    "B1": CategorySpec(
        "B1",
        "Replies (Direct Answers)",
        "answer",
        ANSWER_COLUMNS,
        (
            "Generate answers that directly address the question with the requested "
            "facts or a clear yes/no plus details. Avoid deferrals and referrals."
        ),
    ),
    "B2": CategorySpec(
        "B2",
        "Answers by Implication (Indirect/Partial)",
        "answer",
        ANSWER_COLUMNS,
        (
            "Generate answers that are partial or indirect: give an update, a "
            "condition, or a vague timeline instead of the exact requested item."
        ),
    ),
    "B3": CategorySpec(
        "B3",
        "Non-Replies (Deferral/Referral/Reroute)",
        "answer",
        ANSWER_COLUMNS,
        (
            "Generate non-replies: defer to a future sitting, refer to another PQ, "
            "or reroute to another minister. Do not provide the substance of the "
            "requested information."
        ),
    ),
}

MPS = [
    "Ivan Bartolo",
    "Chris Said",
    "Justin Schembri",
    "Jerome Caruana Cilia",
    "Graziella Attard Previ",
    "Rebekah Borg",
    "Ivan Castillo",
]

MINISTRIES = [
    "Prime Minister",
    "Justice and Reform of the Construction Sector",
    "Environment, Energy and Regeneration of the Grand Harbour",
    "Education, Sport, Youth, Research and Innovation",
    "Home Affairs, Security, Reforms and Equality",
    "Social Policy and Children's Rights",
    "Health and Active Ageing",
    "Economy, European Funds and Lands",
    "Transport, Infrastructure and Public Works",
    "Gozo",
]

SYSTEM_STYLE = (
    "You generate synthetic Maltese parliamentary question (PQ) records in English. "
    "They must be fabricated but plausible, using realistic ministry names and parliamentary phrasing. "
    "Do not use real parliamentary records or real events; do not copy any real PQ verbatim. "
    "Keep the tone formal and consistent with parliamentary replies. JSON output only."
)

ONE_SHOT_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "A1": {
        "Date": "23/04/2026",
        "PQ No.": "23456",
        "MP": "Chris Said",
        "Ministry (EN)": "Health and Active Ageing",
        "Title (EN)": "Community Clinics - Staffing Levels",
        "Question (EN)": "Can the Minister state the current number of doctors assigned to each community clinic by locality?",
        "Answer (EN)": "The requested breakdown is currently being compiled and will be tabled.",
    },
    "A2": {
        "Date": "23/04/2026",
        "PQ No.": "23457",
        "MP": "Jerome Caruana Cilia",
        "Ministry (EN)": "Transport, Infrastructure and Public Works",
        "Title (EN)": "Road Works - Delays",
        "Question (EN)": "Given repeated delays, can the Minister explain why completion deadlines were missed for the scheduled arterial road works?",
        "Answer (EN)": "Works are ongoing and revised timelines are being coordinated with contractors.",
    },
    "A3": {
        "Date": "23/04/2026",
        "PQ No.": "23458",
        "MP": "Rebekah Borg",
        "Ministry (EN)": "Economy, European Funds and Lands",
        "Title (EN)": "Public Contracts - Publication",
        "Question (EN)": "Will the Minister table a full list of consultancy contracts awarded in 2025, including values and beneficiaries?",
        "Answer (EN)": "The information requested will be submitted in a subsequent sitting.",
    },
    "B1": {
        "Date": "23/04/2026",
        "PQ No.": "23459",
        "MP": "Ivan Bartolo",
        "Question (EN)": "How many new social housing units were allocated in Q1 2026 by district?",
        "Answer (EN)": "A total of 312 units were allocated in Q1 2026: Northern Harbour 108, Southern Harbour 79, South Eastern 54, Western 37, Northern 22, Gozo and Comino 12.",
    },
    "B2": {
        "Date": "23/04/2026",
        "PQ No.": "23460",
        "MP": "Graziella Attard Previ",
        "Question (EN)": "When will the national sports complex upgrade be completed?",
        "Answer (EN)": "Works are progressing according to the current phase plan and completion is expected in the coming months, subject to certification.",
    },
    "B3": {
        "Date": "23/04/2026",
        "PQ No.": "23461",
        "MP": "Justin Schembri",
        "Question (EN)": "Provide the full breakdown of pending residency permit applications by nationality.",
        "Answer (EN)": "The Honourable Member is referred to PQ 22890, where related information was provided; further details will be tabled separately.",
    },
}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _default_n() -> int:
    if _DEFAULT_N_OVERRIDE is not None:
        return _DEFAULT_N_OVERRIDE
    return _env_int("PQ_DEFAULT_N", 2)


def _default_temperature() -> float:
    return _env_float("PQ_DEFAULT_TEMPERATURE", 0.7)


def _default_approach() -> APPROACH:
    approach = os.getenv("PQ_DEFAULT_APPROACH", "zero_shot").strip()
    if approach not in ("zero_shot", "one_shot"):
        raise ValueError("PQ_DEFAULT_APPROACH must be 'zero_shot' or 'one_shot'")
    return approach  # type: ignore[return-value]


def _default_output_dir() -> str:
    return os.getenv("PQ_OUTPUT_DIR", "pq_synthetic_outputs")


def _default_max_output_tokens() -> int:
    return _env_int("PQ_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)


def _default_api_retries() -> int:
    return max(0, _env_int("PQ_API_RETRIES", 3))


def _default_api_retry_backoff() -> float:
    return max(0.0, _env_float("PQ_API_RETRY_BACKOFF_SECONDS", 2.0))


def _default_continue_on_error() -> bool:
    return _env_bool("PQ_CONTINUE_ON_ERROR", True)


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _split_env_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    normalized = value.replace("\n", ",").replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _append_unique(target: List[str], value: Optional[str]) -> None:
    if value and value.strip() and value.strip() not in target:
        target.append(value.strip())


def _has_provider_key(provider: str) -> bool:
    env_name = PROVIDER_API_KEY_ENV.get(provider)
    return bool(env_name and os.getenv(env_name))


def _normalize_provider_name(provider: Optional[str], default_provider: str = DEFAULT_PROVIDER) -> str:
    name = (provider or os.getenv("LLM_PROVIDER") or default_provider).strip().lower()
    if name not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unknown provider '{name}'. Supported providers are: {supported}.")
    return name


def _parse_provider_model(value: str) -> Tuple[Optional[str], str]:
    prefix, sep, remainder = value.partition(":")
    if sep and prefix in SUPPORTED_PROVIDERS and remainder:
        return prefix, remainder
    return None, value


def _configured_models_for_provider(provider: str) -> List[ConfiguredModel]:
    models: List[str] = []
    list_env = PROVIDER_MODEL_LIST_ENV[provider]
    single_env = PROVIDER_MODEL_ENV[provider]

    for model in _split_env_list(os.getenv(list_env)):
        _append_unique(models, model)

    _append_unique(models, os.getenv(single_env))

    if os.getenv("LLM_PROVIDER", "").strip().lower() == provider:
        _append_unique(models, os.getenv("LLM_MODEL"))

    if not models:
        if provider == "ollama" or _has_provider_key(provider):
            _append_unique(models, PROVIDER_DEFAULT_MODEL[provider])

    source = list_env if os.getenv(list_env) else single_env if os.getenv(single_env) else "default"
    return [ConfiguredModel(provider=provider, model=model, source=source) for model in models]


def get_configured_models() -> List[ConfiguredModel]:
    """Return configured model choices, using `.env` provider lists where present.

    Supported list variables:
    - OLLAMA_MODELS=llama3.1:8b,mistral:7b
    - GEMINI_MODELS=gemini-2.5-flash,gemini-flash-latest
    - MISTRAL_MODELS=mistral-small-latest,mistral-large-latest
    """

    configured: List[ConfiguredModel] = []
    seen: set[str] = set()
    for provider in SUPPORTED_PROVIDERS:
        for model in _configured_models_for_provider(provider):
            if model.id not in seen:
                configured.append(model)
                seen.add(model.id)

    if not configured:
        configured.append(
            ConfiguredModel(
                provider=DEFAULT_PROVIDER,
                model=PROVIDER_DEFAULT_MODEL[DEFAULT_PROVIDER],
                source="default",
            )
        )
    return configured


def get_configured_model_ids() -> List[str]:
    return [model.id for model in get_configured_models()]


def resolve_model_choice(selection: Optional[str] = None) -> ConfiguredModel:
    """Resolve a CLI model selection against configured `.env` choices."""

    models = get_configured_models()
    if not selection:
        env_selection = os.getenv("LLM_MODEL_ID")
        if env_selection:
            return resolve_model_choice(env_selection)

        provider = _normalize_provider_name(_PROVIDER_OVERRIDE, default_provider=DEFAULT_PROVIDER)
        model_override = _MODEL_OVERRIDE or os.getenv("LLM_MODEL")
        if model_override:
            provider_from_model, model_name = _parse_provider_model(model_override)
            provider = provider_from_model or provider
            return ConfiguredModel(provider=provider, model=model_name, source="runtime")

        for model in models:
            if model.provider == provider:
                return model
        return models[0]

    raw = selection.strip()
    for model in models:
        if raw == model.id:
            return model

    provider_from_model, model_name = _parse_provider_model(raw)
    if provider_from_model:
        return ConfiguredModel(provider=provider_from_model, model=model_name, source="argument")

    bare_matches = [model for model in models if model.model == raw]
    if len(bare_matches) == 1:
        return bare_matches[0]
    if len(bare_matches) > 1:
        choices = ", ".join(model.id for model in bare_matches)
        raise ValueError(f"Model name '{raw}' is ambiguous. Use one of: {choices}")

    configured = ", ".join(model.id for model in models)
    raise ValueError(f"Unknown model '{raw}'. Configured models are: {configured}")


def resolve_provider_and_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    default_provider: str = DEFAULT_PROVIDER,
) -> Tuple[str, str]:
    """Resolve provider/model from args, overrides, and `.env` configuration."""

    selected_model = model_name or model or _MODEL_OVERRIDE
    provider_name = provider or _PROVIDER_OVERRIDE

    if selected_model:
        parsed_provider, parsed_model = _parse_provider_model(selected_model)
        if parsed_provider and not provider_name:
            provider_name = parsed_provider
        elif parsed_provider and provider_name and parsed_provider != provider_name:
            raise ValueError(
                f"Model '{selected_model}' specifies provider '{parsed_provider}' "
                f"but provider '{provider_name}' was selected."
            )
        selected_model = parsed_model

    resolved_provider = _normalize_provider_name(provider_name, default_provider=default_provider)

    if selected_model:
        return resolved_provider, selected_model

    matching = [entry for entry in get_configured_models() if entry.provider == resolved_provider]
    if matching:
        return resolved_provider, matching[0].model

    return resolved_provider, PROVIDER_DEFAULT_MODEL[resolved_provider]


def set_model(model_name: str) -> None:
    """Override the active model name for subsequent imported API calls."""

    global _MODEL_OVERRIDE
    _MODEL_OVERRIDE = model_name
    print(f"Model set to: {_MODEL_OVERRIDE}")


def set_provider(provider: PROVIDER) -> None:
    """Override the active provider for subsequent imported API calls."""

    global _PROVIDER_OVERRIDE
    _normalize_provider_name(provider)
    _PROVIDER_OVERRIDE = provider
    print(f"Provider set to: {_PROVIDER_OVERRIDE}")


def set_default_n(n: int) -> None:
    """Override the default row count used by helper functions."""

    global _DEFAULT_N_OVERRIDE
    if n <= 0:
        raise ValueError("n must be > 0")
    _DEFAULT_N_OVERRIDE = n
    print(f"default_n set to: {_DEFAULT_N_OVERRIDE}")


def active_config() -> Dict[str, Any]:
    """Return effective runtime configuration with secrets masked/excluded."""

    provider, model_name = resolve_provider_and_model()
    cfg = RuntimeConfig(
        provider=provider,
        model_name=model_name,
        default_n=_default_n(),
        default_temperature=_default_temperature(),
        default_approach=_default_approach(),
        output_dir=_default_output_dir(),
        ollama_base_url=_ollama_base_url(),
        max_output_tokens=_default_max_output_tokens(),
        api_retries=_default_api_retries(),
        api_retry_backoff=_default_api_retry_backoff(),
        configured_models=get_configured_model_ids(),
    )
    return asdict(cfg)


def _messages_to_text(messages: Sequence[Dict[str, str]]) -> str:
    """Flatten chat messages for providers that accept a single text prompt."""

    chunks = []
    for message in messages:
        role = (message.get("role") or "user").upper()
        content = message.get("content", "")
        chunks.append(f"{role}:\n{content}")
    chunks.append("Return valid JSON only. Do not include markdown fences.")
    return "\n\n".join(chunks)


def _extract_json_from_text(text: str) -> Any:
    """Parse JSON even when a model wraps it in fences or extra text."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [index for index in (cleaned.find("["), cleaned.find("{")) if index != -1]
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if not starts or end == -1:
            raise
        return json.loads(cleaned[min(starts) : end + 1])


def _normalize_rows_payload(payload: Any) -> List[Dict[str, Any]]:
    """Normalize common LLM JSON shapes to the list-of-rows form."""

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("rows", "data", "items", "results", "records", "output"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        if payload:
            return [payload]

    raise ValueError(f"Could not normalize model output to rows. Got: {type(payload).__name__}")


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        return "".join(chunks)

    return ""


def _redact_secrets(value: Any) -> str:
    text = str(value)
    text = re.sub(r"([?&](?:key|api_key)=)[^&\s]+", r"\1<redacted>", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(\b(?:key|api_key|token|access_token)=)[^&\s,;]+",
        r"\1<redacted>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(Authorization:\s*Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1<redacted>", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1<redacted>", text)

    for env_name in ("GOOGLE_API_KEY", "MISTRAL_API_KEY"):
        secret = os.getenv(env_name)
        if secret:
            text = text.replace(secret, "<redacted>")

    return text


def _shorten(text: str, limit: int = 500) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _gemini_usage_summary(data: Dict[str, Any]) -> str:
    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        return "usage unavailable"

    parts = []
    for key in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount"):
        value = usage.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "usage unavailable"


def _provider_http_error(provider_name: str, response: requests.Response) -> ProviderRequestError:
    status = getattr(response, "status_code", "unknown")
    reason = getattr(response, "reason", "") or ""
    body = _shorten(_redact_secrets(getattr(response, "text", "") or ""))
    detail = f" {reason}" if reason else ""
    message = f"{provider_name} request failed with HTTP {status}{detail}"
    if body:
        message = f"{message}: {body}"
    return ProviderRequestError(message)


def _retry_delay(api_retry_backoff: float, attempt_index: int) -> float:
    if api_retry_backoff <= 0:
        return 0.0
    return min(api_retry_backoff * (2 ** attempt_index), 30.0)


def _print_retry(provider_name: str, reason: str, delay: float, next_attempt: int, total_attempts: int) -> None:
    print(
        f"{provider_name} request failed ({_redact_secrets(reason)}). "
        f"Retrying in {delay:.1f}s ({next_attempt}/{total_attempts})..."
    )


def _post_json_with_retries(
    provider_name: str,
    url: str,
    *,
    api_retries: int,
    api_retry_backoff: float,
    **kwargs: Any,
) -> Any:
    attempts = max(1, api_retries + 1)

    for attempt_index in range(attempts):
        try:
            response = requests.post(url, **kwargs)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                status_code = getattr(response, "status_code", None)
                retryable = status_code in TRANSIENT_HTTP_STATUS_CODES
                if retryable and attempt_index < attempts - 1:
                    delay = _retry_delay(api_retry_backoff, attempt_index)
                    _print_retry(
                        provider_name,
                        f"HTTP {status_code}",
                        delay,
                        attempt_index + 2,
                        attempts,
                    )
                    if delay:
                        time.sleep(delay)
                    continue
                raise _provider_http_error(provider_name, response) from exc

            return response.json()

        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt_index < attempts - 1:
                delay = _retry_delay(api_retry_backoff, attempt_index)
                _print_retry(
                    provider_name,
                    exc.__class__.__name__,
                    delay,
                    attempt_index + 2,
                    attempts,
                )
                if delay:
                    time.sleep(delay)
                continue
            raise ProviderRequestError(
                f"{provider_name} request failed after {attempts} attempts: {_redact_secrets(exc)}"
            ) from exc
        except requests.RequestException as exc:
            raise ProviderRequestError(f"{provider_name} request failed: {_redact_secrets(exc)}") from exc

    raise ProviderRequestError(f"{provider_name} request failed after {attempts} attempts.")


class GeminiProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_json(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        api_retries: int,
        api_retry_backoff: float,
    ) -> Any:
        payload = {
            "contents": [{"parts": [{"text": _messages_to_text(messages)}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }

        data = _post_json_with_retries(
            "Gemini",
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
            api_retries=api_retries,
            api_retry_backoff=api_retry_backoff,
        )

        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError(f"Gemini returned no candidates: {data}")

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        if not text:
            finish_detail = f" finishReason={finish_reason};" if finish_reason else ""
            raise ValueError(
                f"Gemini response had no text content.{finish_detail} {_gemini_usage_summary(data)}"
            )

        try:
            return _normalize_rows_payload(_extract_json_from_text(text))
        except json.JSONDecodeError as exc:
            if finish_reason == "MAX_TOKENS":
                raise ValueError(
                    f"Gemini stopped at maxOutputTokens={max_tokens} before completing valid JSON "
                    f"(finishReason=MAX_TOKENS; {_gemini_usage_summary(data)}). "
                    "Increase PQ_MAX_OUTPUT_TOKENS or pass --max-output-tokens."
                ) from exc
            raise


class MistralProvider:
    API_URL = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_json(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        api_retries: int,
        api_retry_backoff: float,
    ) -> Any:
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        data = _post_json_with_retries(
            "Mistral",
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
            api_retries=api_retries,
            api_retry_backoff=api_retry_backoff,
        )

        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Mistral returned no choices: {data}")

        message = choices[0].get("message", {})
        text = _normalize_message_content(message.get("content")).strip()
        if not text:
            raise ValueError(f"Mistral response had no text content: {data}")

        return _normalize_rows_payload(_extract_json_from_text(text))


class OllamaProvider:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate_json(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        api_retries: int,
        api_retry_backoff: float,
    ) -> Any:
        payload = {
            "model": model,
            "messages": list(messages),
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        data = _post_json_with_retries(
            "Ollama",
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
            api_retries=api_retries,
            api_retry_backoff=api_retry_backoff,
        )

        text = (data.get("message") or {}).get("content", "").strip()
        if not text:
            raise ValueError(f"Ollama returned empty content: {data}")

        return _normalize_rows_payload(_extract_json_from_text(text))


def _get_required_api_key(provider_name: str) -> str:
    api_key_env_name = PROVIDER_API_KEY_ENV[provider_name]
    api_key = os.getenv(api_key_env_name)
    if not api_key:
        raise ValueError(
            f"Missing required environment variable '{api_key_env_name}' for provider '{provider_name}'."
        )
    return api_key


def _build_provider(provider_name: str) -> LLMProvider:
    if provider_name == "ollama":
        return OllamaProvider(base_url=_ollama_base_url())
    if provider_name == "gemini":
        return GeminiProvider(api_key=_get_required_api_key("gemini"))
    if provider_name == "mistral":
        return MistralProvider(api_key=_get_required_api_key("mistral"))
    raise ValueError(f"Unknown provider: {provider_name}")


def call_llm_chat_json(
    messages: Sequence[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2500,
    api_retries: Optional[int] = None,
    api_retry_backoff: Optional[float] = None,
) -> Any:
    """Dispatch a chat request to Ollama, Gemini, or Mistral."""

    provider_name, resolved_model = resolve_provider_and_model(
        provider=provider,
        model=model,
        model_name=model_name,
    )
    client = _build_provider(provider_name)
    return client.generate_json(
        messages=messages,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_retries=_default_api_retries() if api_retries is None else max(0, api_retries),
        api_retry_backoff=(
            _default_api_retry_backoff()
            if api_retry_backoff is None
            else max(0.0, api_retry_backoff)
        ),
    )


def build_messages(spec: CategorySpec, n: int, approach: APPROACH) -> List[Dict[str, str]]:
    """Build a prompt that enforces category, schema, and style constraints."""

    rows_shape = (
        f'a bare JSON array with exactly {n} objects, or a JSON object with '
        f'a top-level "rows" array containing exactly {n} objects'
    )
    base = (
        f"Create {n} synthetic Maltese PQ-style records for category {spec.id}: {spec.name}.\n"
        f"Category instruction: {spec.instruction}\n\n"
        f"Output must be {rows_shape}.\n"
        f"Each object must have exactly these keys: {json.dumps(spec.columns)}\n"
        "Date format: DD/MM/YYYY. PQ No. must be numeric. Use plausible ranges such as 10000-40000.\n"
        f"MP must be one of: {json.dumps(MPS)}\n"
        f"If Ministry (EN) exists, it must be one of: {json.dumps(MINISTRIES)}\n"
        "Titles should be short and look like Maltese PQ titles using 'Topic - Detail'.\n"
        "Questions and answers must be formal parliamentary English.\n"
        "Content must be fictional. Do not use real cases, real events, or copied parliamentary records.\n"
        "Return JSON only. Do not include commentary or markdown."
    )

    messages = [{"role": "system", "content": SYSTEM_STYLE}]

    if approach == "one_shot":
        example = ONE_SHOT_EXAMPLES[spec.id]
        messages.append(
            {
                "role": "user",
                "content": (
                    "Here is one valid example row for format and style reference only:\n"
                    f"{json.dumps(example, ensure_ascii=False)}\n\n"
                    "Generate new fictional records. Do not copy this example."
                ),
            }
        )

    messages.append({"role": "user", "content": base})
    return messages


def validate_rows(rows: List[Dict[str, Any]], columns: List[str], n: int) -> List[str]:
    """Validate row count, schema, and a few field-level constraints."""

    errors: List[str] = []
    if not isinstance(rows, list):
        return [f"Output is not a normalized row list. Got: {type(rows).__name__}"]

    if len(rows) != n:
        errors.append(f"Expected {n} rows, got {len(rows)}.")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Row {index} is not an object.")
            continue

        missing = [column for column in columns if column not in row]
        extra = [key for key in row.keys() if key not in columns]
        if missing:
            errors.append(f"Row {index} missing columns: {missing}")
        if extra:
            errors.append(f"Row {index} has extra columns: {extra}")
        if "Date" in row and isinstance(row.get("Date"), str) and not DATE_RE.match(row["Date"]):
            errors.append(f"Row {index} invalid Date: {row.get('Date')}")
        if "PQ No." in row and not str(row.get("PQ No.")).isdigit():
            errors.append(f"Row {index} non-numeric PQ No.: {row.get('PQ No.')}")

    return errors


def generate_set(
    category_id: str,
    n: Optional[int] = None,
    temperature: Optional[float] = None,
    approach: Optional[APPROACH] = None,
    retries: int = 1,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    api_retries: Optional[int] = None,
    api_retry_backoff: Optional[float] = None,
) -> pd.DataFrame:
    """Generate a validated dataframe for the requested category."""

    if category_id not in CATEGORY_SPECS:
        raise KeyError(f"Unknown category_id: {category_id}. Use {list(CATEGORY_SPECS.keys())}")

    row_count = n if n is not None else _default_n()
    if row_count <= 0:
        raise ValueError("n must be > 0")

    resolved_temperature = temperature if temperature is not None else _default_temperature()
    resolved_approach = approach if approach is not None else _default_approach()
    if resolved_approach not in ("zero_shot", "one_shot"):
        raise ValueError("approach must be 'zero_shot' or 'one_shot'")

    provider_name, resolved_model = resolve_provider_and_model(
        provider=provider,
        model=model,
        model_name=model_name,
    )
    token_limit = max_tokens if max_tokens is not None else _default_max_output_tokens()

    spec = CATEGORY_SPECS[category_id]
    messages = build_messages(spec, n=row_count, approach=resolved_approach)

    last_errors: List[str] = []
    for attempt_index in range(retries + 1):
        try:
            rows = call_llm_chat_json(
                messages=messages,
                provider=provider_name,
                model=resolved_model,
                temperature=resolved_temperature,
                max_tokens=token_limit,
                api_retries=api_retries,
                api_retry_backoff=api_retry_backoff,
            )
        except ValueError as exc:
            last_errors = [f"Provider returned invalid JSON or an unusable response: {_shorten(_redact_secrets(exc))}"]
            if attempt_index >= retries:
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous response could not be parsed as usable JSON. "
                        "Return only valid JSON with exactly the requested schema and row count.\n"
                        "- "
                        + "\n- ".join(last_errors)
                    ),
                }
            )
            time.sleep(0.2)
            continue

        errors = validate_rows(rows, spec.columns, n=row_count)
        if not errors:
            return pd.DataFrame(rows, columns=spec.columns)

        last_errors = errors
        messages.append(
            {
                "role": "user",
                "content": (
                    "Fix the validation errors and return valid JSON only. "
                    "Keep exactly the required keys and exactly the requested row count.\n"
                    "- "
                    + "\n- ".join(errors)
                ),
            }
        )
        time.sleep(0.2)

    raise ValueError("Failed validation:\n- " + "\n- ".join(last_errors))


def evaluate_generation(df: pd.DataFrame, spec: CategorySpec) -> Dict[str, float]:
    """Compute simple structural quality metrics for generated output."""

    n = len(df)
    if n == 0:
        return {"n": 0.0, "schema_valid_rate": 0.0}

    expected_cols = spec.columns
    schema_valid = 1.0 if list(df.columns) == expected_cols else 0.0

    date_valid = df["Date"].astype(str).str.match(DATE_RE).mean() if "Date" in df.columns else 0.0
    pq_numeric = df["PQ No."].astype(str).str.isdigit().mean() if "PQ No." in df.columns else 0.0
    mp_allowed = df["MP"].astype(str).isin(MPS).mean() if "MP" in df.columns else 0.0

    ministry_allowed = 1.0
    if "Ministry (EN)" in df.columns:
        ministry_allowed = df["Ministry (EN)"].astype(str).isin(MINISTRIES).mean()

    non_empty_q = (
        df["Question (EN)"].astype(str).str.strip().ne("").mean()
        if "Question (EN)" in df.columns
        else 0.0
    )
    non_empty_a = (
        df["Answer (EN)"].astype(str).str.strip().ne("").mean()
        if "Answer (EN)" in df.columns
        else 0.0
    )

    dup_ratio = df.astype(str).duplicated().mean()
    avg_q_len = df["Question (EN)"].astype(str).str.len().mean() if "Question (EN)" in df.columns else 0.0
    avg_a_len = df["Answer (EN)"].astype(str).str.len().mean() if "Answer (EN)" in df.columns else 0.0

    score = (
        0.20 * schema_valid
        + 0.15 * date_valid
        + 0.15 * pq_numeric
        + 0.15 * mp_allowed
        + 0.10 * ministry_allowed
        + 0.125 * non_empty_q
        + 0.125 * non_empty_a
    )

    return {
        "n": float(n),
        "schema_valid_rate": float(schema_valid),
        "date_format_valid_rate": float(date_valid),
        "pq_no_numeric_rate": float(pq_numeric),
        "mp_allowed_rate": float(mp_allowed),
        "ministry_allowed_rate": float(ministry_allowed),
        "non_empty_question_rate": float(non_empty_q),
        "non_empty_answer_rate": float(non_empty_a),
        "duplicate_row_ratio": float(dup_ratio),
        "avg_question_length": float(avg_q_len),
        "avg_answer_length": float(avg_a_len),
        "overall_score": float(score),
    }


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "model"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_summary(summary_path: Path, metrics_df: pd.DataFrame) -> None:
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        pd.concat([old, metrics_df], ignore_index=True).to_csv(summary_path, index=False)
    else:
        metrics_df.to_csv(summary_path, index=False)


def _write_failure_metrics(
    category_id: str,
    n: Optional[int],
    temperature: Optional[float],
    approach: Optional[APPROACH],
    provider: str,
    model: str,
    output_dir: Optional[str],
    error: str,
) -> Path:
    spec = CATEGORY_SPECS[category_id]
    row_count = n if n is not None else _default_n()
    resolved_temperature = temperature if temperature is not None else _default_temperature()
    resolved_approach = approach if approach is not None else _default_approach()

    out_dir = Path(output_dir or _default_output_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = (
        f"{category_id}_{resolved_approach}_{provider}_"
        f"{_slugify(model.replace(':', '-'))}_{stamp}_failed"
    )
    metrics_path = out_dir / f"{base}_metrics.csv"
    summary_path = out_dir / "summary_metrics.csv"

    metrics_df = pd.DataFrame(
        [
            {
                "category_id": category_id,
                "category_name": spec.name,
                "approach": resolved_approach,
                "n": row_count,
                "temperature": resolved_temperature,
                "provider": provider,
                "model_name": model,
                "status": "failed",
                "error": _shorten(_redact_secrets(error), limit=1000),
                "timestamp": _utc_timestamp(),
                "schema_valid_rate": 0.0,
                "date_format_valid_rate": 0.0,
                "pq_no_numeric_rate": 0.0,
                "mp_allowed_rate": 0.0,
                "ministry_allowed_rate": 0.0,
                "non_empty_question_rate": 0.0,
                "non_empty_answer_rate": 0.0,
                "duplicate_row_ratio": 0.0,
                "avg_question_length": 0.0,
                "avg_answer_length": 0.0,
                "overall_score": 0.0,
            }
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)
    _append_summary(summary_path, metrics_df)
    return metrics_path


def _aggregate_columns() -> List[str]:
    columns = ["Category ID", "Category Name", "Category Type"]
    for column in QUESTION_COLUMNS + ANSWER_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def _write_aggregate_csv(
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: Optional[str],
    approach: Optional[APPROACH],
    provider: str,
    model: str,
) -> Optional[Path]:
    if not results:
        return None

    resolved_approach = approach if approach is not None else _default_approach()
    out_dir = Path(output_dir or _default_output_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for category_id, (df, _metrics_df) in results.items():
        spec = CATEGORY_SPECS[category_id]
        enriched = df.copy()
        enriched.insert(0, "Category Type", spec.kind)
        enriched.insert(0, "Category Name", spec.name)
        enriched.insert(0, "Category ID", spec.id)
        frames.append(enriched)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    preferred_columns = _aggregate_columns()
    extra_columns = [column for column in combined.columns if column not in preferred_columns]
    combined = combined.reindex(columns=preferred_columns + extra_columns)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / (
        f"all_categories_{resolved_approach}_{provider}_"
        f"{_slugify(model.replace(':', '-'))}_{stamp}.csv"
    )
    combined.to_csv(path, index=False)
    return path


def run_and_save(
    category_id: str,
    n: Optional[int] = None,
    temperature: Optional[float] = None,
    approach: Optional[APPROACH] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    retries: int = 1,
    max_tokens: Optional[int] = None,
    api_retries: Optional[int] = None,
    api_retry_backoff: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate rows, evaluate them, and persist data/metrics CSVs."""

    row_count = n if n is not None else _default_n()
    resolved_temperature = temperature if temperature is not None else _default_temperature()
    resolved_approach = approach if approach is not None else _default_approach()
    provider_name, resolved_model = resolve_provider_and_model(
        provider=provider,
        model=model,
        model_name=model_name,
    )

    spec = CATEGORY_SPECS[category_id]
    df = generate_set(
        category_id=category_id,
        n=row_count,
        temperature=resolved_temperature,
        approach=resolved_approach,
        retries=retries,
        provider=provider_name,
        model=resolved_model,
        max_tokens=max_tokens,
        api_retries=api_retries,
        api_retry_backoff=api_retry_backoff,
    )
    metrics = evaluate_generation(df, spec)

    out_dir = Path(output_dir or _default_output_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = (
        f"{category_id}_{resolved_approach}_{provider_name}_"
        f"{_slugify(resolved_model.replace(':', '-'))}_{stamp}"
    )

    data_path = out_dir / f"{base}.csv"
    metrics_path = out_dir / f"{base}_metrics.csv"
    summary_path = out_dir / "summary_metrics.csv"

    df.to_csv(data_path, index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "category_id": category_id,
                "category_name": spec.name,
                "approach": resolved_approach,
                "n": row_count,
                "temperature": resolved_temperature,
                "provider": provider_name,
                "model_name": resolved_model,
                "status": "success",
                "error": "",
                "timestamp": _utc_timestamp(),
                **metrics,
            }
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)
    _append_summary(summary_path, metrics_df)

    print(f"Saved data: {data_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Updated summary: {summary_path}")

    return df, metrics_df


def run_all_and_save(
    n: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
    temperature: Optional[float] = None,
    approach: Optional[APPROACH] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    retries: int = 1,
    max_tokens: Optional[int] = None,
    api_retries: Optional[int] = None,
    api_retry_backoff: Optional[float] = None,
    continue_on_error: Optional[bool] = None,
) -> GenerationBatchResult:
    """Generate every requested category using the same per-category batch size."""

    selected_categories = list(categories or CATEGORY_SPECS.keys())
    unknown = [category for category in selected_categories if category not in CATEGORY_SPECS]
    if unknown:
        raise KeyError(f"Unknown categories: {unknown}. Use {list(CATEGORY_SPECS.keys())}")

    provider_name, resolved_model = resolve_provider_and_model(
        provider=provider,
        model=model,
        model_name=model_name,
    )

    should_continue = _default_continue_on_error() if continue_on_error is None else continue_on_error
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    failures: List[CategoryFailure] = []
    for category_id in selected_categories:
        print(f"\nGenerating {category_id} ({CATEGORY_SPECS[category_id].name})")
        try:
            results[category_id] = run_and_save(
                category_id=category_id,
                n=n,
                temperature=temperature,
                approach=approach,
                provider=provider_name,
                model=resolved_model,
                output_dir=output_dir,
                retries=retries,
                max_tokens=max_tokens,
                api_retries=api_retries,
                api_retry_backoff=api_retry_backoff,
            )
        except Exception as exc:
            error = _redact_secrets(exc)
            failure = CategoryFailure(
                category_id=category_id,
                category_name=CATEGORY_SPECS[category_id].name,
                error=error,
            )
            failures.append(failure)
            metrics_path = _write_failure_metrics(
                category_id=category_id,
                n=n,
                temperature=temperature,
                approach=approach,
                provider=provider_name,
                model=resolved_model,
                output_dir=output_dir,
                error=error,
            )
            print(f"Failed {category_id}: {_shorten(error)}", file=sys.stderr)
            print(f"Saved failure metrics: {metrics_path}", file=sys.stderr)
            if not should_continue:
                aggregate_path = _write_aggregate_csv(
                    results=results,
                    output_dir=output_dir,
                    approach=approach,
                    provider=provider_name,
                    model=resolved_model,
                )
                if aggregate_path:
                    print(f"Saved combined data: {aggregate_path}")
                return GenerationBatchResult(
                    results,
                    failures=failures,
                    aggregate_path=aggregate_path,
                )

    if failures:
        failed = ", ".join(f"{failure.category_id} ({_shorten(failure.error, 120)})" for failure in failures)
        print(f"\nCompleted with failed categories: {failed}", file=sys.stderr)

    aggregate_path = _write_aggregate_csv(
        results=results,
        output_dir=output_dir,
        approach=approach,
        provider=provider_name,
        model=resolved_model,
    )
    if aggregate_path:
        print(f"\nSaved combined data: {aggregate_path}")

    return GenerationBatchResult(
        results,
        failures=failures,
        aggregate_path=aggregate_path,
    )


def compare_approaches(
    category_id: str,
    n: Optional[int] = None,
    temperature: Optional[float] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """Run the same category with both prompt strategies and compare metrics."""

    provider_name, resolved_model = resolve_provider_and_model(
        provider=provider,
        model=model,
        model_name=model_name,
    )
    spec = CATEGORY_SPECS[category_id]
    rows = []
    for approach in ("zero_shot", "one_shot"):
        df = generate_set(
            category_id=category_id,
            n=n,
            temperature=temperature,
            approach=approach,
            provider=provider_name,
            model=resolved_model,
        )
        metrics = evaluate_generation(df, spec)
        for key, value in metrics.items():
            rows.append(
                {
                    "category_id": category_id,
                    "category_name": spec.name,
                    "approach": approach,
                    "metric": key,
                    "value": value,
                    "provider": provider_name,
                    "model_name": resolved_model,
                    "timestamp": _utc_timestamp(),
                }
            )
    return pd.DataFrame(rows)


def _print_configured_models(models: Sequence[ConfiguredModel]) -> None:
    print("Configured models:")
    for index, model in enumerate(models, start=1):
        print(f"  {index}. {model.id} [{model.source}]")


def _prompt_model(models: Sequence[ConfiguredModel]) -> ConfiguredModel:
    _print_configured_models(models)
    while True:
        raw = input(f"Select model [1-{len(models)}] (default 1): ").strip()
        if raw == "":
            return models[0]
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(models):
                return models[index - 1]
        try:
            return resolve_model_choice(raw)
        except ValueError as exc:
            print(exc)


def _prompt_batch_size(default: int) -> int:
    while True:
        raw = input(f"Batch size per category (default {default}): ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Batch size must be an integer.")
            continue
        if value > 0:
            return value
        print("Batch size must be greater than 0.")


def build_arg_parser() -> argparse.ArgumentParser:
    configured = get_configured_model_ids()
    epilog = (
        "Configured models from .env: "
        + (", ".join(configured) if configured else "none")
        + "\nUse OLLAMA_MODELS, GEMINI_MODELS, and MISTRAL_MODELS to control this list."
    )
    parser = argparse.ArgumentParser(
        description="Generate synthetic Maltese PQ records for all selected categories.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        help="Configured model id or unique bare model name, for example ollama:llama3.1:8b.",
    )
    parser.add_argument(
        "-n",
        "--batch-size",
        type=int,
        help="Rows to generate for each selected category.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(CATEGORY_SPECS.keys()),
        default=list(CATEGORY_SPECS.keys()),
        help="Categories to generate. Defaults to all categories.",
    )
    parser.add_argument(
        "--approach",
        choices=["zero_shot", "one_shot"],
        default=_default_approach(),
        help="Prompting approach to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_default_temperature(),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help="Directory where generated CSV files will be written.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=_default_max_output_tokens(),
        help="Maximum output tokens requested from the model.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Validation retry attempts per category when model output has the wrong shape.",
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=_default_api_retries(),
        help="Provider/API retry attempts for transient failures such as 429, 503, or timeouts.",
    )
    parser.add_argument(
        "--api-retry-backoff",
        type=float,
        default=_default_api_retry_backoff(),
        help="Initial seconds to wait between provider/API retries; doubles per attempt up to 30s.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first category failure instead of continuing with the rest.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print configured model choices and exit.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt when --model or --batch-size is omitted.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    models = get_configured_models()

    if args.list_models:
        _print_configured_models(models)
        return 0

    interactive = not args.non_interactive and sys.stdin.isatty()

    if args.model:
        try:
            selected_model = resolve_model_choice(args.model)
        except ValueError as exc:
            parser.error(str(exc))
    elif interactive:
        selected_model = _prompt_model(models)
    else:
        selected_model = resolve_model_choice()

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = _prompt_batch_size(_default_n()) if interactive else _default_n()
    if batch_size <= 0:
        parser.error("--batch-size must be greater than 0")

    print(f"Using model: {selected_model.id}")
    print(f"Batch size per category: {batch_size}")
    print(f"Categories: {', '.join(args.categories)}")

    try:
        results = run_all_and_save(
            n=batch_size,
            categories=args.categories,
            temperature=args.temperature,
            approach=args.approach,
            provider=selected_model.provider,
            model=selected_model.model,
            output_dir=args.output_dir,
            retries=args.retries,
            max_tokens=args.max_output_tokens,
            api_retries=args.api_retries,
            api_retry_backoff=args.api_retry_backoff,
            continue_on_error=not args.fail_fast,
        )
    except Exception as exc:
        print(f"Generation failed: {_redact_secrets(exc)}", file=sys.stderr)
        return 1

    return 1 if results.failures else 0


__all__ = [
    "ANSWER_COLUMNS",
    "APPROACH",
    "CATEGORY_SPECS",
    "CategoryFailure",
    "ConfiguredModel",
    "CategorySpec",
    "DEFAULT_PROVIDER",
    "GenerationBatchResult",
    "MPS",
    "MINISTRIES",
    "PROVIDER",
    "ProviderRequestError",
    "QUESTION_COLUMNS",
    "SUPPORTED_PROVIDERS",
    "RuntimeConfig",
    "active_config",
    "build_arg_parser",
    "build_messages",
    "call_llm_chat_json",
    "compare_approaches",
    "evaluate_generation",
    "generate_set",
    "get_configured_model_ids",
    "get_configured_models",
    "main",
    "resolve_model_choice",
    "resolve_provider_and_model",
    "run_all_and_save",
    "run_and_save",
    "set_default_n",
    "set_model",
    "set_provider",
    "validate_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
