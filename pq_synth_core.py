"""Core generation utilities for Maltese PQ synthetic data (Gemini + Mistral)."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROVIDER = "gemini"
SUPPORTED_PROVIDERS = ("gemini", "mistral")

PROVIDER_API_KEY_ENV: Dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

PROVIDER_MODEL_ENV: Dict[str, str] = {
    "gemini": "GEMINI_MODEL",
    "mistral": "MISTRAL_MODEL",
}

PROVIDER_DEFAULT_MODEL: Dict[str, str] = {
    "gemini": "gemini-flash-latest",
    "mistral": "mistral-small-latest",
}

DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")

SYSTEM_STYLE = (
    "You generate synthetic Maltese parliamentary question (PQ) records in English. "
    "They must be fabricated but plausible, using realistic ministry names and parliamentary phrasing. "
    "Do NOT use real parliamentary records or real events; do NOT copy any real PQ verbatim. "
    "Keep the tone formal and consistent with parliamentary replies."
)


@dataclass(frozen=True)
class CategorySpec:
    id: str
    name: str
    kind: str  # 'question' or 'answer'
    columns: List[str]
    instruction: str


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
        id="A1",
        name="Information-Seeking Questions",
        kind="question",
        columns=QUESTION_COLUMNS,
        instruction=(
            "Generate information-seeking parliamentary questions. The MP is primarily asking for missing facts "
            "(counts, dates, locations, status). Keep it neutral and specific."
        ),
    ),
    "A2": CategorySpec(
        id="A2",
        name="Assertion / Hidden-Accusation Questions",
        kind="question",
        columns=QUESTION_COLUMNS,
        instruction=(
            "Generate assertion/challenge parliamentary questions with a hidden accusation or accountability push. "
            "The wording should imply criticism (delay, lack of transparency, disruption, poor performance) while still being phrased as a question."
        ),
    ),
    "A3": CategorySpec(
        id="A3",
        name="Request / Directive Questions",
        kind="question",
        columns=QUESTION_COLUMNS,
        instruction=(
            "Generate directive parliamentary questions that ask the Minister/PM to do something: lay documents on the Table, provide lists, "
            "provide a site plan/timeframe, confirm a contract, publish a breakdown, etc."
        ),
    ),
    "B1": CategorySpec(
        id="B1",
        name="Replies (Direct Answers)",
        kind="answer",
        columns=ANSWER_COLUMNS,
        instruction=(
            "Generate answers that directly address the question with the requested facts or a clear yes/no plus details. "
            "Avoid deferrals and avoid referrals."
        ),
    ),
    "B2": CategorySpec(
        id="B2",
        name="Answers by Implication (Indirect/Partial)",
        kind="answer",
        columns=ANSWER_COLUMNS,
        instruction=(
            "Generate answers that are partial/indirect: give an update, a condition, or a vague timeline instead of the exact requested item. "
            "Examples: 'being processed', 'in the coming months', 'depends on certification', 'will be provided in a subsequent sitting'."
        ),
    ),
    "B3": CategorySpec(
        id="B3",
        name="Non-Replies (Deferral/Referral/Reroute)",
        kind="answer",
        columns=ANSWER_COLUMNS,
        instruction=(
            "Generate non-replies: defer to a future sitting, refer to another PQ, or reroute to another minister. "
            "Do not provide the substance of the requested info."
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
    "Social Policy and Children’s Rights",
    "Health and Active Ageing",
    "Economy, European Funds and Lands",
    "Transport, Infrastructure and Public Works",
    "Gozo",
]


class LLMProvider(Protocol):
    def generate_json(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        ...


def _messages_to_gemini_prompt(messages: Sequence[Dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        role = (message.get("role") or "user").upper()
        content = message.get("content", "")
        chunks.append(f"{role}:\n{content}")
    chunks.append("Return valid JSON only. Do not include markdown fences.")
    return "\n\n".join(chunks)


def _extract_json_from_text(text: str) -> Any:
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
        snippet = cleaned[min(starts) : end + 1]
        return json.loads(snippet)


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    return ""


class GeminiProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_json(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        prompt = _messages_to_gemini_prompt(messages)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }

        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError(f"Gemini returned no candidates: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        if not text:
            raise ValueError(f"Gemini response had no text content: {data}")

        parsed = _extract_json_from_text(text)
        if isinstance(parsed, dict) and "rows" in parsed:
            return parsed["rows"]
        return parsed


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
    ) -> Any:
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        resp = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Mistral returned no choices: {data}")

        message = choices[0].get("message", {})
        text = _normalize_message_content(message.get("content")).strip()
        if not text:
            raise ValueError(f"Mistral response had no text content: {data}")

        parsed = _extract_json_from_text(text)
        if isinstance(parsed, dict) and "rows" in parsed:
            return parsed["rows"]
        return parsed


def _normalize_provider_name(provider: Optional[str]) -> str:
    name = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if name not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unknown provider '{name}'. Supported providers are: {supported}. "
            "Set LLM_PROVIDER or pass provider='gemini'/'mistral'."
        )
    return name


def _resolve_model(provider_name: str, model: Optional[str]) -> str:
    if model:
        return model

    model_env_name = PROVIDER_MODEL_ENV[provider_name]
    model_from_env = os.getenv(model_env_name)
    if model_from_env:
        return model_from_env

    return PROVIDER_DEFAULT_MODEL[provider_name]


def _get_required_api_key(provider_name: str) -> str:
    api_key_env_name = PROVIDER_API_KEY_ENV[provider_name]
    api_key = os.getenv(api_key_env_name)
    if not api_key:
        raise ValueError(
            f"Missing required environment variable '{api_key_env_name}' for provider '{provider_name}'."
        )
    return api_key


def resolve_provider_and_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str]:
    provider_name = _normalize_provider_name(provider)
    resolved_model = _resolve_model(provider_name, model)
    return provider_name, resolved_model


def _build_provider(provider_name: str) -> LLMProvider:
    api_key = _get_required_api_key(provider_name)

    if provider_name == "gemini":
        return GeminiProvider(api_key=api_key)

    if provider_name == "mistral":
        return MistralProvider(api_key=api_key)

    raise ValueError(f"Unknown provider: {provider_name}")


def call_llm_chat_json(
    messages: Sequence[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> Any:
    provider_name, resolved_model = resolve_provider_and_model(provider=provider, model=model)
    client = _build_provider(provider_name)
    return client.generate_json(
        messages=messages,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def validate_rows(rows: List[Dict[str, Any]], columns: List[str], n: int) -> List[str]:
    errors: List[str] = []
    if not isinstance(rows, list):
        return ["Output is not a JSON array."]

    if len(rows) != n:
        errors.append(f"Expected {n} rows, got {len(rows)}.")

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Row {i} is not an object.")
            continue

        missing = [column for column in columns if column not in row]
        extra = [key for key in row.keys() if key not in columns]
        if missing:
            errors.append(f"Row {i} missing columns: {missing}")
        if extra:
            errors.append(f"Row {i} has extra columns: {extra}")

        if "Date" in row and isinstance(row.get("Date"), str) and not DATE_RE.match(row["Date"]):
            errors.append(f"Row {i} has invalid Date format: {row.get('Date')}")

        if "PQ No." in row:
            pq_number = str(row.get("PQ No."))
            if not pq_number.isdigit():
                errors.append(f"Row {i} has non-numeric PQ No.: {row.get('PQ No.')}")

    return errors


def generate_set(
    category_id: str,
    n: int = 10,
    temperature: float = 0.7,
    retries: int = 1,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> pd.DataFrame:
    if category_id not in CATEGORY_SPECS:
        raise KeyError(f"Unknown category_id: {category_id}. Use one of {list(CATEGORY_SPECS.keys())}")

    spec = CATEGORY_SPECS[category_id]
    columns = spec.columns

    base_user_prompt = {
        "role": "user",
        "content": (
            f"Create {n} synthetic Maltese PQ-style records for category {spec.id}: {spec.name}.\n\n"
            f"Category instruction: {spec.instruction}\n\n"
            "Hard requirements:\n"
            "- Output MUST be valid JSON. Prefer a bare JSON array of objects.\n"
            f"- Each object MUST have exactly these keys: {columns}.\n"
            "- Date must be DD/MM/YYYY.\n"
            "- PQ No. must be numeric. Use plausible ranges (e.g., 10000–40000).\n"
            f"- MP must be one of: {MPS}.\n"
            f"- If the schema includes Ministry (EN), it must be one of: {MINISTRIES}.\n"
            "- Titles should be short and look like Maltese PQ titles (topic – detail).\n"
            "- Questions and answers must be formal parliamentary English.\n"
            "- Content must be fictional (no real cases, no real named projects beyond generic labels).\n"
            "- Do not add commentary or markdown—JSON only.\n"
        ),
    }

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_STYLE},
        base_user_prompt,
    ]

    provider_name, model_name = resolve_provider_and_model(provider=provider, model=model)

    last_errors: List[str] = []
    for _ in range(retries + 1):
        rows = call_llm_chat_json(
            messages=messages,
            provider=provider_name,
            model=model_name,
            temperature=temperature,
            max_tokens=2500,
        )
        errors = validate_rows(rows, columns, n)
        if not errors:
            return pd.DataFrame(rows, columns=columns)

        last_errors = errors
        messages.append(
            {
                "role": "user",
                "content": (
                    "The previous output failed validation. Fix it and output JSON again. "
                    "Do not change the schema. Ensure exactly the required keys and exactly N rows. "
                    "Errors were:\n- "
                    + "\n- ".join(errors)
                ),
            }
        )
        time.sleep(0.2)

    raise ValueError("Failed to generate a valid set. Validation errors:\n- " + "\n- ".join(last_errors))


def run_and_save(
    category_id: str,
    n: int = 10,
    temperature: float = 0.7,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    output_dir: str = "pq_synthetic_outputs",
) -> pd.DataFrame:
    df = generate_set(
        category_id=category_id,
        n=n,
        temperature=temperature,
        retries=1,
        provider=provider,
        model=model,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"{category_id}_{stamp}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


__all__ = [
    "ANSWER_COLUMNS",
    "CATEGORY_SPECS",
    "CategorySpec",
    "MPS",
    "MINISTRIES",
    "QUESTION_COLUMNS",
    "SUPPORTED_PROVIDERS",
    "call_llm_chat_json",
    "generate_set",
    "resolve_provider_and_model",
    "run_and_save",
    "validate_rows",
]
