"""Synthetic Maltese parliamentary question generation utilities.

This module extracts the notebook logic into a reusable Python API and adds
provider abstraction for both Ollama and Gemini backends.
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
APPROACH = Literal["zero_shot", "one_shot"]
PROVIDER = Literal["ollama", "gemini"]


@dataclass
class RuntimeConfig:
    """Runtime defaults sourced from environment variables when available."""

    provider: PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    model_name: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    default_n: int = int(os.getenv("PQ_DEFAULT_N", "2"))
    default_temperature: float = float(os.getenv("PQ_DEFAULT_TEMPERATURE", "0.7"))
    default_approach: APPROACH = os.getenv("PQ_DEFAULT_APPROACH", "zero_shot")
    output_dir: str = os.getenv("PQ_OUTPUT_DIR", "pq_synthetic_outputs")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    gemini_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    max_output_tokens: int = int(os.getenv("PQ_MAX_OUTPUT_TOKENS", "2500"))


CONFIG = RuntimeConfig()
os.makedirs(CONFIG.output_dir, exist_ok=True)


def set_model(model_name: str) -> None:
    """Override the active model name for subsequent generations."""

    CONFIG.model_name = model_name
    print(f"Model set to: {CONFIG.model_name}")


def set_provider(provider: PROVIDER) -> None:
    """Switch the active LLM provider after validating the value."""

    if provider not in ("ollama", "gemini"):
        raise ValueError("provider must be 'ollama' or 'gemini'")
    CONFIG.provider = provider
    print(f"Provider set to: {CONFIG.provider}")


def set_default_n(n: int) -> None:
    """Update the default row count used by helper functions."""

    if n <= 0:
        raise ValueError("n must be > 0")
    CONFIG.default_n = n
    print(f"default_n set to: {CONFIG.default_n}")


def active_config() -> Dict[str, Any]:
    """Return the effective runtime configuration with secrets masked."""

    cfg = asdict(CONFIG).copy()
    if cfg.get("gemini_api_key"):
        cfg["gemini_api_key"] = "***set***"
    return cfg


@dataclass(frozen=True)
class CategorySpec:
    """Metadata describing each output category and its required schema."""

    id: str
    name: str
    kind: str
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
        "A1",
        "Information-Seeking Questions",
        "question",
        QUESTION_COLUMNS,
        "Generate neutral, specific, information-seeking parliamentary questions.",
    ),
    "A2": CategorySpec(
        "A2",
        "Assertion / Hidden-Accusation Questions",
        "question",
        QUESTION_COLUMNS,
        "Generate challenge-style questions implying criticism/accountability concerns.",
    ),
    "A3": CategorySpec(
        "A3",
        "Request / Directive Questions",
        "question",
        QUESTION_COLUMNS,
        "Generate directive questions asking for tabling documents/lists/breakdowns/timeframes.",
    ),
    "B1": CategorySpec(
        "B1",
        "Replies (Direct Answers)",
        "answer",
        ANSWER_COLUMNS,
        "Generate direct answers that provide clear requested information.",
    ),
    "B2": CategorySpec(
        "B2",
        "Answers by Implication (Indirect/Partial)",
        "answer",
        ANSWER_COLUMNS,
        "Generate partial/indirect answers with conditions or vague timelines.",
    ),
    "B3": CategorySpec(
        "B3",
        "Non-Replies (Deferral/Referral/Reroute)",
        "answer",
        ANSWER_COLUMNS,
        "Generate non-replies: deferrals/referrals/reroutes without substance.",
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

SYSTEM_STYLE = (
    "You generate synthetic Maltese parliamentary question (PQ) records in English. "
    "Fabricated but plausible. Formal parliamentary tone. JSON output only."
)

ONE_SHOT_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "A1": {
        "Date": "23/04/2026",
        "PQ No.": "23456",
        "MP": "Chris Said",
        "Ministry (EN)": "Health and Active Ageing",
        "Title (EN)": "Community Clinics – Staffing Levels",
        "Question (EN)": "Can the Minister state the current number of doctors assigned to each community clinic by locality?",
        "Answer (EN)": "The requested breakdown is currently being compiled and will be tabled.",
    },
    "A2": {
        "Date": "23/04/2026",
        "PQ No.": "23457",
        "MP": "Jerome Caruana Cilia",
        "Ministry (EN)": "Transport, Infrastructure and Public Works",
        "Title (EN)": "Road Works – Delays",
        "Question (EN)": "Given repeated delays, can the Minister explain why completion deadlines were missed for the scheduled arterial road works?",
        "Answer (EN)": "Works are ongoing and revised timelines are being coordinated with contractors.",
    },
    "A3": {
        "Date": "23/04/2026",
        "PQ No.": "23458",
        "MP": "Rebekah Borg",
        "Ministry (EN)": "Economy, European Funds and Lands",
        "Title (EN)": "Public Contracts – Publication",
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


def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    """Flatten chat messages for providers that accept a single text prompt."""

    return "\n\n".join(
        f"{(message.get('role') or 'user').upper()}:\n{message.get('content', '')}"
        for message in messages
    )


def _extract_json_from_text(text: str) -> Any:
    """Parse JSON even when the model wraps it in fences or extra text."""

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
    """Normalize common LLM JSON shapes to the list-of-rows form expected downstream."""

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        # Many models wrap the actual array in a top-level object.
        for key in ("rows", "data", "items", "results", "records", "output"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        # If the model returned a single record object, treat it as a one-row batch.
        if payload:
            return [payload]

    raise ValueError(f"Could not normalize model output to a row list. Got: {type(payload).__name__}")


def build_messages(spec: CategorySpec, n: int, approach: APPROACH) -> List[Dict[str, str]]:
    """Build a prompt that enforces category, schema, and style constraints."""

    base = (
        f"Create {n} synthetic Maltese PQ-style records for category {spec.id}: {spec.name}.\n"
        f"Category instruction: {spec.instruction}\n\n"
        f"Output must be a JSON array with exactly {n} objects.\n"
        f"Each object must have exactly these keys: {spec.columns}\n"
        "Date format: DD/MM/YYYY. PQ No. numeric.\n"
        f"MP must be one of: {MPS}\n"
        f"If Ministry (EN) exists, it must be one of: {MINISTRIES}\n"
        "Formal parliamentary English. Fictional only. No markdown."
    )

    messages = [{"role": "system", "content": SYSTEM_STYLE}]

    if approach == "one_shot":
        example = ONE_SHOT_EXAMPLES[spec.id]
        messages.append(
            {
                "role": "user",
                "content": (
                    "Here is one valid example row (format/style reference only):\n"
                    f"{json.dumps(example, ensure_ascii=False)}\n\n"
                    "Generate new fictional records, do not copy this example."
                ),
            }
        )

    messages.append({"role": "user", "content": base})
    return messages


def call_gemini_chat_json(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    model_name: str,
) -> Any:
    """Call Gemini and return parsed JSON rows."""

    if not CONFIG.gemini_api_key:
        raise ValueError("GOOGLE_API_KEY is required for provider='gemini'")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": _messages_to_text(messages)}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    response = requests.post(
        f"{url}?key={CONFIG.gemini_api_key}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError(f"Gemini returned no candidates: {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    parsed = _extract_json_from_text(text)
    return _normalize_rows_payload(parsed)


def call_ollama_chat_json(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    model_name: str,
) -> Any:
    """Call Ollama's chat API with JSON mode enabled."""

    url = f"{CONFIG.ollama_base_url}/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    text = (data.get("message") or {}).get("content", "").strip()
    if not text:
        raise ValueError(f"Ollama returned empty content: {data}")

    parsed = _extract_json_from_text(text)
    return _normalize_rows_payload(parsed)


def call_llm_chat_json(
    messages: List[Dict[str, str]],
    provider: PROVIDER,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    """Dispatch chat completion requests to the selected backend."""

    if provider == "ollama":
        return call_ollama_chat_json(messages, temperature, max_tokens, model_name)
    if provider == "gemini":
        return call_gemini_chat_json(messages, temperature, max_tokens, model_name)
    raise ValueError(f"Unsupported provider: {provider}")


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
    provider: Optional[PROVIDER] = None,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """Generate a validated dataframe for the requested category."""

    if category_id not in CATEGORY_SPECS:
        raise KeyError(f"Unknown category_id: {category_id}. Use {list(CATEGORY_SPECS.keys())}")

    n = n if n is not None else CONFIG.default_n
    temperature = temperature if temperature is not None else CONFIG.default_temperature
    approach = approach if approach is not None else CONFIG.default_approach
    provider = provider if provider is not None else CONFIG.provider
    model_name = model_name if model_name is not None else CONFIG.model_name

    spec = CATEGORY_SPECS[category_id]
    messages = build_messages(spec, n=n, approach=approach)

    last_errors: List[str] = []
    for _ in range(retries + 1):
        rows = call_llm_chat_json(
            messages=messages,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=CONFIG.max_output_tokens,
        )
        errors = validate_rows(rows, spec.columns, n=n)
        if not errors:
            return pd.DataFrame(rows, columns=spec.columns)

        # Feed the validation failures back into the model for a constrained retry.
        last_errors = errors
        messages.append(
            {
                "role": "user",
                "content": "Fix validation errors and return valid JSON only:\n- "
                + "\n- ".join(errors),
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
