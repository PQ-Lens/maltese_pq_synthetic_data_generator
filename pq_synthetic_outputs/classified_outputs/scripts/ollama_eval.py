"""Evaluate final synthetic PQ CSV category assignments with local Ollama models.

This mirrors the Mistral evaluator's input schema, prompt shape, row-level
details, and summary metrics while calling a local Ollama chat endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_BATCH_SIZE = 5

DATE_COLUMN = "DATE"
PQ_NO_COLUMN = "PQ NO."
MP_COLUMN = "MP"
TITLE_COLUMN = "Title"
QUESTION_COLUMN = "Question"
QUESTION_TYPE_COLUMN = "Question Type"
QUESTION_ACCURACY_COLUMN = "Question Categorization accuracy"
ANSWER_COLUMN = "Answer"
ANSWER_TYPE_COLUMN = "Answer Type"
ANSWER_ACCURACY_COLUMN = "Answer Categorization accuracy"

REQUIRED_COLUMNS = [
    DATE_COLUMN,
    PQ_NO_COLUMN,
    MP_COLUMN,
    TITLE_COLUMN,
    QUESTION_COLUMN,
    QUESTION_TYPE_COLUMN,
    QUESTION_ACCURACY_COLUMN,
    ANSWER_COLUMN,
    ANSWER_TYPE_COLUMN,
    ANSWER_ACCURACY_COLUMN,
]

CATEGORY_SPECS: Dict[str, Dict[str, str]] = {
    "A1": {
        "name": "Information-Seeking Questions",
        "kind": "question",
        "instruction": (
            "Generate information-seeking parliamentary questions. The MP is primarily "
            "asking for missing facts such as counts, dates, locations, or status. "
            "Keep it neutral and specific."
        ),
    },
    "A2": {
        "name": "Assertion / Hidden-Accusation Questions",
        "kind": "question",
        "instruction": (
            "Generate assertion or challenge parliamentary questions with a hidden "
            "accusation or accountability push. The wording should imply criticism "
            "while still being phrased as a question."
        ),
    },
    "A3": {
        "name": "Request / Directive Questions",
        "kind": "question",
        "instruction": (
            "Generate directive parliamentary questions that ask the Minister or PM "
            "to table documents, provide lists, provide a timeframe, confirm a "
            "contract, publish a breakdown, or take a similar action."
        ),
    },
    "B1": {
        "name": "Replies (Direct Answers)",
        "kind": "answer",
        "instruction": (
            "Generate answers that directly address the question with the requested "
            "facts or a clear yes/no plus details. Avoid deferrals and referrals."
        ),
    },
    "B2": {
        "name": "Answers by Implication (Indirect/Partial)",
        "kind": "answer",
        "instruction": (
            "Generate answers that are partial or indirect: give an update, a "
            "condition, or a vague timeline instead of the exact requested item."
        ),
    },
    "B3": {
        "name": "Non-Replies (Deferral/Referral/Reroute)",
        "kind": "answer",
        "instruction": (
            "Generate non-replies: defer to a future sitting, refer to another PQ, "
            "or reroute to another minister. Do not provide the substance of the "
            "requested information."
        ),
    },
}

QUESTION_CATEGORY_IDS = tuple(
    category_id
    for category_id, spec in CATEGORY_SPECS.items()
    if spec["kind"] == "question"
)
ANSWER_CATEGORY_IDS = tuple(
    category_id for category_id, spec in CATEGORY_SPECS.items() if spec["kind"] == "answer"
)
CATEGORY_NAME_TO_ID = {spec["name"].lower(): category_id for category_id, spec in CATEGORY_SPECS.items()}


@dataclass(frozen=True)
class EvaluationRecord:
    row_index: int
    target: str
    assigned_category_id: str
    assigned_category_name: str
    question: str
    answer: str
    title: str


@dataclass(frozen=True)
class EvaluationResult:
    row_index: int
    target: str
    assigned_category_id: str
    assigned_category_name: str
    predicted_category_id: str
    predicted_category_name: str
    accurate: str
    confidence: str
    rationale: str


class OllamaEvaluationError(RuntimeError):
    """Raised when Ollama evaluation cannot complete."""


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_category(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return ""

    upper = text.upper()
    match = re.search(r"\b[AB][123]\b", upper)
    if match:
        return match.group(0)

    return CATEGORY_NAME_TO_ID.get(text.lower(), text)


def _category_name(category_id: str) -> str:
    return CATEGORY_SPECS.get(category_id, {}).get("name", "")


def _normalize_yes_no(value: Any) -> str:
    text = _safe_text(value).lower()
    if text in {"yes", "y", "true", "correct", "accurate"}:
        return "yes"
    if text in {"no", "n", "false", "incorrect", "inaccurate"}:
        return "no"
    return ""


def _chunks(items: Sequence[EvaluationRecord], size: int) -> Iterable[List[EvaluationRecord]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])

    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object from Ollama, got {type(value).__name__}")
    return value


def _retry_delay(retry_backoff: float, attempt_index: int) -> float:
    if retry_backoff <= 0:
        return 0.0
    return min(retry_backoff * (2**attempt_index), 30.0)


def _request_ollama_json(
    messages: Sequence[Mapping[str, str]],
    *,
    host: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_retries: int,
    retry_backoff: float,
) -> Dict[str, Any]:
    endpoint = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": list(messages),
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 8192,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    attempts = max(1, api_retries + 1)

    for attempt_index in range(attempts):
        try:
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = (data.get("message") or {}).get("content", "")
            if not content:
                raise OllamaEvaluationError(f"Ollama returned no message content: {data}")
            return _extract_json_object(str(content))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            if attempt_index < attempts - 1:
                delay = _retry_delay(retry_backoff, attempt_index)
                print(
                    f"Ollama request/parse error {exc.__class__.__name__}; retrying in "
                    f"{delay:.1f}s ({attempt_index + 2}/{attempts})..."
                )
                if delay:
                    time.sleep(delay)
                continue
            raise OllamaEvaluationError(
                f"Ollama request failed after {attempts} attempts: {exc}"
            ) from exc

    raise OllamaEvaluationError(f"Ollama request failed after {attempts} attempts.")


def _category_definitions(category_ids: Sequence[str]) -> str:
    return "\n".join(
        f"- {category_id}: {CATEGORY_SPECS[category_id]['name']}. "
        f"{CATEGORY_SPECS[category_id]['instruction']}"
        for category_id in category_ids
    )


def _build_batch_messages(records: Sequence[EvaluationRecord]) -> List[Dict[str, str]]:
    if not records:
        raise ValueError("Cannot build evaluation prompt for an empty record batch.")

    target = records[0].target
    allowed_ids = QUESTION_CATEGORY_IDS if target == "question" else ANSWER_CATEGORY_IDS
    review_focus = (
        "Use the Question field as the primary text. The Answer field is context only."
        if target == "question"
        else "Use the Answer field as the primary text, judged in relation to the Question field."
    )
    payload = [
        {
            "row_index": record.row_index,
            "assigned_category": record.assigned_category_id,
            "assigned_category_name": record.assigned_category_name,
            "title": record.title,
            "question": record.question,
            "answer": record.answer,
        }
        for record in records
    ]

    return [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator of Maltese Parliamentary Question discourse categories. "
                "For each row, choose the single best category from the allowed set and mark whether "
                "the assigned category is accurate. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Evaluate these {target} category assignments.\n\n"
                f"Allowed {target} categories:\n{_category_definitions(allowed_ids)}\n\n"
                f"{review_focus}\n\n"
                f"Rows to evaluate:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                "Return exactly this JSON object shape:\n"
                "{"
                "\"evaluations\":["
                "{"
                "\"row_index\":123,"
                "\"predicted_category\":\"A1/A2/A3 or B1/B2/B3\","
                "\"accurate\":\"yes or no\","
                "\"confidence\":\"low, medium, or high\","
                "\"rationale\":\"one concise sentence\""
                "}"
                "]"
                "}\n"
                "Set accurate to yes only when assigned_category equals the best category."
            ),
        },
    ]


def _parse_batch_results(
    data: Dict[str, Any],
    records: Sequence[EvaluationRecord],
) -> List[EvaluationResult]:
    raw_evaluations = data.get("evaluations")
    if isinstance(raw_evaluations, dict):
        raw_evaluations = [raw_evaluations]
    if not isinstance(raw_evaluations, list):
        raise OllamaEvaluationError("Ollama response did not contain an evaluations list.")

    by_index = {record.row_index: record for record in records}
    target = records[0].target if records else ""
    allowed = QUESTION_CATEGORY_IDS if target == "question" else ANSWER_CATEGORY_IDS
    results: List[EvaluationResult] = []

    for item in raw_evaluations:
        if not isinstance(item, dict):
            raise OllamaEvaluationError("Ollama returned a non-object evaluation item.")

        row_index = int(item.get("row_index"))
        if row_index not in by_index:
            raise OllamaEvaluationError(f"Ollama returned unknown row_index {row_index}.")

        record = by_index[row_index]
        predicted = _normalize_category(
            item.get("predicted_category")
            or item.get("predicted_category_id")
            or item.get("category")
        )
        if predicted not in allowed:
            raise OllamaEvaluationError(
                f"Ollama predicted invalid {target} category {predicted!r} for row {row_index}."
            )

        accurate = _normalize_yes_no(item.get("accurate"))
        expected = "yes" if predicted == record.assigned_category_id else "no"
        if accurate not in {"yes", "no"}:
            accurate = expected
        if accurate != expected:
            accurate = expected

        confidence = _safe_text(item.get("confidence")).lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = ""

        results.append(
            EvaluationResult(
                row_index=row_index,
                target=target,
                assigned_category_id=record.assigned_category_id,
                assigned_category_name=record.assigned_category_name,
                predicted_category_id=predicted,
                predicted_category_name=_category_name(predicted),
                accurate=accurate,
                confidence=confidence,
                rationale=_safe_text(item.get("rationale")),
            )
        )

    returned = {result.row_index for result in results}
    missing = sorted(set(by_index) - returned)
    if missing:
        raise OllamaEvaluationError(f"Ollama omitted row_index values: {missing}")

    return sorted(results, key=lambda result: result.row_index)


def _evaluate_batch(
    records: Sequence[EvaluationRecord],
    *,
    host: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_retries: int,
    retry_backoff: float,
) -> List[EvaluationResult]:
    data = _request_ollama_json(
        _build_batch_messages(records),
        host=host,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_retries=api_retries,
        retry_backoff=retry_backoff,
    )
    return _parse_batch_results(data, records)


def validate_input_columns(fieldnames: Optional[Sequence[str]]) -> None:
    if fieldnames is None:
        raise ValueError("Input CSV has no header row.")
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def collect_records(rows: Sequence[Mapping[str, str]]) -> List[EvaluationRecord]:
    records: List[EvaluationRecord] = []
    invalid_rows: List[str] = []

    for index, row in enumerate(rows):
        question_type = _safe_text(row.get(QUESTION_TYPE_COLUMN))
        answer_type = _safe_text(row.get(ANSWER_TYPE_COLUMN))
        has_question_type = bool(question_type)
        has_answer_type = bool(answer_type)

        if has_question_type == has_answer_type:
            invalid_rows.append(
                f"row {index}: expected exactly one of Question Type or Answer Type to be filled"
            )
            continue

        target = "question" if has_question_type else "answer"
        assigned_name = question_type if target == "question" else answer_type
        assigned_id = _normalize_category(assigned_name)
        allowed = QUESTION_CATEGORY_IDS if target == "question" else ANSWER_CATEGORY_IDS
        if assigned_id not in allowed:
            invalid_rows.append(
                f"row {index}: {target} category {assigned_name!r} could not be mapped to {allowed}"
            )
            continue

        records.append(
            EvaluationRecord(
                row_index=index,
                target=target,
                assigned_category_id=assigned_id,
                assigned_category_name=_category_name(assigned_id),
                question=_safe_text(row.get(QUESTION_COLUMN)),
                answer=_safe_text(row.get(ANSWER_COLUMN)),
                title=_safe_text(row.get(TITLE_COLUMN)),
            )
        )

    if invalid_rows:
        preview = "\n- ".join(invalid_rows[:10])
        extra = "" if len(invalid_rows) <= 10 else f"\n... and {len(invalid_rows) - 10} more"
        raise ValueError(f"Input CSV contains invalid row typing:\n- {preview}{extra}")

    return records


def evaluate_records(
    records: Sequence[EvaluationRecord],
    *,
    host: str,
    model: str,
    temperature: float,
    max_tokens: int,
    batch_size: int,
    api_retries: int,
    retry_backoff: float,
) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    grouped = [
        ("question", [record for record in records if record.target == "question"]),
        ("answer", [record for record in records if record.target == "answer"]),
    ]

    for target, target_records in grouped:
        for batch in _chunks(target_records, max(1, batch_size)):
            print(
                f"Evaluating {target} rows {batch[0].row_index}-{batch[-1].row_index} "
                f"({len(batch)} rows)"
            )
            try:
                results.extend(
                    _evaluate_batch(
                        batch,
                        host=host,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_retries=api_retries,
                        retry_backoff=retry_backoff,
                    )
                )
            except Exception:
                if len(batch) == 1:
                    raise
                print("Batch response was invalid; retrying rows one at a time.")
                for record in batch:
                    results.extend(
                        _evaluate_batch(
                            [record],
                            host=host,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            api_retries=api_retries,
                            retry_backoff=retry_backoff,
                        )
                    )

    return sorted(results, key=lambda result: result.row_index)


def apply_results(
    rows: Sequence[Mapping[str, str]],
    results: Sequence[EvaluationResult],
) -> List[Dict[str, str]]:
    evaluated = [dict(row) for row in rows]

    for row in evaluated:
        row[QUESTION_ACCURACY_COLUMN] = ""
        row[ANSWER_ACCURACY_COLUMN] = ""

    for result in results:
        if result.target == "question":
            evaluated[result.row_index][QUESTION_ACCURACY_COLUMN] = result.accurate
        elif result.target == "answer":
            evaluated[result.row_index][ANSWER_ACCURACY_COLUMN] = result.accurate

    return evaluated


def _category_metrics(
    records: Sequence[EvaluationResult],
    labels: Sequence[str],
    *,
    target: str,
    accuracy_column: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = len(records)

    for label in labels:
        tp = sum(
            1
            for record in records
            if record.assigned_category_id == label and record.predicted_category_id == label
        )
        fp = sum(
            1
            for record in records
            if record.assigned_category_id == label and record.predicted_category_id != label
        )
        fn = sum(
            1
            for record in records
            if record.assigned_category_id != label and record.predicted_category_id == label
        )
        assigned_count = sum(1 for record in records if record.assigned_category_id == label)
        reference_count = sum(1 for record in records if record.predicted_category_id == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "target": target,
                "metric_scope": "category",
                "category_id": label,
                "category_name": _category_name(label),
                "evaluated_count": total,
                "assigned_count": assigned_count,
                "ollama_reference_count": reference_count,
                "yes_count": tp,
                "no_count": assigned_count - tp,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "accuracy": "",
                "based_on_column": accuracy_column,
            }
        )

    return rows


def build_metrics_rows(results: Sequence[EvaluationResult]) -> List[Dict[str, Any]]:
    metric_rows: List[Dict[str, Any]] = []

    for target, labels, accuracy_column in (
        ("question", QUESTION_CATEGORY_IDS, QUESTION_ACCURACY_COLUMN),
        ("answer", ANSWER_CATEGORY_IDS, ANSWER_ACCURACY_COLUMN),
    ):
        target_results = [result for result in results if result.target == target]
        if not target_results:
            continue

        category_rows = _category_metrics(
            target_results,
            labels,
            target=target,
            accuracy_column=accuracy_column,
        )
        metric_rows.extend(category_rows)

        yes_count = sum(1 for result in target_results if result.accurate == "yes")
        no_count = sum(1 for result in target_results if result.accurate == "no")
        total_tp = sum(int(row["true_positive"]) for row in category_rows)
        total_fp = sum(int(row["false_positive"]) for row in category_rows)
        total_fn = sum(int(row["false_negative"]) for row in category_rows)
        macro_precision = sum(float(row["precision"]) for row in category_rows) / len(category_rows)
        macro_recall = sum(float(row["recall"]) for row in category_rows) / len(category_rows)
        macro_f1 = sum(float(row["f1"]) for row in category_rows) / len(category_rows)
        micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
        accuracy = yes_count / len(target_results) if target_results else 0.0

        metric_rows.append(
            {
                "target": target,
                "metric_scope": "macro_average",
                "category_id": "ALL",
                "category_name": "All categories",
                "evaluated_count": len(target_results),
                "assigned_count": len(target_results),
                "ollama_reference_count": len(target_results),
                "yes_count": yes_count,
                "no_count": no_count,
                "true_positive": total_tp,
                "false_positive": total_fp,
                "false_negative": total_fn,
                "precision": round(macro_precision, 6),
                "recall": round(macro_recall, 6),
                "f1": round(macro_f1, 6),
                "accuracy": round(accuracy, 6),
                "based_on_column": accuracy_column,
            }
        )
        metric_rows.append(
            {
                "target": target,
                "metric_scope": "micro_average",
                "category_id": "ALL",
                "category_name": "All categories",
                "evaluated_count": len(target_results),
                "assigned_count": len(target_results),
                "ollama_reference_count": len(target_results),
                "yes_count": yes_count,
                "no_count": no_count,
                "true_positive": total_tp,
                "false_positive": total_fp,
                "false_negative": total_fn,
                "precision": round(micro_precision, 6),
                "recall": round(micro_recall, 6),
                "f1": round(micro_f1, 6),
                "accuracy": round(accuracy, 6),
                "based_on_column": accuracy_column,
            }
        )

    return metric_rows


def _details_rows(results: Sequence[EvaluationResult]) -> List[Dict[str, Any]]:
    return [asdict(result) for result in results]


def _resolve_input_path(csv_file: str) -> Path:
    raw = Path(csv_file)
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    return (Path(__file__).resolve().parent / raw).resolve()


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()


def _output_paths(input_path: Path, model: str, output_prefix: str = "") -> Tuple[Path, Path, Path]:
    stem = output_prefix or f"{input_path.stem}_ollama_{_model_slug(model)}"
    return (
        input_path.with_name(f"{stem}_evaluated.csv"),
        input_path.with_name(f"{stem}_evaluation_details.csv"),
        input_path.with_name(f"{stem}_evaluation_metrics.csv"),
    )


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        validate_input_columns(reader.fieldnames)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final synthetic PQ question/answer type assignments with Ollama."
    )
    parser.add_argument("csv_file", help="Input CSV path or file name in pq_synthetic_outputs.")
    parser.add_argument("--model", required=True, help="Ollama model to use, for example gemma2:2b.")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix for the evaluated/details/metrics output files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = _resolve_input_path(args.csv_file)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV file not found: {input_path}")

    fieldnames, rows = _read_csv(input_path)
    records = collect_records(rows)
    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Evaluating {len(records)} typed rows with Ollama model {args.model}")

    results = evaluate_records(
        records,
        host=args.ollama_host,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=max(1, args.batch_size),
        api_retries=max(0, args.api_retries),
        retry_backoff=max(0.0, args.retry_backoff),
    )

    evaluated = apply_results(rows, results)
    metrics = build_metrics_rows(results)
    details = _details_rows(results)

    evaluated_path, details_path, metrics_path = _output_paths(
        input_path,
        args.model,
        args.output_prefix,
    )
    _write_csv(evaluated_path, fieldnames, evaluated)
    _write_csv(details_path, [field.name for field in EvaluationResult.__dataclass_fields__.values()], details)
    _write_csv(
        metrics_path,
        [
            "target",
            "metric_scope",
            "category_id",
            "category_name",
            "evaluated_count",
            "assigned_count",
            "ollama_reference_count",
            "yes_count",
            "no_count",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "based_on_column",
        ],
        metrics,
    )

    print(f"Wrote evaluated CSV: {evaluated_path}")
    print(f"Wrote details CSV: {details_path}")
    print(f"Wrote metrics CSV: {metrics_path}")

    summary = [row for row in metrics if row["metric_scope"] == "micro_average"]
    if summary:
        print("\nMicro averages:")
        for row in summary:
            print(
                f"  {row['target']}: precision={row['precision']}, "
                f"recall={row['recall']}, f1={row['f1']}, accuracy={row['accuracy']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
