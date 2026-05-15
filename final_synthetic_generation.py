"""Final runnable entrypoint for one-file Maltese PQ synthetic data output.

The maintained generation engine lives in `maltese_pq_synthetic_generator.py`.
This script controls the final run shape: 100 total rows by default, spread
across the selected categories, and saved as one combined CSV.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_TOTAL_ROWS = 100
FINAL_COLUMNS = [
    "DATE",
    "PQ NO.",
    "MP",
    "Title",
    "Question",
    "Question Type",
    "Question Categorization accuracy",
    "Answer",
    "Answer Type",
    "Answer Categorization accuracy",
]


def _restart_inside_local_venv() -> None:
    """Use the project venv automatically when the script is run with python3."""

    if sys.prefix != sys.base_prefix:
        return

    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    os.execv(str(venv_python), [str(venv_python), *sys.argv])


def _row_distribution(total_rows: int, categories: Sequence[str]) -> Dict[str, int]:
    base, remainder = divmod(total_rows, len(categories))
    return {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(categories)
    }


def _select_model(generator: Any, args: Any, parser: Any) -> Any:
    if args.model:
        try:
            return generator.resolve_model_choice(args.model)
        except ValueError as exc:
            parser.error(str(exc))

    interactive = not args.non_interactive and sys.stdin.isatty()
    if not interactive:
        return generator.resolve_model_choice()

    models = generator.get_configured_models()
    print("Configured models:")
    for index, model in enumerate(models, start=1):
        print(f"  {index}. {model.id} [{model.source}]")

    while True:
        raw = input(f"Select model [1-{len(models)}] (default 1): ").strip()
        if raw == "":
            return models[0]
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        try:
            return generator.resolve_model_choice(raw)
        except ValueError as exc:
            print(exc)


def _build_parser(generator: Any) -> Any:
    parser = generator.build_arg_parser()
    for action in parser._actions:
        if action.dest == "batch_size":
            action.metavar = "TOTAL_ROWS"
            action.help = (
                "Total rows to generate across all selected categories. "
                f"Defaults to {DEFAULT_TOTAL_ROWS}."
            )
    return parser


def _slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-").lower()


def _final_rows(generator: Any, category_id: str, df: Any) -> List[Dict[str, str]]:
    spec = generator.CATEGORY_SPECS[category_id]
    is_question_category = spec.kind == "question"
    question_type = spec.name if is_question_category else ""
    answer_type = spec.name if not is_question_category else ""

    rows: List[Dict[str, str]] = []
    for record in df.to_dict(orient="records"):
        rows.append(
            {
                "DATE": record.get("Date", ""),
                "PQ NO.": record.get("PQ No.", ""),
                "MP": record.get("MP", ""),
                "Title": record.get("Title (EN)", ""),
                "Question": record.get("Question (EN)", ""),
                "Question Type": question_type,
                "Question Categorization accuracy": "",
                "Answer": record.get("Answer (EN)", ""),
                "Answer Type": answer_type,
                "Answer Categorization accuracy": "",
            }
        )
    return rows


def _write_final_csv(
    generator: Any,
    rows: Sequence[Dict[str, str]],
    *,
    output_dir: str,
    approach: str,
    provider: str,
    model: str,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_slug = _slugify(model.replace(":", "-"))
    path = out_dir / f"final_synthetic_pq_{approach}_{provider}_{model_slug}_{stamp}.csv"
    generator.pd.DataFrame(rows, columns=FINAL_COLUMNS).to_csv(path, index=False)
    return path


def main() -> int:
    _restart_inside_local_venv()

    try:
        import maltese_pq_synthetic_generator as generator
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(
            f"Missing dependency: {missing}. Install requirements with "
            "`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`, "
            "then rerun this script.",
            file=sys.stderr,
        )
        return 1

    parser = _build_parser(generator)
    args = parser.parse_args()

    if args.list_models:
        print("Configured models:")
        for index, model in enumerate(generator.get_configured_models(), start=1):
            print(f"  {index}. {model.id} [{model.source}]")
        return 0

    selected_model = _select_model(generator, args, parser)
    total_rows = args.batch_size if args.batch_size is not None else DEFAULT_TOTAL_ROWS
    if total_rows <= 0:
        parser.error("--batch-size must be greater than 0")

    categories: List[str] = list(args.categories)
    distribution = _row_distribution(total_rows, categories)
    distribution_text = ", ".join(
        f"{category}={count}" for category, count in distribution.items()
    )

    print(f"Using model: {selected_model.id}")
    print(f"Total rows: {total_rows}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Row distribution: {distribution_text}")

    rows: List[Dict[str, str]] = []
    failures: List[Any] = []
    for category_id in categories:
        row_count = distribution[category_id]
        print(f"\nGenerating {category_id} ({generator.CATEGORY_SPECS[category_id].name})")
        print(f"Rows for {category_id}: {row_count}")
        try:
            df = generator.generate_set(
                category_id=category_id,
                n=row_count,
                temperature=args.temperature,
                approach=args.approach,
                provider=selected_model.provider,
                model=selected_model.model,
                retries=args.retries,
                max_tokens=args.max_output_tokens,
                api_retries=args.api_retries,
                api_retry_backoff=args.api_retry_backoff,
            )
            rows.extend(_final_rows(generator, category_id, df))
        except Exception as exc:
            error = generator._redact_secrets(exc)
            failures.append(
                generator.CategoryFailure(
                    category_id=category_id,
                    category_name=generator.CATEGORY_SPECS[category_id].name,
                    error=error,
                )
            )
            print(f"Failed {category_id}: {error}", file=sys.stderr)
            if args.fail_fast:
                break

    output_path = _write_final_csv(
        generator,
        rows,
        output_dir=args.output_dir,
        approach=args.approach,
        provider=selected_model.provider,
        model=selected_model.model,
    )
    print(f"\nSaved final data: {output_path}")
    print(f"Final row count: {len(rows)}")

    if failures:
        failed_ids = ", ".join(failure.category_id for failure in failures)
        print(f"\nCompleted with failed categories: {failed_ids}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
