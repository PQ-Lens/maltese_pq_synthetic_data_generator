"""Runnable entrypoint for the Maltese PQ synthetic data generator.

This file is intentionally small: the maintained generator implementation lives
in `maltese_pq_synthetic_generator.py`. Keeping this as a launcher means VPS
runs can call one stable filename while the real logic stays in one place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_TOTAL_ROWS = 100


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

    results: Dict[str, Tuple[Any, Any]] = {}
    failures: List[Any] = []
    for category_id in categories:
        row_count = distribution[category_id]
        print(f"\nGenerating {category_id} ({generator.CATEGORY_SPECS[category_id].name})")
        print(f"Rows for {category_id}: {row_count}")
        try:
            results[category_id] = generator.run_and_save(
                category_id=category_id,
                n=row_count,
                temperature=args.temperature,
                approach=args.approach,
                provider=selected_model.provider,
                model=selected_model.model,
                output_dir=args.output_dir,
                retries=args.retries,
                max_tokens=args.max_output_tokens,
                api_retries=args.api_retries,
                api_retry_backoff=args.api_retry_backoff,
            )
        except Exception as exc:
            error = generator._redact_secrets(exc)
            failures.append(
                generator.CategoryFailure(
                    category_id=category_id,
                    category_name=generator.CATEGORY_SPECS[category_id].name,
                    error=error,
                )
            )
            metrics_path = generator._write_failure_metrics(
                category_id=category_id,
                n=row_count,
                temperature=args.temperature,
                approach=args.approach,
                provider=selected_model.provider,
                model=selected_model.model,
                output_dir=args.output_dir,
                error=error,
            )
            print(f"Failed {category_id}: {error}", file=sys.stderr)
            print(f"Saved failure metrics: {metrics_path}", file=sys.stderr)
            if args.fail_fast:
                break

    aggregate_path = generator._write_aggregate_csv(
        results=results,
        output_dir=args.output_dir,
        approach=args.approach,
        provider=selected_model.provider,
        model=selected_model.model,
    )
    if aggregate_path:
        print(f"\nSaved combined data: {aggregate_path}")

    if failures:
        failed_ids = ", ".join(failure.category_id for failure in failures)
        print(f"\nCompleted with failed categories: {failed_ids}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
