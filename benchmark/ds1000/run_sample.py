from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.ds1000.loader import load_ds1000_tasks
from benchmark.ds1000.runner import run_ds1000_task
from benchmark.ds1000.subset import apply_linear_pandas_manifest, load_linear_pandas_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sample of DS-1000 Pandas tasks.")
    parser.add_argument(
        "--dataset",
        default="benchmark/ds1000/upstream/ds1000.jsonl.gz",
        help="Path to the DS-1000 dataset file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of filtered tasks to run.",
    )
    parser.add_argument(
        "--library",
        default="Pandas",
        help="Library label to load from DS-1000.",
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Skip records that fail normalization while loading.",
    )
    parser.add_argument(
        "--output",
        default="benchmark/results/ds1000_sample_results.json",
        help="Path to write the run summary JSON.",
    )
    parser.add_argument(
        "--manifest",
        default="benchmark/ds1000/linear_pandas_subset_manifest.json",
        help="Path to a frozen subset manifest file.",
    )
    args = parser.parse_args()

    tasks = load_ds1000_tasks(
        args.dataset,
        library=args.library,
        skip_failures=args.skip_failures,
    )
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Subset manifest not found: {manifest_path}. Generate it before running the sample."
        )
    manifest = load_linear_pandas_manifest(manifest_path)
    selected, rejected = apply_linear_pandas_manifest(tasks, manifest, include_rejections=True)
    sample = selected[: args.limit]

    print(f"Loaded {len(tasks)} {args.library} tasks")
    print(f"Filtered subset size: {len(selected)}")
    print(f"Rejected during subset filter: {len(rejected)}")
    print(f"Running {len(sample)} task(s)")

    results = []
    pass_count = 0

    for index, task in enumerate(sample, start=1):
        print(f"\n[{index}/{len(sample)}] {task.task_id}")
        outcome = run_ds1000_task(task)
        status = "PASS" if outcome.comparison_passed else "FAIL"
        stage = "comparison"
        if not outcome.plan_success:
            stage = "planner"
        elif not outcome.validation_success:
            stage = "validator"
        elif not outcome.compile_success:
            stage = "compiler"
        elif not outcome.run_success:
            stage = "runtime"
        elif not outcome.comparison_passed:
            stage = "comparator"

        print(f"  {status} at {stage}")
        if outcome.validation_errors:
            print(f"  validation errors: {len(outcome.validation_errors)}")
        if outcome.compile_error:
            print("  compile error captured")
        if outcome.runtime_error:
            print("  runtime error captured")
        if outcome.comparison and outcome.comparison.failures:
            print(f"  comparison failures: {len(outcome.comparison.failures)}")

        if outcome.comparison_passed:
            pass_count += 1

        results.append(
            {
                "task_id": outcome.task_id,
                "plan_success": outcome.plan_success,
                "validation_success": outcome.validation_success,
                "compile_success": outcome.compile_success,
                "run_success": outcome.run_success,
                "comparison_passed": outcome.comparison_passed,
                "validation_errors": outcome.validation_errors,
                "compile_error": outcome.compile_error,
                "runtime_error": outcome.runtime_error,
                "comparison_failures": outcome.comparison.failures if outcome.comparison else [],
                "plan": outcome.plan,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": str(args.dataset),
        "library": args.library,
        "loaded_tasks": len(tasks),
        "filtered_tasks": len(selected),
        "requested_limit": args.limit,
        "run_count": len(sample),
        "pass_count": pass_count,
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nSummary")
    print(f"  Passed: {pass_count}/{len(sample)}")
    print(f"  Results written to: {output_path}")


if __name__ == "__main__":
    main()
