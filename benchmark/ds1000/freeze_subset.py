from __future__ import annotations

import argparse

from benchmark.ds1000.loader import load_ds1000_tasks
from benchmark.ds1000.subset import write_linear_pandas_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the DS-1000 linear Pandas subset into a manifest.")
    parser.add_argument(
        "--dataset",
        default="benchmark/ds1000/upstream/ds1000.jsonl.gz",
        help="Path to the DS-1000 dataset file.",
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
        default="benchmark/ds1000/linear_pandas_subset_manifest.json",
        help="Path to write the frozen subset manifest JSON.",
    )
    args = parser.parse_args()

    tasks = load_ds1000_tasks(
        args.dataset,
        library=args.library,
        skip_failures=args.skip_failures,
    )
    output_path = write_linear_pandas_manifest(tasks, args.output)
    print(f"Wrote subset manifest: {output_path}")


if __name__ == "__main__":
    main()
