from __future__ import annotations

import pandas as pd

from nodes.contracts import normalize_node_parameters
from nodes.templates.frame_support import resolve_column_labels


def value_counts_reporter(
    df: pd.DataFrame,
    columns=None,
    dropna: bool = False,
    sort: bool = True,
    include_header: bool = True,
    **kwargs,
) -> str:
    params = {
        "columns": columns,
        "dropna": dropna,
        "sort": sort,
        "include_header": include_header,
    }
    params.update(kwargs)
    normalized = normalize_node_parameters("ValueCountsReporter", params)

    selected_columns = resolve_column_labels(df, normalized.get("columns")) or list(df.columns)
    report = pd.concat(
        [
            df[column].value_counts(dropna=bool(normalized.get("dropna", False))).rename(column)
            for column in selected_columns
        ],
        axis=1,
        sort=False,
    ).fillna(0)

    try:
        report = report.astype(int)
    except Exception:
        pass

    if normalized.get("sort", True):
        try:
            report = report.sort_index(kind="stable")
        except TypeError:
            report = report.reindex(
                sorted(report.index, key=lambda value: str(value)),
            )

    result = report.to_string(header=bool(normalized.get("include_header", True)))
    print("[ValueCountsReporter] Produced value-count text report")
    return result
