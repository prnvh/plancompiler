import pandas as pd


def _describe_result(value) -> str:
    if isinstance(value, pd.DataFrame):
        return "DataFrame"
    if isinstance(value, pd.Series):
        return "Series"
    if pd.api.types.is_scalar(value):
        return "Scalar"
    return type(value).__name__


def _columnwise_extreme_index(
    frame: pd.DataFrame,
    *,
    anchor_extreme: str,
    anchor_occurrence: str,
    target_extreme: str,
    target_occurrence: str,
    direction: str,
    skipna: bool = True,
) -> pd.Series:
    results = {}

    for column in frame.columns:
        series = frame[column]
        if skipna:
            series = series.dropna()
        if series.empty:
            results[column] = None
            continue

        anchor_value = series.min() if anchor_extreme == "min" else series.max()
        anchor_matches = series[series == anchor_value]
        anchor_index = anchor_matches.index[0] if anchor_occurrence == "first" else anchor_matches.index[-1]

        position = series.index.get_loc(anchor_index)
        if isinstance(position, slice):
            position = position.start
        if isinstance(position, list):
            position = position[0]

        if direction == "after":
            candidate = series.iloc[position:]
        else:
            candidate = series.iloc[: position + 1]

        if candidate.empty:
            results[column] = None
            continue

        target_value = candidate.min() if target_extreme == "min" else candidate.max()
        target_matches = candidate[candidate == target_value]
        results[column] = (
            target_matches.index[0]
            if target_occurrence == "first"
            else target_matches.index[-1]
        )

    return pd.Series(results)


def reduce_output(value, method: str = "identity", **options):
    """
    Reduces a DataFrame or Series to a final DataFrame, Series, or scalar.
    Supported methods: identity, column, row, iloc, loc, head, tail,
    squeeze, scalar_agg, item.
    Node: ReduceOutput
    """
    if method == "identity":
        result = value

    elif method == "column":
        if not isinstance(value, pd.DataFrame):
            raise TypeError("ReduceOutput column method requires a DataFrame input.")
        result = value[options["column"]]

    elif method == "row":
        if not isinstance(value, pd.DataFrame):
            raise TypeError("ReduceOutput row method requires a DataFrame input.")
        if "position" in options:
            result = value.iloc[options["position"]]
        elif "label" in options:
            result = value.loc[options["label"]]
        else:
            raise ValueError("ReduceOutput row method requires 'position' or 'label'.")

    elif method == "iloc":
        if isinstance(value, pd.DataFrame):
            if "row" in options and "column" in options:
                result = value.iloc[options["row"], options["column"]]
            elif "row" in options:
                result = value.iloc[options["row"]]
            elif "column" in options:
                result = value.iloc[:, options["column"]]
            else:
                raise ValueError("ReduceOutput iloc method requires 'row' or 'column'.")
        elif isinstance(value, pd.Series):
            result = value.iloc[options["position"]]
        else:
            raise TypeError("ReduceOutput iloc method requires a DataFrame or Series input.")

    elif method == "loc":
        if isinstance(value, pd.DataFrame):
            if "row" in options and "column" in options:
                result = value.loc[options["row"], options["column"]]
            elif "row" in options:
                result = value.loc[options["row"]]
            elif "column" in options:
                result = value.loc[:, options["column"]]
            else:
                raise ValueError("ReduceOutput loc method requires 'row' or 'column'.")
        elif isinstance(value, pd.Series):
            result = value.loc[options["label"]]
        else:
            raise TypeError("ReduceOutput loc method requires a DataFrame or Series input.")

    elif method == "head":
        result = value.head(options.get("n", 5))

    elif method == "tail":
        result = value.tail(options.get("n", 5))

    elif method == "squeeze":
        if not hasattr(value, "squeeze"):
            raise TypeError("ReduceOutput squeeze method requires a pandas object.")
        result = value.squeeze()

    elif method == "scalar_agg":
        agg = options.get("agg")
        if not agg:
            raise ValueError("ReduceOutput scalar_agg method requires 'agg'.")
        if isinstance(value, pd.DataFrame):
            column = options.get("column")
            if not column:
                raise ValueError("ReduceOutput scalar_agg on DataFrame requires 'column'.")
            result = value[column].agg(agg)
        elif isinstance(value, pd.Series):
            result = value.agg(agg)
        else:
            raise TypeError("ReduceOutput scalar_agg method requires a DataFrame or Series input.")

    elif method == "item":
        if isinstance(value, pd.DataFrame):
            result = value.squeeze().item()
        elif isinstance(value, pd.Series):
            result = value.item()
        else:
            result = value

    elif method == "columnwise_extreme_index":
        if not isinstance(value, pd.DataFrame):
            raise TypeError("ReduceOutput columnwise_extreme_index method requires a DataFrame input.")
        result = _columnwise_extreme_index(
            value,
            anchor_extreme=options["anchor_extreme"],
            anchor_occurrence=options["anchor_occurrence"],
            target_extreme=options["target_extreme"],
            target_occurrence=options["target_occurrence"],
            direction=options["direction"],
            skipna=bool(options.get("skipna", True)),
        )

    else:
        raise ValueError(f"Unsupported ReduceOutput method: {method}")

    print(f"[ReduceOutput] Produced {_describe_result(result)} output")
    return result
