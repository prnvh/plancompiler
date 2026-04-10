import pandas as pd


def _describe_result(value) -> str:
    if isinstance(value, pd.DataFrame):
        return "DataFrame"
    if isinstance(value, pd.Series):
        return "Series"
    if pd.api.types.is_scalar(value):
        return "Scalar"
    return type(value).__name__


def reduce_output(value, method: str = "identity", **options):
    """
    Reduces a DataFrame or Series to a DataFrame, Series, or scalar.
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

    else:
        raise ValueError(f"Unsupported ReduceOutput method: {method}")

    print(f"[ReduceOutput] Produced {_describe_result(result)} output")
    return result
