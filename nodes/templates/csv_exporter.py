import pandas as pd


def _as_export_frame(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        value_name = data.name or "count"
        index_names = [
            name if name is not None else f"level_{index}"
            for index, name in enumerate(data.index.names)
        ]
        if len(index_names) == 1:
            axis_name = index_names[0]
        else:
            axis_name = index_names
        return data.rename(value_name).rename_axis(axis_name).reset_index()
    return data


def csv_exporter(df: pd.DataFrame | pd.Series, output_path: str) -> str:
    """
    Exports DataFrame or Series to CSV.
    Node: CSVExporter
    """
    export_frame = _as_export_frame(df)
    export_frame.to_csv(output_path, index=False)
    print(f"[CSVExporter] Exported to {output_path}")
    return output_path
