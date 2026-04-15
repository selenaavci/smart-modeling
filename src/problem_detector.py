from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def suggest_problem_type(df: pd.DataFrame, target_col: str | None) -> str:
    if target_col is None or target_col not in df.columns:
        return "classification"

    series = df[target_col].dropna()
    if series.empty:
        return "classification"

    unique_count = series.nunique()
    unique_ratio = unique_count / max(len(series), 1)

    if is_numeric_dtype(series) and unique_count > 20 and unique_ratio > 0.02:
        return "regression"
    return "classification"
