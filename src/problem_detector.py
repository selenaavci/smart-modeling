"""Veri setine bakarak problem tipini (öneri) belirler."""
from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype


def suggest_problem_type(df: pd.DataFrame, target_col: str | None) -> str:
    """Hedef kolona göre classification/regression/clustering önerir.

    Sezgisel kurallar
    -----------------
    * Hedef yoksa  -> **clustering**
    * Hedef sayısal ve çok sayıda (>20) benzersiz değere sahipse -> **regression**
    * Aksi halde -> **classification**
    """
    if target_col is None or target_col not in df.columns:
        return "clustering"

    series = df[target_col].dropna()
    if series.empty:
        return "clustering"

    unique_count = series.nunique()
    unique_ratio = unique_count / max(len(series), 1)

    if is_numeric_dtype(series) and unique_count > 20 and unique_ratio > 0.02:
        return "regression"
    return "classification"
