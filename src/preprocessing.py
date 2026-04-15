from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    scale: bool = False,
) -> pd.DataFrame:
    if not feature_cols:
        raise ValueError("En az bir feature seçmelisiniz.")

    X = df[list(feature_cols)].copy()

    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype(str)

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median = X[col].median()
            X[col] = X[col].fillna(median if pd.notna(median) else 0)
        else:
            mode = X[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "missing"
            X[col] = X[col].fillna(fill).astype(str)

    X = pd.get_dummies(X, drop_first=False)
    X = X.astype(float)

    if scale and not X.empty:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X.values)
        X = pd.DataFrame(scaled, columns=X.columns, index=X.index)

    return X


def prepare_supervised(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = False,
):
    if target_col in feature_cols:
        feature_cols = [c for c in feature_cols if c != target_col]

    X = build_feature_matrix(df, feature_cols, scale=scale)
    y = df[target_col].loc[X.index]

    mask = y.notna()
    X, y = X[mask], y[mask]

    if X.empty:
        raise ValueError("Temizlik sonrası veri seti boş kaldı. Lütfen farklı kolonlar seçin.")

    stratify = None
    try:
        if y.dtype == object or y.nunique(dropna=True) < 20:
            counts = y.value_counts()
            if (counts >= 2).all():
                stratify = y
    except Exception:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test
