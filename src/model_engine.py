"""Model eğitimi, değerlendirme ve karşılaştırma."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder


CLASSIFIERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, n_jobs=None),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
}

REGRESSORS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
}


def train_classification(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_models: Sequence[str],
) -> Dict[str, dict]:
    """Seçili sınıflandırma modellerini eğitip metrikleri döner."""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.astype(str))
    y_test_enc = le.transform(y_test.astype(str))
    n_classes = len(le.classes_)

    results: Dict[str, dict] = {}
    for name in selected_models:
        if name not in CLASSIFIERS:
            continue
        model = CLASSIFIERS[name]()
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        metrics = {
            "Accuracy": float(accuracy_score(y_test_enc, y_pred)),
            "F1 (weighted)": float(f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)),
            "Precision": float(precision_score(y_test_enc, y_pred, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y_test_enc, y_pred, average="weighted", zero_division=0)),
            "Train Accuracy": float(accuracy_score(y_train_enc, y_pred_train)),
        }

        if hasattr(model, "predict_proba") and n_classes == 2:
            try:
                proba = model.predict_proba(X_test)[:, 1]
                metrics["ROC-AUC"] = float(roc_auc_score(y_test_enc, proba))
            except Exception:  # noqa: BLE001
                pass

        results[name] = {
            "model": model,
            "metrics": metrics,
            "y_pred": le.inverse_transform(y_pred),
            "y_test": le.inverse_transform(y_test_enc),
            "label_encoder": le,
            "confusion_matrix": confusion_matrix(y_test_enc, y_pred),
            "classes": le.classes_,
        }
    return results


def train_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_models: Sequence[str],
) -> Dict[str, dict]:
    """Seçili regresyon modellerini eğitip metrikleri döner."""
    # Sayısal'a zorla (kullanıcı yanlış bir kolon seçmiş olabilir)
    y_train_num = pd.to_numeric(y_train, errors="coerce")
    y_test_num = pd.to_numeric(y_test, errors="coerce")
    mask_train = y_train_num.notna()
    mask_test = y_test_num.notna()
    X_train, y_train_num = X_train[mask_train], y_train_num[mask_train]
    X_test, y_test_num = X_test[mask_test], y_test_num[mask_test]

    if X_train.empty or X_test.empty:
        raise ValueError("Regresyon için hedef kolon sayısala dönüştürülemedi.")

    results: Dict[str, dict] = {}
    for name in selected_models:
        if name not in REGRESSORS:
            continue
        model = REGRESSORS[name]()
        model.fit(X_train, y_train_num)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        mse = mean_squared_error(y_test_num, y_pred)
        metrics = {
            "RMSE": float(np.sqrt(mse)),
            "MAE": float(mean_absolute_error(y_test_num, y_pred)),
            "R2": float(r2_score(y_test_num, y_pred)),
            "Train R2": float(r2_score(y_train_num, y_pred_train)),
        }

        results[name] = {
            "model": model,
            "metrics": metrics,
            "y_pred": np.asarray(y_pred),
            "y_test": np.asarray(y_test_num),
        }
    return results


def run_clustering(
    X: pd.DataFrame,
    selected_models: Sequence[str],
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Dict[str, dict]:
    """K-Means ve/veya DBSCAN çalıştırır."""
    results: Dict[str, dict] = {}

    if "K-Means" in selected_models:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        metrics = {"Inertia": float(km.inertia_), "Küme Sayısı": float(n_clusters)}
        if len(set(labels)) > 1:
            try:
                metrics["Silhouette"] = float(silhouette_score(X, labels))
            except Exception:  # noqa: BLE001
                pass
        results["K-Means"] = {"model": km, "labels": labels, "metrics": metrics}

    if "DBSCAN" in selected_models:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        metrics = {
            "Küme Sayısı": float(n_found),
            "Gürültü Noktası": float(int((labels == -1).sum())),
        }
        mask = labels != -1
        if mask.sum() > 1 and len(set(labels[mask])) > 1:
            try:
                metrics["Silhouette"] = float(silhouette_score(X[mask], labels[mask]))
            except Exception:  # noqa: BLE001
                pass
        results["DBSCAN"] = {"model": db, "labels": labels, "metrics": metrics}

    return results


def select_best_classification(results: Dict[str, dict]) -> str:
    return max(results, key=lambda name: results[name]["metrics"].get("F1 (weighted)", 0.0))


def select_best_regression(results: Dict[str, dict]) -> str:
    return max(results, key=lambda name: results[name]["metrics"].get("R2", float("-inf")))
