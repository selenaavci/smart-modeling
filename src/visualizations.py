from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve


plt.style.use("dark_background")

_BG = "#0E1117"
_PANEL = "#1C1F26"
_GRID = "#3A3A3A"
_TEXT = "#FAFAFA"
_PRIMARY = "#4A90D9"
_ACCENT = "#F87171"

plt.rcParams.update(
    {
        "figure.facecolor": _BG,
        "axes.facecolor": _PANEL,
        "savefig.facecolor": _BG,
        "axes.edgecolor": _GRID,
        "axes.labelcolor": _TEXT,
        "axes.titlecolor": _TEXT,
        "xtick.color": _TEXT,
        "ytick.color": _TEXT,
        "text.color": _TEXT,
        "grid.color": _GRID,
        "axes.grid": False,
    }
)


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax, cbar_kws={"label": ""},
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc_curve(model, X_test, y_test_enc):
    if not hasattr(model, "predict_proba"):
        return None
    try:
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_enc, y_score)
        roc_auc = auc(fpr, tpr)
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=_PRIMARY, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color=_GRID)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", facecolor=_PANEL, edgecolor=_GRID)
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n: int = 15):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        return None

    series = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=True)
        .tail(top_n)
    )
    fig, ax = plt.subplots(figsize=(6, max(3, len(series) * 0.35)))
    series.plot(kind="barh", ax=ax, color=_PRIMARY)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Önem Skoru")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred):
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_test, y_pred, alpha=0.6, color=_PRIMARY, edgecolor="none")
    mn = float(min(y_test.min(), y_pred.min()))
    mx = float(max(y_test.max(), y_pred.max()))
    ax.plot([mn, mx], [mn, mx], linestyle="--", color=_ACCENT)
    ax.set_xlabel("Gerçek Değer")
    ax.set_ylabel("Tahmin")
    ax.set_title("Actual vs Predicted")
    fig.tight_layout()
    return fig


def plot_residuals(y_test, y_pred):
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_pred, residuals, alpha=0.6, color=_PRIMARY, edgecolor="none")
    ax.axhline(0, linestyle="--", color=_ACCENT)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Residual (gerçek - tahmin)")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    return fig
