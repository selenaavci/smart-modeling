"""
Smart Modeling — Streamlit UI
Self-service ML aracı (dark mode, sade arayüz).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import SUPPORTED_EXTENSIONS, load_dataframe  # noqa: E402
from src.exporter import build_excel_report  # noqa: E402
from src.model_engine import (  # noqa: E402
    CLASSIFIERS,
    REGRESSORS,
    run_clustering,
    select_best_classification,
    select_best_regression,
    train_classification,
    train_regression,
)
from src.model_info import (  # noqa: E402
    METRIC_DESCRIPTIONS,
    MODEL_DESCRIPTIONS,
    PROBLEM_DESCRIPTIONS,
)
from src.preprocessing import build_feature_matrix, prepare_supervised  # noqa: E402
from src.problem_detector import suggest_problem_type  # noqa: E402
from src.visualizations import (  # noqa: E402
    plot_actual_vs_predicted,
    plot_clusters_2d,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_residuals,
    plot_roc_curve,
)


# ---------------------------------------------------------------------------
# Streamlit sürüm uyumluluğu
# ---------------------------------------------------------------------------
def safe_rerun() -> None:
    """Streamlit sürümünden bağımsız rerun çağrısı."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# ---------------------------------------------------------------------------
# Sayfa ayarları
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Modeling Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Smart Modeling Agent")
st.caption(
    "Veri setinizi yükleyin, problem tipini belirleyin ve otomatik olarak "
    "eğitilen modelleri karşılaştırın."
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "df": None,
    "file_name": None,
    "target_col": None,
    "problem_type": None,
    "feature_cols": [],
    "results": None,
}
for _key, _value in _DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


# ---------------------------------------------------------------------------
# Kenar çubuğu — Dosya yükleme
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Dosya Yükleme")
    uploaded_file = st.file_uploader(
        "CSV, Excel veya XML dosyası yükleyin",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        help="Maksimum dosya boyutu: 5 GB.",
    )

    st.divider()
    st.markdown(
        "**Desteklenen formatlar:** CSV, XLSX, XLS, XML\n\n"
        "**Özellikler:**\n"
        "- Otomatik problem tipi önerisi\n"
        "- Sınıflandırma, regresyon, kümeleme\n"
        "- Otomatik veri ön işleme\n"
        "- Model karşılaştırma ve görselleştirme\n"
        "- Excel formatında dışa aktarma"
    )

    st.divider()
    if st.button("Oturumu Sıfırla", use_container_width=True):
        for _k in list(st.session_state.keys()):
            del st.session_state[_k]
        safe_rerun()


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        if st.session_state.get("file_name") != uploaded_file.name:
            with st.spinner("Veri okunuyor..."):
                df_new = load_dataframe(uploaded_file)
            st.session_state["df"] = df_new
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["results"] = None
            st.session_state["target_col"] = None
            st.session_state["problem_type"] = None
            st.session_state["feature_cols"] = []
    except Exception as exc:  # noqa: BLE001
        st.error(f"Dosya okunamadı: {exc}")
        st.stop()

df = st.session_state.get("df")

if df is None:
    st.info("Başlamak için kenar çubuğundan bir dosya yükleyin.")
    st.stop()


# ---------------------------------------------------------------------------
# Sekmeler
# ---------------------------------------------------------------------------
tab_preview, tab_profile, tab_problem, tab_train, tab_results = st.tabs(
    [
        "Veri Önizleme",
        "Veri Profili",
        "Problem Tanımı",
        "Model Eğitimi",
        "Sonuçlar",
    ]
)


# ===================== Sekme 1: Veri Önizleme =============================
with tab_preview:
    st.subheader("Veri Seti Önizleme")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satır Sayısı", f"{len(df):,}")
    c2.metric("Sütun Sayısı", len(df.columns))
    c3.metric("Eksik Değer", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Numerik Sütun", df.select_dtypes(include="number").shape[1])

    st.dataframe(df.head(100), use_container_width=True, height=400)

    with st.expander("Sütun Bilgileri"):
        rows = []
        for col in df.columns:
            samples = df[col].dropna()
            rows.append(
                {
                    "Sütun": col,
                    "Veri Tipi": str(df[col].dtype),
                    "Benzersiz Değer": int(df[col].nunique(dropna=True)),
                    "Eksik Değer": int(df[col].isna().sum()),
                    "Örnek Değer": str(samples.iloc[0]) if len(samples) > 0 else "-",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ===================== Sekme 2: Veri Profili =============================
with tab_profile:
    st.subheader("Veri Profili")

    num_df = df.select_dtypes(include="number")
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not num_df.empty:
        st.markdown("#### Numerik Sütun İstatistikleri")
        st.dataframe(num_df.describe().T, use_container_width=True)
    else:
        st.info("Numerik sütun bulunamadı.")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.markdown("#### Eksik Değerler")
        miss_df = pd.DataFrame(
            {
                "Adet": missing.values,
                "Oran (%)": (missing.values / len(df) * 100).round(2),
            },
            index=missing.index,
        )
        miss_df.index.name = "Sütun"
        st.dataframe(miss_df, use_container_width=True)

    if cat_cols:
        st.markdown("#### Kategorik Sütun Kardinalitesi")
        card_rows = []
        for col in cat_cols:
            nunique = int(df[col].nunique(dropna=True))
            if nunique <= 10:
                tip = "Düşük"
            elif nunique <= 50:
                tip = "Orta"
            else:
                tip = "Yüksek"
            card_rows.append(
                {"Sütun": col, "Benzersiz Değer": nunique, "Kardinalite": tip}
            )
        st.dataframe(pd.DataFrame(card_rows), use_container_width=True, hide_index=True)


# ===================== Sekme 3: Problem Tanımı =============================
with tab_problem:
    st.subheader("Problem Tanımı")
    st.markdown(
        "Tahmin edilecek hedef sütunu seçin. Hedef yoksa kümeleme (clustering) "
        "moduna geçilir."
    )

    options = ["(Yok - Kümeleme)"] + list(df.columns)
    current_target = st.session_state.get("target_col") or "(Yok - Kümeleme)"
    try:
        default_idx = options.index(current_target)
    except ValueError:
        default_idx = 0
    target = st.selectbox("Hedef Sütun", options, index=default_idx)

    if target == "(Yok - Kümeleme)":
        st.session_state["target_col"] = None
        suggested = "clustering"
    else:
        st.session_state["target_col"] = target
        suggested = suggest_problem_type(df, target)

    st.markdown("#### Önerilen Problem Tipi")
    st.info(PROBLEM_DESCRIPTIONS[suggested])

    type_labels = {
        "classification": "Sınıflandırma (Classification)",
        "regression": "Regresyon (Regression)",
        "clustering": "Kümeleme (Clustering)",
    }
    keys = list(type_labels.keys())
    chosen_label = st.radio(
        "Problem tipini seçin",
        list(type_labels.values()),
        index=keys.index(suggested),
        horizontal=True,
    )
    st.session_state["problem_type"] = keys[
        list(type_labels.values()).index(chosen_label)
    ]

    st.markdown("#### Özellik (Feature) Seçimi")
    st.caption(
        "Varsayılan olarak hedef dışındaki tüm sütunlar kullanılır. Alakasız "
        "sütunlar listeden çıkarılabilir."
    )

    if st.session_state["problem_type"] == "clustering":
        available = list(df.columns)
    else:
        available = [c for c in df.columns if c != st.session_state["target_col"]]

    selected_features = st.multiselect(
        "Kullanılacak sütunlar",
        options=available,
        default=available,
    )
    st.session_state["feature_cols"] = selected_features

    if not selected_features:
        st.warning("En az bir özellik seçilmelidir.")


# ===================== Sekme 4: Model Eğitimi =============================
with tab_train:
    st.subheader("Model Eğitimi")

    ptype = st.session_state.get("problem_type")
    feature_cols = st.session_state.get("feature_cols") or []

    if not ptype:
        st.info("Önce **Problem Tanımı** sekmesinden problem tipini belirleyin.")
    elif not feature_cols:
        st.warning("**Problem Tanımı** sekmesinden en az bir özellik seçin.")
    else:
        if ptype == "classification":
            model_choices = list(CLASSIFIERS.keys())
            label_map = {
                "Logistic Regression": "Logistic Regression",
                "Random Forest": "Random Forest (Classifier)",
                "Gradient Boosting": "Gradient Boosting (Classifier)",
            }
        elif ptype == "regression":
            model_choices = list(REGRESSORS.keys())
            label_map = {
                "Linear Regression": "Linear Regression",
                "Random Forest": "Random Forest (Regressor)",
                "Gradient Boosting": "Gradient Boosting (Regressor)",
            }
        else:
            model_choices = ["K-Means", "DBSCAN"]
            label_map = {"K-Means": "K-Means", "DBSCAN": "DBSCAN"}

        st.markdown("#### Model Seçimi")
        selected_models = st.multiselect(
            "Denenecek modeller",
            options=model_choices,
            default=model_choices,
        )

        with st.expander("Model Açıklamaları"):
            for m in selected_models:
                desc = MODEL_DESCRIPTIONS.get(label_map[m], "")
                st.markdown(f"**{m}** — {desc}")

        test_size = 0.2
        n_clusters = 3
        eps = 0.5
        min_samples = 5

        if ptype in ("classification", "regression"):
            st.markdown("#### Eğitim Parametreleri")
            test_size = (
                st.slider(
                    "Test seti oranı (%)",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Verinin ne kadarı test için ayrılacak.",
                )
                / 100
            )
        elif ptype == "clustering":
            st.markdown("#### Kümeleme Parametreleri")
            cc1, cc2, cc3 = st.columns(3)
            n_clusters = cc1.slider("K-Means küme sayısı", 2, 15, 3)
            eps = cc2.slider("DBSCAN eps", 0.1, 3.0, 0.5, step=0.1)
            min_samples = cc3.slider("DBSCAN min örnek", 2, 20, 5)

        if st.button("Eğitimi Başlat", type="primary", use_container_width=True):
            if not selected_models:
                st.warning("En az bir model seçilmelidir.")
            else:
                try:
                    with st.spinner("Modeller eğitiliyor..."):
                        if ptype == "classification":
                            X_train, X_test, y_train, y_test = prepare_supervised(
                                df,
                                st.session_state["target_col"],
                                feature_cols,
                                test_size=test_size,
                            )
                            results = train_classification(
                                X_train, X_test, y_train, y_test, selected_models
                            )
                            best = select_best_classification(results)
                            st.session_state["results"] = {
                                "type": "classification",
                                "results": results,
                                "best": best,
                                "X_test": X_test,
                                "feature_names": list(X_train.columns),
                            }

                        elif ptype == "regression":
                            X_train, X_test, y_train, y_test = prepare_supervised(
                                df,
                                st.session_state["target_col"],
                                feature_cols,
                                test_size=test_size,
                            )
                            results = train_regression(
                                X_train, X_test, y_train, y_test, selected_models
                            )
                            best = select_best_regression(results)
                            st.session_state["results"] = {
                                "type": "regression",
                                "results": results,
                                "best": best,
                                "X_test": X_test,
                                "feature_names": list(X_train.columns),
                            }

                        else:
                            X = build_feature_matrix(df, feature_cols, scale=True)
                            results = run_clustering(
                                X,
                                selected_models,
                                n_clusters=n_clusters,
                                eps=eps,
                                min_samples=min_samples,
                            )
                            st.session_state["results"] = {
                                "type": "clustering",
                                "results": results,
                                "X": X,
                            }
                    st.success(
                        "Eğitim tamamlandı. **Sonuçlar** sekmesinden detaylar "
                        "incelenebilir."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Eğitim sırasında hata: {exc}")


# ===================== Sekme 5: Sonuçlar =============================
with tab_results:
    st.subheader("Sonuçlar ve Dışa Aktarma")

    res = st.session_state.get("results")

    if not res:
        st.info(
            "Henüz bir model eğitilmedi. **Model Eğitimi** sekmesinden eğitim "
            "başlatılabilir."
        )
    elif res["type"] in ("classification", "regression"):
        rows = []
        for name, info in res["results"].items():
            row = {"Model": name}
            row.update({k: round(float(v), 4) for k, v in info["metrics"].items()})
            rows.append(row)
        comp_df = pd.DataFrame(rows)

        st.markdown("#### Model Karşılaştırma")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.success(f"En iyi model: **{res['best']}**")

        with st.expander("Metrik Açıklamaları"):
            for col in comp_df.columns:
                if col == "Model":
                    continue
                desc = METRIC_DESCRIPTIONS.get(col)
                if desc:
                    st.markdown(f"- **{col}** — {desc}")

        best_info = res["results"][res["best"]]
        if res["type"] == "classification":
            train_m = best_info["metrics"].get("Train Accuracy", 0.0)
            test_m = best_info["metrics"].get("Accuracy", 0.0)
            label_train, label_test = "eğitim doğruluğu", "test doğruluğu"
        else:
            train_m = best_info["metrics"].get("Train R2", 0.0)
            test_m = best_info["metrics"].get("R2", 0.0)
            label_train, label_test = "eğitim R²", "test R²"

        if train_m - test_m > 0.15:
            st.warning(
                f"Overfitting riski: {label_train} ({train_m:.2f}) "
                f"{label_test}'nden ({test_m:.2f}) belirgin şekilde yüksek. "
                "Daha fazla veri, daha az özellik veya daha basit bir model "
                "denenebilir."
            )

        st.markdown("#### Model Detayları")
        model_names = list(res["results"].keys())
        chosen = st.selectbox(
            "Detayı görüntülenecek model",
            model_names,
            index=model_names.index(res["best"]),
        )
        info = res["results"][chosen]

        if res["type"] == "classification":
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(
                    plot_confusion_matrix(info["confusion_matrix"], info["classes"])
                )
            with c2:
                le = info["label_encoder"]
                y_test_enc = le.transform(pd.Series(info["y_test"]).astype(str))
                fig = plot_roc_curve(info["model"], res["X_test"], y_test_enc)
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.info(
                        "ROC eğrisi yalnızca ikili sınıflandırma için gösterilir."
                    )

            fig = plot_feature_importance(info["model"], res["feature_names"])
            if fig is not None:
                st.pyplot(fig)
            else:
                st.info("Bu model için feature importance bilgisi bulunmuyor.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_actual_vs_predicted(info["y_test"], info["y_pred"]))
            with c2:
                st.pyplot(plot_residuals(info["y_test"], info["y_pred"]))

            fig = plot_feature_importance(info["model"], res["feature_names"])
            if fig is not None:
                st.pyplot(fig)

        sheets = {"Model Karsilastirma": comp_df}
        best_info = res["results"][res["best"]]
        if res["type"] == "classification":
            pred_df = pd.DataFrame(
                {"Gerçek": best_info["y_test"], "Tahmin": best_info["y_pred"]}
            )
        else:
            pred_df = pd.DataFrame(
                {
                    "Gerçek": list(best_info["y_test"]),
                    "Tahmin": list(best_info["y_pred"]),
                }
            )
        sheets["Tahminler"] = pred_df

        excel_bytes = build_excel_report(sheets)
        st.download_button(
            "Sonuçları Excel olarak indir",
            data=excel_bytes,
            file_name="smart_modeling_sonuclari.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    else:  # clustering
        rows = []
        for name, info in res["results"].items():
            row = {"Model": name}
            row.update({k: round(float(v), 4) for k, v in info["metrics"].items()})
            rows.append(row)
        comp_df = pd.DataFrame(rows)

        st.markdown("#### Kümeleme Sonuçları")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        with st.expander("Metrik Açıklamaları"):
            for col in comp_df.columns:
                if col == "Model":
                    continue
                desc = METRIC_DESCRIPTIONS.get(col)
                if desc:
                    st.markdown(f"- **{col}** — {desc}")

        st.markdown("#### Küme Görselleştirmesi")
        chosen = st.selectbox("Model", list(res["results"].keys()))
        labels = res["results"][chosen]["labels"]
        st.pyplot(plot_clusters_2d(res["X"], labels, title=f"{chosen} - PCA 2D"))

        st.markdown("#### Etiketlenmiş Veri (ilk 20 satır)")
        clustered_df = df.copy()
        clustered_df[f"cluster_{chosen}"] = labels
        st.dataframe(clustered_df.head(20), use_container_width=True)

        sheets = {
            "Kumeleme Ozeti": comp_df,
            "Kumelenmis Veri": clustered_df,
        }
        excel_bytes = build_excel_report(sheets)
        st.download_button(
            "Sonuçları Excel olarak indir",
            data=excel_bytes,
            file_name="smart_modeling_kumeleme.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
