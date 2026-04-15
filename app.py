from __future__ import annotations

import io
import json
import pickle
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import SUPPORTED_EXTENSIONS, load_dataframe
from src.exporter import build_excel_report
from src.model_engine import (
    CLASSIFIERS,
    REGRESSORS,
    select_best_classification,
    select_best_regression,
    train_classification,
    train_regression,
)
from src.model_info import (
    METRIC_DESCRIPTIONS,
    MODEL_DESCRIPTIONS,
    PROBLEM_DESCRIPTIONS,
)
from src.preprocessing import prepare_supervised
from src.problem_detector import suggest_problem_type
from src.visualizations import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_residuals,
    plot_roc_curve,
)


OVERFIT_ACC_THRESHOLD = 0.97
OVERFIT_R2_THRESHOLD = 0.97
OVERFIT_GAP_THRESHOLD = 0.15


def safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


st.set_page_config(
    page_title="Smart Modeling Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Smart Modeling Agent")
st.caption(
    "Veri setinizi yükleyin, problem tipini belirleyin ve otomatik olarak "
    "eğitilen regresyon/sınıflandırma modellerini karşılaştırın."
)

st.warning(
    " **Bu uygulama bir hızlı prototipleme / keşif aracıdır.** "
    "Eğitilen modeller **doğrudan canlıya (production) alınmak için uygun değildir.** "
    "Deployment öncesinde model; veri kalitesi, data drift, bias/fairness, "
    "güvenlik ve performans testleri gibi ek validasyon ve kontrol süreçlerinden "
    "geçirilmeli, **Big Data & MLOps ekibiyle birlikte değerlendirilmelidir.**"
)


_DEFAULTS = {
    "df": None,
    "file_name": None,
    "target_col": None,
    "id_col": None,
    "problem_type": None,
    "feature_cols": [],
    "use_cv": False,
    "use_scaling": False,
    "results": None,
}
for _key, _value in _DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


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
        "- Otomatik problem tipi önerisi (sınıflandırma / regresyon)\n"
        "- Tanımlayıcı (ID) kolonu seçme\n"
        "- Cross-validation ve standardizasyon seçenekleri\n"
        "- Otomatik veri ön işleme\n"
        "- Model karşılaştırma ve görselleştirme\n"
        "- Excel: train/test ayrımı + tüm veri seti üzerinde tahminler"
    )

    st.divider()
    if st.button("Oturumu Sıfırla", use_container_width=True):
        for _k in list(st.session_state.keys()):
            del st.session_state[_k]
        safe_rerun()


if uploaded_file is not None:
    try:
        if st.session_state.get("file_name") != uploaded_file.name:
            with st.spinner("Veri okunuyor..."):
                df_new = load_dataframe(uploaded_file)
            st.session_state["df"] = df_new
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["results"] = None
            st.session_state["target_col"] = None
            st.session_state["id_col"] = None
            st.session_state["problem_type"] = None
            st.session_state["feature_cols"] = []
    except Exception as exc:
        st.error(f"Dosya okunamadı: {exc}")
        st.stop()

df = st.session_state.get("df")

if df is None:
    st.info("Başlamak için kenar çubuğundan bir dosya yükleyin.")
    st.stop()


tab_preview, tab_profile, tab_problem, tab_train, tab_results = st.tabs(
    [
        "Veri Önizleme",
        "Veri Profili",
        "Problem Tanımı",
        "Model Eğitimi",
        "Sonuçlar",
    ]
)


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


with tab_problem:
    st.subheader("Problem Tanımı")

    st.markdown("#### Tanımlayıcı (ID) Kolonu")
    st.caption(
        "Her satırı tanımlayan kolon varsa seçin (örn. müşteri_no, sipariş_id). "
        "Tanımlayıcı kolon özellik olarak kullanılmaz; Excel çıktısında her "
        "satırı tanımlamaya yarar."
    )
    id_options = ["(Tanımlayıcı yok)"] + list(df.columns)
    current_id = st.session_state.get("id_col") or "(Tanımlayıcı yok)"
    try:
        id_default_idx = id_options.index(current_id)
    except ValueError:
        id_default_idx = 0
    id_choice = st.selectbox("Tanımlayıcı kolon", id_options, index=id_default_idx)
    st.session_state["id_col"] = None if id_choice == "(Tanımlayıcı yok)" else id_choice

    st.markdown("#### Hedef Kolon")
    st.caption("Tahmin edilecek sütunu seçin.")
    target_options = [c for c in df.columns if c != st.session_state.get("id_col")]
    current_target = st.session_state.get("target_col")
    if current_target not in target_options:
        current_target = target_options[0] if target_options else None
    target = st.selectbox(
        "Hedef Sütun",
        target_options,
        index=target_options.index(current_target) if current_target in target_options else 0,
    )
    st.session_state["target_col"] = target
    suggested = suggest_problem_type(df, target)

    st.markdown("#### Önerilen Problem Tipi")
    st.info(PROBLEM_DESCRIPTIONS[suggested])

    type_labels = {
        "classification": "Sınıflandırma (Classification)",
        "regression": "Regresyon (Regression)",
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

    st.markdown("#### Eğitim Seçenekleri")
    oc1, oc2 = st.columns(2)
    with oc1:
        st.session_state["use_cv"] = st.checkbox(
            "Cross-validation uygula (5-fold)",
            value=st.session_state.get("use_cv", False),
            help=(
                "Modelin genelleme başarısını daha güvenilir ölçmek için eğitim "
                "seti üzerinde 5-fold cross-validation çalıştırır."
            ),
        )
    with oc2:
        st.session_state["use_scaling"] = st.checkbox(
            "Standardizasyon uygula (StandardScaler)",
            value=st.session_state.get("use_scaling", False),
            help=(
                "Sayısal özellikleri ortalaması 0, standart sapması 1 olacak "
                "şekilde ölçekler. Özellikle Logistic/Linear Regression için "
                "önerilir."
            ),
        )

    st.markdown("#### Özellik (Feature) Seçimi")
    st.caption(
        "Varsayılan olarak hedef ve tanımlayıcı dışındaki tüm sütunlar kullanılır. "
        "Alakasız sütunlar listeden çıkarılabilir."
    )

    exclude = {st.session_state.get("target_col"), st.session_state.get("id_col")}
    available = [c for c in df.columns if c not in exclude and c is not None]

    previous = st.session_state.get("feature_cols") or available
    default_feats = [c for c in previous if c in available] or available
    selected_features = st.multiselect(
        "Kullanılacak sütunlar",
        options=available,
        default=default_feats,
    )
    st.session_state["feature_cols"] = selected_features

    if not selected_features:
        st.warning("En az bir özellik seçilmelidir.")


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
        else:
            model_choices = list(REGRESSORS.keys())
            label_map = {
                "Linear Regression": "Linear Regression",
                "Random Forest": "Random Forest (Regressor)",
                "Gradient Boosting": "Gradient Boosting (Regressor)",
            }

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

        info_cols = st.columns(2)
        info_cols[0].info(
            f"Cross-validation: {'Açık (5-fold)' if st.session_state.get('use_cv') else 'Kapalı'}"
        )
        info_cols[1].info(
            f"Standardizasyon: {'Açık' if st.session_state.get('use_scaling') else 'Kapalı'}"
        )

        if st.button("Eğitimi Başlat", type="primary", use_container_width=True):
            if not selected_models:
                st.warning("En az bir model seçilmelidir.")
            else:
                try:
                    progress = st.progress(0, text="Veri seti inceleniyor...")
                    time.sleep(2.0)

                    progress.progress(20, text="Özellikler hazırlanıyor...")
                    time.sleep(1.8)

                    progress.progress(40, text="Train-test ayırımı yapılıyor...")
                    X_train, X_test, y_train, y_test = prepare_supervised(
                        df,
                        st.session_state["target_col"],
                        feature_cols,
                        test_size=test_size,
                        scale=st.session_state.get("use_scaling", False),
                    )
                    time.sleep(2.0)

                    progress.progress(
                        60,
                        text=(
                            "Modeller eğitiliyor"
                            + (" (cross-validation ile)..." if st.session_state.get("use_cv") else "...")
                        ),
                    )
                    if ptype == "classification":
                        results = train_classification(
                            X_train, X_test, y_train, y_test, selected_models,
                            use_cv=st.session_state.get("use_cv", False),
                        )
                        best = select_best_classification(results)
                    else:
                        results = train_regression(
                            X_train, X_test, y_train, y_test, selected_models,
                            use_cv=st.session_state.get("use_cv", False),
                        )
                        best = select_best_regression(results)
                    time.sleep(2.5)

                    progress.progress(90, text="Metrikler derleniyor...")
                    time.sleep(1.7)
                    progress.progress(100, text="Tamamlandı.")

                    st.session_state["results"] = {
                        "type": ptype,
                        "results": results,
                        "best": best,
                        "X_test": X_test,
                        "X_train": X_train,
                        "feature_names": list(X_train.columns),
                        "train_index": list(X_train.index),
                        "test_index": list(X_test.index),
                        "id_col": st.session_state.get("id_col"),
                        "target_col": st.session_state.get("target_col"),
                    }
                    st.success(
                        "Eğitim tamamlandı. **Sonuçlar** sekmesinden detaylar incelenebilir."
                    )
                except Exception as exc:
                    st.error(f"Eğitim sırasında hata: {exc}")


def _check_overfitting(res_type: str, metrics: dict) -> bool:
    if res_type == "classification":
        train_m = metrics.get("Train Accuracy", 0.0)
        test_m = metrics.get("Accuracy", 0.0)
        if test_m >= OVERFIT_ACC_THRESHOLD:
            return True
    else:
        train_m = metrics.get("Train R2", 0.0)
        test_m = metrics.get("R2", 0.0)
        if test_m >= OVERFIT_R2_THRESHOLD:
            return True
    if (train_m - test_m) > OVERFIT_GAP_THRESHOLD:
        return True
    return False


def _build_full_predictions(df_src: pd.DataFrame, res: dict) -> pd.DataFrame:
    """Tüm veri seti + son kolonda model tahmini (train + test birleşik)."""
    info = res["results"][res["best"]]
    pred_series = pd.Series(index=df_src.index, dtype=object)

    for idx, val in zip(res["train_index"], info["y_pred_train"]):
        pred_series.loc[idx] = val
    for idx, val in zip(res["test_index"], info["y_pred"]):
        pred_series.loc[idx] = val

    split_series = pd.Series("kullanılmadı", index=df_src.index, dtype=object)
    split_series.loc[res["train_index"]] = "train"
    split_series.loc[res["test_index"]] = "test"

    full = df_src.copy()
    full.insert(0, "Satır No", full.index)
    full["Veri Seti Ayırımı"] = split_series.values
    full[f"Tahmin ({res['best']})"] = pred_series.values
    return full


with tab_results:
    st.subheader("Sonuçlar ve Dışa Aktarma")

    res = st.session_state.get("results")

    if not res:
        st.info(
            "Henüz bir model eğitilmedi. **Model Eğitimi** sekmesinden eğitim başlatılabilir."
        )
    else:
        rows = []
        for name, info in res["results"].items():
            row = {"Model": name}
            row.update({k: round(float(v), 4) for k, v in info["metrics"].items()})
            rows.append(row)
        comp_df = pd.DataFrame(rows)

        st.markdown("#### Model Karşılaştırma")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.success(f"En iyi model: **{res['best']}**")

        best_info = res["results"][res["best"]]

        if _check_overfitting(res["type"], best_info["metrics"]):
            st.warning(
                "️ **Bu model overfitting olmuş gibi duruyor.** "
                "Test metrikleri olağanüstü yüksek ya da eğitim–test farkı büyük. "
                "Bu öğrenme sonucundaki regresyon/sınıflandırma ile bir aksiyon "
                "almadan önce **Big Data & MLOps ekibine danışabilirsin.**"
            )

        with st.expander("Metrik Açıklamaları"):
            for col in comp_df.columns:
                if col == "Model":
                    continue
                desc = METRIC_DESCRIPTIONS.get(col)
                if desc:
                    st.markdown(f"- **{col}** — {desc}")

        st.markdown("#### Modeling Artifact — Çalıştırma Özeti")
        st.caption(
            "Bu çalıştırmada yapılan tüm adımların özeti. Aşağıdan ilgili "
            "artifact paketini (model + train/test + feature listesi + metadata) "
            "indirebilirsiniz."
        )

        artifact_meta = {
            "zaman_damgasi": datetime.now().isoformat(timespec="seconds"),
            "problem_tipi": res["type"],
            "hedef_kolon": res.get("target_col"),
            "tanimlayici_kolon": res.get("id_col"),
            "secilen_feature_sayisi": len(st.session_state.get("feature_cols") or []),
            "secilen_featurelar": list(st.session_state.get("feature_cols") or []),
            "engineered_feature_sayisi": len(res["feature_names"]),
            "train_satir_sayisi": len(res["train_index"]),
            "test_satir_sayisi": len(res["test_index"]),
            "preprocessing": {
                "missing_value_imputation": "numerik: median, kategorik: mode",
                "kategorik_encoding": "One-Hot (get_dummies)",
                "standardizasyon": bool(st.session_state.get("use_scaling", False)),
            },
            "cross_validation": {
                "etkin": bool(st.session_state.get("use_cv", False)),
                "folds": 5 if st.session_state.get("use_cv", False) else None,
            },
            "denenen_modeller": list(res["results"].keys()),
            "en_iyi_model": res["best"],
            "en_iyi_model_metrikleri": {
                k: float(v) for k, v in res["results"][res["best"]]["metrics"].items()
            },
        }

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(
                f"- **Problem tipi:** {artifact_meta['problem_tipi']}\n"
                f"- **Hedef kolon:** `{artifact_meta['hedef_kolon']}`\n"
                f"- **Tanımlayıcı:** `{artifact_meta['tanimlayici_kolon'] or '—'}`\n"
                f"- **Seçilen feature sayısı:** {artifact_meta['secilen_feature_sayisi']}\n"
                f"- **Engineered feature sayısı:** {artifact_meta['engineered_feature_sayisi']}"
            )
        with sc2:
            st.markdown(
                f"- **Train satır:** {artifact_meta['train_satir_sayisi']:,}\n"
                f"- **Test satır:** {artifact_meta['test_satir_sayisi']:,}\n"
                f"- **Standardizasyon:** {'Açık' if artifact_meta['preprocessing']['standardizasyon'] else 'Kapalı'}\n"
                f"- **Cross-validation:** {'Açık (5-fold)' if artifact_meta['cross_validation']['etkin'] else 'Kapalı'}\n"
                f"- **En iyi model:** `{artifact_meta['en_iyi_model']}`"
            )

        with st.expander("Denenen modeller ve metrikleri (JSON)"):
            st.json(artifact_meta)

        def _build_artifact_zip() -> bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(
                    "metadata.json",
                    json.dumps(artifact_meta, ensure_ascii=False, indent=2),
                )
                zf.writestr(
                    "features.json",
                    json.dumps(
                        {
                            "selected_features": artifact_meta["secilen_featurelar"],
                            "engineered_features": res["feature_names"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
                zf.writestr(
                    "model.pkl",
                    pickle.dumps(res["results"][res["best"]]["model"]),
                )
                x_train_out = res["X_train"].copy()
                x_test_out = res["X_test"].copy()
                zf.writestr("X_train.csv", x_train_out.to_csv(index=True))
                zf.writestr("X_test.csv", x_test_out.to_csv(index=True))
                target = res.get("target_col") or "target"
                y_train_df = pd.DataFrame(
                    {target: res["results"][res["best"]]["y_train"]},
                    index=res["train_index"],
                )
                y_test_df = pd.DataFrame(
                    {target: res["results"][res["best"]]["y_test"]},
                    index=res["test_index"],
                )
                zf.writestr("y_train.csv", y_train_df.to_csv(index=True))
                zf.writestr("y_test.csv", y_test_df.to_csv(index=True))
                zf.writestr(
                    "README.txt",
                    (
                        "Smart Modeling Agent — Artifact Paketi\n"
                        "======================================\n\n"
                        "Bu paket yalnızca yeniden üretilebilirlik ve inceleme amaçlıdır.\n"
                        "Model DOĞRUDAN PRODUCTION'A ALINMAMALIDIR. Deployment öncesi\n"
                        "veri kalitesi, drift, bias/fairness, güvenlik ve performans\n"
                        "testlerinin Big Data & MLOps ekibiyle birlikte yapılması gerekir.\n\n"
                        "İçerik:\n"
                        "- metadata.json    : Çalıştırma özeti ve parametreler\n"
                        "- features.json    : Seçilen ve engineered feature listesi\n"
                        "- model.pkl        : Eğitilmiş en iyi model (pickle)\n"
                        "- X_train.csv      : Eğitim seti (engineered feature'lar)\n"
                        "- X_test.csv       : Test seti (engineered feature'lar)\n"
                        "- y_train.csv      : Eğitim seti hedef değerleri\n"
                        "- y_test.csv       : Test seti hedef değerleri\n"
                    ),
                )
            return buf.getvalue()

        st.download_button(
            " Modeling Artifact (ZIP) indir",
            data=_build_artifact_zip(),
            file_name=f"modeling_artifact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True,
            help=(
                "İçerik: metadata.json, features.json, model.pkl, "
                "X_train/X_test/y_train/y_test CSV'leri ve README."
            ),
        )

        st.info(
            "ℹ️ Artifact paketi yalnızca **yeniden üretilebilirlik ve inceleme** "
            "amaçlıdır. Model dosyası (.pkl) doğrudan canlı sisteme yüklenmemelidir; "
            "önce MLOps pipeline'ında ek validasyondan geçmelidir."
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

        st.markdown("#### Tahmin Önizlemesi (ilk 20 satır)")
        st.caption(
            "En iyi modelin tüm veri seti üzerindeki tahminleri. "
            "`Veri Seti Ayırımı` kolonu satırın train mı test mi olduğunu gösterir."
        )
        full_preds_df = _build_full_predictions(df, res)
        id_col = res.get("id_col")
        preview_cols = []
        if id_col and id_col in full_preds_df.columns:
            preview_cols.append(id_col)
        preview_cols.extend([
            "Satır No",
            "Veri Seti Ayırımı",
            res["target_col"],
            f"Tahmin ({res['best']})",
        ])
        preview_cols = [c for c in preview_cols if c in full_preds_df.columns]
        st.dataframe(
            full_preds_df[preview_cols].head(20),
            use_container_width=True,
            hide_index=True,
        )

        split_df = pd.DataFrame(
            {
                "Satır No": list(res["train_index"]) + list(res["test_index"]),
                "Ayırım": ["train"] * len(res["train_index"]) + ["test"] * len(res["test_index"]),
            }
        )
        if id_col and id_col in df.columns:
            split_df["Tanımlayıcı"] = df.loc[split_df["Satır No"], id_col].values

        sheets = {
            "Model Karsilastirma": comp_df,
            "Train-Test Ayirimi": split_df,
            "Tum Veri ve Tahminler": full_preds_df,
        }

        excel_bytes = build_excel_report(sheets)
        st.download_button(
            "Sonuçları Excel olarak indir",
            data=excel_bytes,
            file_name="smart_modeling_sonuclari.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
