import streamlit as st
import pandas as pd

from data_processor import load_data, analyze_columns, recommend_features, preprocess_data
from clustering_engine import find_optimal_k, run_clustering, generate_cluster_profiles, prepare_llm_summary
from llm_interpreter import interpret_segments
from visualizer import (
    plot_silhouette_scores,
    plot_elbow,
    plot_2d_clusters,
    plot_cluster_sizes,
    plot_feature_comparison,
    plot_radar_chart,
)
from exporter import export_to_excel, export_to_json


st.set_page_config(
    page_title="Segment Intelligence Agent",
    page_icon="🧠",
    layout="wide",
)

st.title("Segment Intelligence Agent")
st.caption("Verinizi yükleyin, segmentleri keşfedelim, yapay zekâ ile yorumlayalım.")

DEFAULT_STATE = {
    "df": None,
    "analysis": None,
    "selected_features": [],
    "clustering_data": None,
    "scaled_df": None,
    "cluster_results_df": None,
    "best_k": None,
    "labels": None,
    "sil_score": None,
    "profiles": None,
    "llm_result": None,
    "segmentation_done": False,
    "_file_signature": None,
}

for _key, _value in DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


st.header("1. Veri Yükleme")

uploaded_file = st.file_uploader("CSV veya Excel dosyanızı yükleyin", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    file_signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state._file_signature != file_signature:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.session_state._file_signature = file_signature
            st.session_state.segmentation_done = False
            st.session_state.llm_result = None
            st.session_state.profiles = None
            st.session_state.labels = None
            st.session_state.clustering_data = None
            st.success(f"Veri başarıyla yüklendi: {df.shape[0]} satır, {df.shape[1]} kolon")
        except Exception as e:
            st.error(f"Veri yüklenirken hata oluştu: {e}")

    if st.session_state.df is not None:
        df_preview = st.session_state.df
        with st.expander("Veri Önizleme", expanded=False):
            st.dataframe(df_preview.head(20), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Veri Tipleri:**")
                st.dataframe(df_preview.dtypes.reset_index().rename(columns={"index": "Kolon", 0: "Tip"}))
            with col2:
                st.write("**Eksik Değer Özeti:**")
                missing = df_preview.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) > 0:
                    st.dataframe(missing.reset_index().rename(columns={"index": "Kolon", 0: "Eksik Sayı"}))
                else:
                    st.info("Eksik değer yok.")


if st.session_state.df is not None:
    st.header("2. Özellik Seçimi")

    df = st.session_state.df
    analysis = analyze_columns(df)
    st.session_state.analysis = analysis

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sayısal Kolonlar:**", ", ".join(analysis["numeric"]) if analysis["numeric"] else "Yok")
        st.write("**Kategorik Kolonlar:**", ", ".join(analysis["categorical"]) if analysis["categorical"] else "Yok")
    with col2:
        st.write("**Tarih Alanları:**", ", ".join(analysis["datetime"]) if analysis["datetime"] else "Yok")
        st.write("**Elenen Kolonlar (ID/Anlamsız):**", ", ".join(analysis["id_or_useless"]) if analysis["id_or_useless"] else "Yok")

    recommended, _ = recommend_features(analysis)

    if not recommended:
        st.warning("Analiz için uygun kolon bulunamadı. Lütfen verinizdeki kolonları kontrol edin.")
        st.stop()

    selected = st.multiselect(
        "Analize dahil edilecek özellikleri seçin:",
        options=list(df.columns),
        default=recommended,
        help="Sistem otomatik olarak uygun kolonları önermiştir. İsterseniz düzenleyebilirsiniz.",
        key="feature_select",
    )
    st.session_state.selected_features = selected

    if selected:
        st.header("3. Ön İşleme ve Kümeleme")

        col1, col2 = st.columns(2)
        with col1:
            apply_pca = st.checkbox("PCA (Boyut İndirgeme) Uygula", value=False, key="apply_pca")
            pca_components = 2
            if apply_pca:
                max_comp = min(len(selected), 10)
                pca_components = st.slider("PCA Bileşen Sayısı", 2, max_comp, 2, key="pca_components")

        with col2:
            k_min = st.number_input("Minimum Küme Sayısı", min_value=2, max_value=20, value=2, key="k_min")
            k_max = st.number_input(
                "Maksimum Küme Sayısı",
                min_value=2,
                max_value=20,
                value=min(10, len(df) - 1),
                key="k_max",
            )

        if st.button("Segmentasyonu Başlat", type="primary", use_container_width=True):
            try:
                with st.spinner("Veri işleniyor..."):
                    clustering_data, scaled_df, _, _, _ = preprocess_data(
                        df, selected, apply_pca, pca_components
                    )
                    st.session_state.clustering_data = clustering_data
                    st.session_state.scaled_df = scaled_df

                with st.spinner("Optimal küme sayısı hesaplanıyor..."):
                    best_k, results_df = find_optimal_k(clustering_data, k_range=(int(k_min), int(k_max)))
                    st.session_state.cluster_results_df = results_df
                    st.session_state.best_k = int(best_k)

                with st.spinner("Kümeleme çalışıyor..."):
                    labels, _, sil_score = run_clustering(clustering_data, int(best_k))
                    st.session_state.labels = labels
                    st.session_state.sil_score = float(sil_score)

                with st.spinner("Segment profilleri oluşturuluyor..."):
                    profiles = generate_cluster_profiles(df, selected, labels)
                    st.session_state.profiles = profiles

                st.session_state.segmentation_done = True
                st.session_state.llm_result = None
            except Exception as e:
                st.session_state.segmentation_done = False
                st.error(f"Segmentasyon sırasında hata oluştu: {e}")

        if st.session_state.segmentation_done and st.session_state.profiles is not None:
            results_df = st.session_state.cluster_results_df
            profiles = st.session_state.profiles
            labels = st.session_state.labels
            clustering_data = st.session_state.clustering_data

            st.subheader("Optimal Küme Analizi")
            tab1, tab2 = st.tabs(["Silhouette Skoru", "Dirsek Yöntemi"])
            with tab1:
                st.plotly_chart(plot_silhouette_scores(results_df), use_container_width=True)
            with tab2:
                st.plotly_chart(plot_elbow(results_df), use_container_width=True)

            st.success(
                f"Kümeleme tamamlandı! Önerilen k={st.session_state.best_k} — "
                f"Silhouette Skoru: {st.session_state.sil_score:.3f}"
            )

            st.header("4. Segment Analizi")

            st.plotly_chart(plot_cluster_sizes(profiles), use_container_width=True)
            st.plotly_chart(
                plot_2d_clusters(clustering_data, labels, selected),
                use_container_width=True,
            )

            numeric_feats = [f for f in selected if pd.api.types.is_numeric_dtype(df[f])]
            if len(numeric_feats) >= 3:
                radar = plot_radar_chart(profiles, numeric_feats)
                if radar:
                    st.plotly_chart(radar, use_container_width=True)

            st.subheader("Özellik Karşılaştırması")
            if numeric_feats:
                feat_to_compare = st.selectbox("Özellik seçin:", numeric_feats, key="feat_compare")
                fig = plot_feature_comparison(profiles, feat_to_compare)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("Detaylı Segment Profilleri"):
                for cid, p in profiles.items():
                    st.markdown(f"**Segment {cid}** — {p['size']} kayıt (%{p['percentage']})")
                    if p["distinguishing_features"]:
                        for feat in p["distinguishing_features"]:
                            st.markdown(f"- {feat}")
                    st.divider()

            st.header("5. Yapay Zekâ Yorumlama")

            context = st.text_area(
                "Ek bağlamsal bilgi (isteğe bağlı):",
                placeholder="Örneğin: Bu veri seti banka bireysel müşterilerine aittir. Amacımız kampanya hedeflemesi yapmaktır.",
                help="Yapay zekânın daha isabetli yorumlar üretmesi için veri seti hakkında ek bilgi verebilirsiniz.",
                key="llm_context",
            )

            if st.button("Yapay Zekâ ile Yorumla", type="primary", use_container_width=True):
                try:
                    llm_summary = prepare_llm_summary(profiles, selected)
                    with st.spinner("Yapay zekâ segmentleri yorumluyor..."):
                        st.session_state.llm_result = interpret_segments(llm_summary, context)
                except Exception as e:
                    st.session_state.llm_result = None
                    st.error(f"Yapay zekâ yorumlama sırasında hata oluştu: {e}")

            if st.session_state.llm_result is not None:
                llm_result = st.session_state.llm_result

                if "raw_response" in llm_result:
                    st.markdown(llm_result["raw_response"])
                else:
                    if llm_result.get("executive_summary"):
                        st.subheader("Yönetici Özeti")
                        st.info(llm_result["executive_summary"])

                    if llm_result.get("cross_segment_insights"):
                        st.subheader("Segmentler Arası İçgörüler")
                        for insight in llm_result["cross_segment_insights"]:
                            st.markdown(f"- {insight}")

                    st.subheader("Segment Detayları")
                    for seg in llm_result.get("segments", []):
                        title = seg.get("name") or f"Segment {seg.get('id', '?')}"
                        with st.expander(f"🏷️ {title}"):
                            st.markdown(f"**Profil:** {seg.get('profile', '')}")
                            st.markdown(f"**Davranış Analizi:** {seg.get('behavioral_analysis', '')}")

                            if seg.get("key_insights"):
                                st.markdown("**İçgörüler:**")
                                for ins in seg["key_insights"]:
                                    st.markdown(f"- {ins}")

                            if seg.get("recommended_actions"):
                                st.markdown("**Önerilen Aksiyonlar:**")
                                for act in seg["recommended_actions"]:
                                    st.markdown(f"- ✅ {act}")

                            if seg.get("risk_notes"):
                                st.markdown("**Risk / Dikkat Noktaları:**")
                                for risk in seg["risk_notes"]:
                                    st.markdown(f"- ⚠️ {risk}")

                st.header("6. Geri Bildirim")
                st.slider(
                    "Yorumların kalitesini puanlayın:",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="1: Çok kötü, 5: Mükemmel",
                    key="feedback_score",
                )
                st.text_area("Ek yorumunuz (isteğe bağlı):", key="feedback_text")
                if st.button("Geri Bildirim Gönder"):
                    st.success("Geri bildiriminiz kaydedildi. Teşekkürler!")

            st.header("7. Rapor Dışa Aktarma")

            llm_res = st.session_state.llm_result or {}
            col1, col2 = st.columns(2)
            with col1:
                excel_data = export_to_excel(df, labels, profiles, llm_res, selected)
                st.download_button(
                    label="📥 Excel Raporu İndir",
                    data=excel_data,
                    file_name="segment_analiz_raporu.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            with col2:
                json_data = export_to_json(profiles, llm_res)
                st.download_button(
                    label="📥 JSON Raporu İndir",
                    data=json_data,
                    file_name="segment_analiz_raporu.json",
                    mime="application/json",
                    use_container_width=True,
                )
