import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA


def plot_silhouette_scores(results_df):
    fig = px.line(
        results_df,
        x="k",
        y="silhouette_score",
        markers=True,
        title="Optimal Küme Sayısı (Silhouette Skoru)",
        labels={"k": "Küme Sayısı (k)", "silhouette_score": "Silhouette Skoru"},
    )
    best_k = results_df.loc[results_df["silhouette_score"].idxmax()]
    fig.add_annotation(
        x=best_k["k"],
        y=best_k["silhouette_score"],
        text=f"Önerilen: k={int(best_k['k'])}",
        showarrow=True,
        arrowhead=2,
        font=dict(color="red", size=13),
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_elbow(results_df):
    fig = px.line(
        results_df,
        x="k",
        y="inertia",
        markers=True,
        title="Dirsek Yöntemi (Inertia)",
        labels={"k": "Küme Sayısı (k)", "inertia": "Inertia"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_2d_clusters(data_df, labels, selected_features):
    plot_df = data_df.copy()

    if data_df.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(data_df)
        plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=data_df.index)
        x_col, y_col = "PC1", "PC2"
        explained = pca.explained_variance_ratio_
        title = f"Segment Dağılımı (PCA — Açıklanan Varyans: %{sum(explained)*100:.1f})"
    else:
        cols = list(data_df.columns[:2])
        x_col, y_col = cols[0], cols[1]
        title = "Segment Dağılımı"

    plot_df["Segment"] = [f"Segment {l}" for l in labels]

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="Segment",
        title=title,
        opacity=0.7,
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_cluster_sizes(profiles):
    data = []
    for cid, p in profiles.items():
        data.append({"Segment": f"Segment {cid}", "Kayıt Sayısı": p["size"], "Yüzde": p["percentage"]})

    df = pd.DataFrame(data)
    fig = px.bar(
        df,
        x="Segment",
        y="Kayıt Sayısı",
        text="Yüzde",
        title="Segment Büyüklükleri",
        color="Segment",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_feature_comparison(profiles, feature_name):
    data = []
    for cid, p in profiles.items():
        stats = p["stats"].get(feature_name, {})
        if "mean" in stats:
            data.append({
                "Segment": f"Segment {cid}",
                "Ortalama": stats["mean"],
                "Genel Ortalama": stats["overall_mean"],
            })

    if not data:
        return None

    df = pd.DataFrame(data)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Segment Ortalaması", x=df["Segment"], y=df["Ortalama"]))
    fig.add_trace(go.Bar(name="Genel Ortalama", x=df["Segment"], y=df["Genel Ortalama"]))
    fig.update_layout(
        barmode="group",
        title=f"{feature_name} — Segment Karşılaştırması",
        template="plotly_white",
    )
    return fig


def plot_radar_chart(profiles, numeric_features):
    if len(numeric_features) < 3:
        return None

    fig = go.Figure()
    features_to_show = numeric_features[:8]

    for cid, p in profiles.items():
        values = []
        for feat in features_to_show:
            stats = p["stats"].get(feat, {})
            values.append(stats.get("diff_from_overall_pct", 0))
        values.append(values[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=features_to_show + [features_to_show[0]],
            fill="toself",
            name=f"Segment {cid}",
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, title="Genel Ort. Fark %")),
        title="Segment Profil Karşılaştırması (Radar)",
        template="plotly_white",
    )
    return fig
