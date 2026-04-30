import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(data, k_range=(2, 10)):
    k_min, k_max = k_range
    k_max = min(k_max, len(data) - 1)

    results = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        results.append({"k": k, "silhouette_score": score, "inertia": kmeans.inertia_})

    results_df = pd.DataFrame(results)
    best_k = results_df.loc[results_df["silhouette_score"].idxmax(), "k"]
    return int(best_k), results_df


def run_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    score = silhouette_score(data, labels)
    return labels, kmeans, score


def generate_cluster_profiles(original_df, selected_features, labels):
    df = original_df[selected_features].copy()
    df["Cluster"] = labels

    profiles = {}

    for cluster_id in sorted(df["Cluster"].unique()):
        cluster_data = df[df["Cluster"] == cluster_id]
        profile = {
            "size": len(cluster_data),
            "percentage": round(len(cluster_data) / len(df) * 100, 1),
            "stats": {},
            "distinguishing_features": [],
        }

        numeric_cols = cluster_data[selected_features].select_dtypes(include=[np.number]).columns
        categorical_cols = cluster_data[selected_features].select_dtypes(include=["object", "category"]).columns

        for col in numeric_cols:
            cluster_mean = cluster_data[col].mean()
            overall_mean = df[col].mean()
            overall_std = df[col].std()
            diff_pct = ((cluster_mean - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0

            profile["stats"][col] = {
                "mean": round(cluster_mean, 2),
                "median": round(cluster_data[col].median(), 2),
                "std": round(cluster_data[col].std(), 2),
                "overall_mean": round(overall_mean, 2),
                "diff_from_overall_pct": round(diff_pct, 1),
            }

            if overall_std > 0 and abs(cluster_mean - overall_mean) > 0.5 * overall_std:
                direction = "yuksek" if cluster_mean > overall_mean else "dusuk"
                profile["distinguishing_features"].append(
                    f"{col}: genel ortalamaya gore %{abs(diff_pct):.0f} {direction}"
                )

        for col in categorical_cols:
            top_values = cluster_data[col].value_counts().head(3)
            profile["stats"][col] = {
                "top_values": top_values.to_dict(),
                "dominant_value": top_values.index[0] if len(top_values) > 0 else None,
                "dominant_pct": round(top_values.iloc[0] / len(cluster_data) * 100, 1) if len(top_values) > 0 else 0,
            }

        profile["distinguishing_features"] = profile["distinguishing_features"][:5]
        profiles[cluster_id] = profile

    return profiles


def prepare_llm_summary(profiles, selected_features):
    summary_parts = []
    summary_parts.append(f"Analiz edilen ozellikler: {', '.join(selected_features)}")
    summary_parts.append(f"Toplam segment sayisi: {len(profiles)}")
    summary_parts.append("")

    for cluster_id, profile in profiles.items():
        part = f"--- Segment {cluster_id} ---\n"
        part += f"Buyukluk: {profile['size']} kayit (%{profile['percentage']})\n"

        if profile["distinguishing_features"]:
            part += "Ayirt edici ozellikler:\n"
            for feat in profile["distinguishing_features"]:
                part += f"  - {feat}\n"

        part += "Istatistikler:\n"
        for col, stats in profile["stats"].items():
            if "mean" in stats:
                part += f"  {col}: ortalama={stats['mean']}, genel ortalama={stats['overall_mean']}, fark=%{stats['diff_from_overall_pct']}\n"
            elif "top_values" in stats:
                part += f"  {col}: en sik={stats.get('dominant_value', 'N/A')} (%{stats.get('dominant_pct', 0)})\n"

        summary_parts.append(part)

    return "\n".join(summary_parts)
