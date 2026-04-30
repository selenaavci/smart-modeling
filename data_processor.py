import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Desteklenen formatlar: CSV, XLSX, XLS")


def analyze_columns(df):
    analysis = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "id_or_useless": [],
    }

    for col in df.columns:
        if col.lower() in ("id", "index", "row", "unnamed: 0") or col.lower().endswith("_id"):
            analysis["id_or_useless"].append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            analysis["datetime"].append(col)
            continue

        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col].dropna().head(20))
                analysis["datetime"].append(col)
                continue
            except (ValueError, TypeError):
                pass

        if pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique()
            if nunique == len(df) and df[col].is_monotonic_increasing:
                analysis["id_or_useless"].append(col)
            else:
                analysis["numeric"].append(col)
            continue

        if df[col].dtype == object or pd.api.types.is_categorical_dtype(df[col]):
            nunique = df[col].nunique()
            if nunique > 50 or nunique == len(df):
                analysis["id_or_useless"].append(col)
            else:
                analysis["categorical"].append(col)
            continue

        analysis["id_or_useless"].append(col)

    return analysis


def recommend_features(analysis):
    recommended = analysis["numeric"] + analysis["categorical"]
    excluded = analysis["id_or_useless"] + analysis["datetime"]
    return recommended, excluded


def preprocess_data(df, selected_features, apply_pca=False, pca_components=2):
    work_df = df[selected_features].copy()

    for col in work_df.columns:
        if pd.api.types.is_numeric_dtype(work_df[col]):
            work_df[col] = work_df[col].fillna(work_df[col].median())
        else:
            mode = work_df[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            work_df[col] = work_df[col].fillna(fill_value)

    label_encoders = {}
    categorical_cols = work_df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        work_df[col] = le.fit_transform(work_df[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(work_df)
    scaled_df = pd.DataFrame(scaled_data, columns=work_df.columns, index=work_df.index)

    pca_model = None
    if apply_pca and scaled_data.shape[1] > pca_components:
        pca_model = PCA(n_components=pca_components)
        pca_data = pca_model.fit_transform(scaled_data)
        pca_df = pd.DataFrame(
            pca_data,
            columns=[f"PC{i+1}" for i in range(pca_components)],
            index=work_df.index,
        )
        return pca_df, scaled_df, scaler, label_encoders, pca_model

    return scaled_df, scaled_df, scaler, label_encoders, pca_model
