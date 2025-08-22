
# app.py (简化展示)
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.neighbors import BallTree
from sklearn.metrics import mean_squared_error, r2_score
import random

st.set_page_config(page_title="Wheat Yield Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    global_model = joblib.load("model_global_xgb.joblib")
    specialists = joblib.load("model_specialists.joblib")
    feat_list = joblib.load("feature_list.joblib")
    weather_ref = pd.read_csv("weather_ref.csv")
    return global_model, specialists, feat_list, weather_ref

global_model, specialists, FEATS, WEATHER_REF = load_artifacts()

def nearest_cluster(lat, lon, year, weather_df, max_year_span=3):
    df = weather_df[weather_df["year"] == year]
    if df.empty:
        for span in range(1, max_year_span + 1):
            df = weather_df[(weather_df["year"] >= year - span) & (weather_df["year"] <= year + span)]
            if not df.empty:
                break
    if df.empty:
        return np.nan, None

    pts = np.radians(df[["cityLat","cityLon"]].values)
    tree = BallTree(pts, metric="haversine")
    dist, idx = tree.query(np.radians([[lat, lon]]), k=1)
    j = int(idx[0, 0])
    row = df.iloc[j]
    return int(row["climate_cluster"]), row["city"]

def predict_hybrid(X_df: pd.DataFrame, cluster_vec: np.ndarray):
    yhat = global_model.predict(X_df[FEATS].values)
    for cid, model in specialists.items():
        mask = (cluster_vec == cid)
        if np.any(mask):
            yhat[mask] = model.predict(X_df.loc[mask, FEATS].values)
    return yhat

def ensure_feature_frame(df_like: pd.DataFrame) -> pd.DataFrame:
    X = df_like.copy()
    for c in FEATS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEATS]
    return X.astype(float).fillna(X.median(numeric_only=True))

def generate_insight(feature_row: dict, prediction: float, cluster_id: int, lang: str = "English"):
    fields = []
    if feature_row.get("sown_area_hectare", 0) > 500:
        fields.append(("large sown area", "播种面积较大"))
    if feature_row.get("cityLat", 0) > 35:
        fields.append(("higher latitude", "较高纬度"))
    if feature_row.get("cityLon", 0) > 110:
        fields.append(("eastern region", "偏东地区"))
    if cluster_id == 2:
        fields.append(("optimal climate", "适宜气候"))
    selected = random.sample(fields, min(len(fields), 3))
    if lang == "中文":
        feature_phrase = "、".join([cn for _, cn in selected])
        return f"洞察：基于输入特征（如 {feature_phrase}），模型预测该地区（气候区 {cluster_id}）的小麦单产为 {prediction:.2f} 吨/公顷。"
    else:
        feature_phrase = ", ".join([en for en, _ in selected])
        return f"Insight: Based on features like {feature_phrase}, the model predicts a yield of {prediction:.2f} tons per hectare for Cluster {cluster_id}."

# ------------------ UI ------------------
st.title("Wheat Yield per Hectare — Hybrid (Global + Specialists)")
with st.sidebar:
    st.header("Prediction Mode")
    mode = st.radio("Select", ["Single prediction", "Batch prediction (CSV)"])
    lang = st.radio("Language / 语言", ["English", "中文"], index=0)
    st.markdown("Expected features: **year, sown_area_hectare, cityLat, cityLon, climate_cluster**")
    st.caption("If climate_cluster is not provided, the app can auto-detect by nearest weather city.")

if mode == "Single prediction":
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=1990, max_value=2100, value=2018, step=1)
        sown_area = st.number_input("Sown area (hectare)", min_value=0.0, value=1000.0, step=10.0, format="%.3f")
    with col2:
        lat = st.number_input("Latitude (cityLat)", min_value=-90.0, max_value=90.0, value=34.75, step=0.01, format="%.6f")
        lon = st.number_input("Longitude (cityLon)", min_value=-180.0, max_value=180.0, value=113.62, step=0.01, format="%.6f")
    auto_cluster = st.checkbox("Auto-detect climate cluster", value=True)
    cluster = None
    nearest_city = None
    if auto_cluster:
        if st.button("Detect cluster"):
            cluster, nearest_city = nearest_cluster(lat, lon, int(year), WEATHER_REF)
            if np.isnan(cluster):
                st.error("No weather reference found for this year/coords.")
            else:
                st.success(f"Detected cluster: {cluster} (nearest weather city: {nearest_city})")
    else:
        cluster = st.number_input("Climate cluster", min_value=0, value=0, step=1)
    if st.button("Predict"):
        if cluster is None or pd.isna(cluster):
            cluster, nearest_city = nearest_cluster(lat, lon, int(year), WEATHER_REF)
        X_row = pd.DataFrame([{
            "year": year,
            "sown_area_hectare": sown_area,
            "cityLat": lat,
            "cityLon": lon,
            "climate_cluster": int(cluster) if cluster is not None else np.nan
        }])
        X_prepared = ensure_feature_frame(X_row)
        yhat = predict_hybrid(X_prepared, np.array([int(cluster)]))
        st.subheader(f"Predicted yield per hectare: {yhat[0]:.3f}")
        insight = generate_insight({
            "sown_area_hectare": sown_area,
            "cityLat": lat,
            "cityLon": lon
        }, prediction=yhat[0], cluster_id=int(cluster), lang=lang)
        st.info(insight)
