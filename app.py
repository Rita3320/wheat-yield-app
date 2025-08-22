import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import BallTree
from sklearn.metrics import mean_squared_error, r2_score
import pydeck as pdk
import plotly.express as px

st.set_page_config(page_title="Wheat Yield Predictor", layout="wide")

# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state["history"] = []  # Store prediction history

# ---------- Load models and data ----------
@st.cache_resource
def load_artifacts():
    global_model = joblib.load("model_global_xgb.joblib")
    specialists = joblib.load("model_specialists.joblib")
    feat_list = joblib.load("feature_list.joblib")
    weather_ref = pd.read_csv("weather_ref.csv")
    return global_model, specialists, feat_list, weather_ref

global_model, specialists, FEATS, WEATHER_REF = load_artifacts()

# ---------- Utility functions ----------
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

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Mode Selection")
    mode = st.radio("Choose prediction mode:", ["Single prediction", "Batch prediction"])
    st.caption("Input required: year, sown_area_hectare, cityLat, cityLon. Cluster optional.")

    # History in sidebar
    st.markdown("---")
    with st.expander("Prediction History", expanded=False):
        if len(st.session_state["history"]) == 0:
            st.info("No predictions yet.")
        else:
            hist_df = pd.DataFrame(st.session_state["history"])
            st.dataframe(hist_df)
            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download History", data=csv_data, file_name="prediction_history.csv")
            if st.button("Clear History"):
                st.session_state["history"].clear()
                st.success("History cleared.")

# ---------- Cluster Map ----------
st.subheader("Climate Cluster Map")
map_df = WEATHER_REF[["cityLat", "cityLon", "climate_cluster"]].dropna()
map_df["cluster"] = map_df["climate_cluster"].astype(str)
fig_map = px.scatter_mapbox(
    map_df,
    lat="cityLat",
    lon="cityLon",
    color="cluster",
    zoom=2,
    height=400,
    mapbox_style="carto-positron",
    title="Reference Weather Cities by Cluster"
)
st.plotly_chart(fig_map, use_container_width=True)

# ---------- Main Prediction Interface ----------
if mode == "Single prediction":
    st.header("Single Point Prediction")
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=1990, max_value=2100, value=2018)
        sown_area = st.number_input("Sown area (ha)", min_value=0.0, value=1000.0, step=10.0)
    with col2:
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.75)
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=113.62)

    auto_cluster = st.checkbox("Auto-detect cluster from coordinates", value=True)
    cluster = None
    nearest_city = None
    if auto_cluster:
        if st.button("Detect cluster"):
            cluster, nearest_city = nearest_cluster(lat, lon, int(year), WEATHER_REF)
            if np.isnan(cluster):
                st.error("No matching cluster found.")
            else:
                st.success(f"Cluster {cluster} (nearest city: {nearest_city})")
    else:
        cluster = st.number_input("Enter climate cluster", min_value=0, value=0, step=1)

    if st.button("Predict"):
        if cluster is None or pd.isna(cluster):
            cluster, nearest_city = nearest_cluster(lat, lon, int(year), WEATHER_REF)

        input_row = pd.DataFrame([{
            "year": year,
            "sown_area_hectare": sown_area,
            "cityLat": lat,
            "cityLon": lon,
            "climate_cluster": int(cluster) if cluster is not None else np.nan
        }])
        X_prepared = ensure_feature_frame(input_row)
        yhat = predict_hybrid(X_prepared, np.array([int(cluster)]))

        st.subheader(f"Predicted Yield: {yhat[0]:.3f} tons/ha")
        st.caption("Feature vector used:")
        st.dataframe(X_prepared)

        # Save to history
        st.session_state["history"].append({
            "year": year,
            "sown_area_hectare": sown_area,
            "cityLat": lat,
            "cityLon": lon,
            "climate_cluster": cluster,
            "predicted_yield": round(yhat[0], 3)
        })

elif mode == "Batch prediction":
    st.header("Batch Prediction Upload")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
        df_proc = df_in.copy()

        if "climate_cluster" not in df_proc.columns:
            df_proc["climate_cluster"] = np.nan

        # Auto cluster assignment
        miss = df_proc["climate_cluster"].isna()
        if miss.any():
            st.info(f"Assigning clusters to {miss.sum()} rows...")
            for i, r in df_proc.loc[miss, ["cityLat", "cityLon", "year"]].iterrows():
                cc, _ = nearest_cluster(r["cityLat"], r["cityLon"], int(r["year"]), WEATHER_REF)
                df_proc.at[i, "climate_cluster"] = cc

        X_prepared = ensure_feature_frame(df_proc)
        cluster_vec = df_proc["climate_cluster"].astype(int).values
        yhat = predict_hybrid(X_prepared, cluster_vec)

        df_out = df_in.copy()
        df_out["predicted_yield"] = yhat
        st.success("Batch prediction completed.")
        st.dataframe(df_out.head())

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", data=csv, file_name="batch_results.csv")
