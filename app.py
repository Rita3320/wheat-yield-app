import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import BallTree
import random
import plotly.express as px

# -------- App Settings --------
st.set_page_config(layout="wide")
st.title("Wheat Yield Prediction App")

# -------- Load models and data --------
@st.cache_resource
def load_artifacts():
    global_model = joblib.load("model_global_xgb.joblib")
    specialists = joblib.load("model_specialists.joblib")
    feats = joblib.load("feature_list.joblib")
    weather_df = pd.read_csv("weather_ref.csv")
    return global_model, specialists, feats, weather_df

model_global, specialists, FEATS, WEATHER_REF = load_artifacts()

# -------- Utility Functions --------
def nearest_cluster(lat, lon, year, df, max_span=3):
    df_y = df[df["year"] == year]
    if df_y.empty:
        for s in range(1, max_span+1):
            df_y = df[(df["year"] >= year-s) & (df["year"] <= year+s)]
            if not df_y.empty:
                break
    if df_y.empty:
        return np.nan, None
    tree = BallTree(np.radians(df_y[["cityLat", "cityLon"]].values), metric="haversine")
    dist, idx = tree.query(np.radians([[lat, lon]]), k=1)
    row = df_y.iloc[idx[0][0]]
    return int(row["climate_cluster"]), row["city"]

def ensure_feature_frame(df_like):
    X = df_like.copy()
    for c in FEATS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEATS]
    return X.astype(float).fillna(X.median(numeric_only=True))

def predict_hybrid(X_df, cluster_vec):
    yhat = model_global.predict(X_df[FEATS])
    for cid, model in specialists.items():
        mask = (cluster_vec == cid)
        if np.any(mask):
            yhat[mask] = model.predict(X_df.loc[mask, FEATS])
    return yhat

# -------- Random Insights --------
INSIGHTS = {
    0: "Cluster 0 has stable temperatures and low wind — yield is mainly climate-driven.",
    1: "Cluster 1 has high humidity — diseases may impact yields.",
    2: "Cluster 2 experiences continental climate swings — large yield variation.",
    3: "Cluster 3 is dry — irrigation is critical to yield."
}

def random_insight(cluster_id):
    return INSIGHTS.get(cluster_id, "")

# -------- Cluster Map --------
st.subheader("Climate Cluster Map")
fig = px.scatter_mapbox(
    WEATHER_REF,
    lat="cityLat", lon="cityLon",
    color="climate_cluster",
    color_continuous_scale="Viridis",
    hover_data=["climate_cluster"],
    zoom=3, height=400
)
fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

# -------- Prediction Mode --------
mode = st.sidebar.radio("Select Mode", ["Single Prediction", "Batch Prediction (CSV)"])

if mode == "Single Prediction":
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", 2000, 2100, 2018)
        sown_area = st.number_input("Sown Area (ha)", 0.0, 1e6, 1000.0)
    with col2:
        lat = st.number_input("Latitude", -90.0, 90.0, 35.0)
        lon = st.number_input("Longitude", -180.0, 180.0, 110.0)

    if st.button("Predict"):
        cluster, city = nearest_cluster(lat, lon, year, WEATHER_REF)
        if pd.isna(cluster):
            st.error("No cluster found for these coordinates.")
        else:
            st.success(f"Cluster {cluster} (nearest city: {city})")
            X_row = pd.DataFrame([{
                "year": year,
                "sown_area_hectare": sown_area,
                "cityLat": lat,
                "cityLon": lon,
                "climate_cluster": cluster
            }])
            X_prepared = ensure_feature_frame(X_row)
            yhat = predict_hybrid(X_prepared, np.array([cluster]))
            st.metric(label="Predicted Yield", value=f"{yhat[0]:.3f} tons/ha")
            st.caption("Feature Vector Used")
            st.dataframe(X_prepared)
            st.info(f"Insight: {random_insight(cluster)}")

else:
    st.download_button("Download CSV Template",
        data=pd.DataFrame(columns=["year", "sown_area_hectare", "cityLat", "cityLon"]).to_csv(index=False),
        file_name="template.csv")

    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        df["climate_cluster"] = df.apply(
            lambda r: nearest_cluster(r["cityLat"], r["cityLon"], r["year"], WEATHER_REF)[0]
            if pd.isna(r.get("climate_cluster", np.nan)) else int(r["climate_cluster"]), axis=1)
        cluster_vec = df["climate_cluster"].values.astype(int)
        X_prep = ensure_feature_frame(df)
        yhat = predict_hybrid(X_prep, cluster_vec)
        df["predicted_yield"] = yhat
        st.success("Prediction complete.")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", data=csv, file_name="prediction_results.csv", mime="text/csv")
