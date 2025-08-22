# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import BallTree
import pydeck as pdk
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score

# ======= 0. 配置页面 =======
st.set_page_config(page_title="Wheat Yield Predictor", layout="wide")
st.title("Wheat Yield Prediction App")

# ======= 1. 加载资源 =======
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_global_xgb.joblib")
    specialists = joblib.load("model_specialists.joblib")
    feats = joblib.load("feature_list.joblib")
    weather_df = pd.read_csv("weather_ref.csv")
    return model, specialists, feats, weather_df

global_model, specialists, FEATS, WEATHER_REF = load_artifacts()

# ======= 2. 工具函数 =======
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
    row = df.iloc[int(idx[0, 0])]
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

# ======= 3. 页面导航栏 =======
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to:", ["Prediction", "Map", "Cluster Stats", "Charts"])

# ======= 4. 页面：预测 =======
if page == "Prediction":
    st.subheader("Yield Prediction")
    mode = st.radio("Mode", ["Single", "Batch (CSV)"])

    if mode == "Single":
        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Year", 1990, 2100, 2020)
            area = st.number_input("Sown Area (ha)", 0.0, 1e6, 1000.0)
        with c2:
            lat = st.number_input("Latitude", -90.0, 90.0, 35.0)
            lon = st.number_input("Longitude", -180.0, 180.0, 115.0)

        auto_cluster = st.checkbox("Auto detect cluster", value=True)
        cluster = st.number_input("Cluster", 0, 9, 0) if not auto_cluster else None

        if st.button("Predict"):
            if auto_cluster:
                cluster, nearest_city = nearest_cluster(lat, lon, year, WEATHER_REF)
                st.info(f"Auto-detected cluster {cluster} (nearest: {nearest_city})")

            X_row = pd.DataFrame([{"year": year, "sown_area_hectare": area, "cityLat": lat, "cityLon": lon, "climate_cluster": cluster}])
            X_prepped = ensure_feature_frame(X_row)
            yhat = predict_hybrid(X_prepped, np.array([cluster]))
            st.success(f"Predicted yield: {yhat[0]:.2f} tons/ha")
            st.caption("Feature input:")
            st.dataframe(X_prepped)

    else:
        st.write("Upload a CSV with columns: year, sown_area_hectare, cityLat, cityLon, [optional] climate_cluster")
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            if "climate_cluster" not in df:
                df["climate_cluster"] = [nearest_cluster(r["cityLat"], r["cityLon"], r["year"], WEATHER_REF)[0] for _, r in df.iterrows()]
            X = ensure_feature_frame(df)
            yhat = predict_hybrid(X, df["climate_cluster"].astype(int).values)
            df["yield_pred"] = yhat
            st.dataframe(df.head(20))
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, file_name="predicted_yield.csv")

# ======= 5. 页面：地图 =======
elif page == "Map":
    st.subheader("Weather City Map (colored by climate cluster)")
    df_map = WEATHER_REF.copy()
    year_sel = st.slider("Filter by year", int(df_map.year.min()), int(df_map.year.max()), int(df_map.year.max()))
    df_map = df_map[df_map.year == year_sel]

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[cityLon, cityLat]',
        get_color="[climate_cluster * 50, 100, 150]",
        get_radius=8000,
        pickable=True
    )
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=df_map.cityLat.mean(),
            longitude=df_map.cityLon.mean(),
            zoom=4,
            pitch=0
        ),
        layers=[layer],
        tooltip={"text": "City: {city}\nCluster: {climate_cluster}"}
    ))

# ======= 6. 页面：聚类数量统计 =======
elif page == "Cluster Stats":
    st.subheader("Climate Cluster Distribution")
    cluster_counts = WEATHER_REF.groupby("climate_cluster").city.nunique().reset_index(name="city_count")
    fig = px.bar(cluster_counts, x="climate_cluster", y="city_count", color="climate_cluster", title="City count per cluster")
    st.plotly_chart(fig)

# ======= 7. 页面：图像上传与展示 =======
elif page == "Charts":
    st.subheader("Upload Charts or Screenshots")
    img_file = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"])
    if img_file is not None:
        st.image(img_file, use_column_width=True)
        st.caption("Click right-click > Save image if needed")
