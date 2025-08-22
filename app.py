import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.neighbors import BallTree
import pydeck as pdk

# ========== 配置页面 ==========
st.set_page_config(page_title="Wheat Yield App", layout="wide")

# ========== 读取资源 ==========
@st.cache_resource
def load_artifacts():
    model_global = joblib.load("model_global_xgb.joblib")
    specialists = joblib.load("model_specialists.joblib")
    feat_list = joblib.load("feature_list.joblib")
    weather_df = pd.read_csv("weather_ref.csv")
    return model_global, specialists, feat_list, weather_df

model_global, specialists, FEATS, WEATHER_REF = load_artifacts()

# ========== 工具函数 ==========
def nearest_cluster(lat, lon, year, weather_df, max_year_span=3):
    df = weather_df[weather_df["year"] == year]
    if df.empty:
        for span in range(1, max_year_span + 1):
            df = weather_df[(weather_df["year"] >= year - span) & (weather_df["year"] <= year + span)]
            if not df.empty:
                break
    if df.empty:
        return np.nan, None

    pts = np.radians(df[["cityLat", "cityLon"]].values)
    tree = BallTree(pts, metric="haversine")
    dist, idx = tree.query(np.radians([[lat, lon]]), k=1)
    j = int(idx[0, 0])
    row = df.iloc[j]
    return int(row["climate_cluster"]), row["city"]

def predict_hybrid(X_df: pd.DataFrame, cluster_vec: np.ndarray):
    yhat = model_global.predict(X_df[FEATS].values)
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

# ========== 页面侧边栏 ==========
with st.sidebar:
    st.header("模式选择")
    mode = st.radio("请选择:", ["单条预测", "批量预测", "气候城市地图"])
    st.markdown("字段: year, sown_area_hectare, cityLat, cityLon, climate_cluster")
    st.caption("若缺少cluster可自动匹配")

# ========== 单条预测 ==========
if mode == "单条预测":
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("年份", min_value=1990, max_value=2100, value=2018, step=1)
        sown_area = st.number_input("播种面积 (公顷)", min_value=0.0, value=1000.0, step=10.0, format="%.3f")
    with col2:
        lat = st.number_input("纬度 (cityLat)", min_value=-90.0, max_value=90.0, value=34.75, step=0.01)
        lon = st.number_input("经度 (cityLon)", min_value=-180.0, max_value=180.0, value=113.62, step=0.01)

    auto_cluster = st.checkbox("自动检测气候区", value=True)
    cluster = None
    nearest_city = None
    if auto_cluster:
        if st.button("检测气候区"):
            cluster, nearest_city = nearest_cluster(lat, lon, int(year), WEATHER_REF)
            if np.isnan(cluster):
                st.error("无匹配城市")
            else:
                st.success(f"检测结果: cluster={cluster}, 最近城市={nearest_city}")
    else:
        cluster = st.number_input("气候区 (cluster)", min_value=0, value=0, step=1)

    if st.button("开始预测"):
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
        st.subheader(f"预测产量: {yhat[0]:.3f} 吨/公顷")
        st.caption("特征向量:")
        st.dataframe(X_prepared)

# ========== 批量预测 ==========
elif mode == "批量预测":
    st.write("上传包含 year, sown_area_hectare, cityLat, cityLon, [climate_cluster] 的CSV")
    up = st.file_uploader("上传CSV文件", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        df_proc = df_in.copy()

        if "climate_cluster" not in df_proc.columns:
            df_proc["climate_cluster"] = np.nan

        miss = df_proc["climate_cluster"].isna()
        if miss.any():
            st.info(f"自动检测 {miss.sum()} 行缺失的cluster...")
            clusters = []
            for (i, r) in df_proc.loc[miss, ["cityLat","cityLon","year"]].iterrows():
                cc, _ = nearest_cluster(float(r["cityLat"]), float(r["cityLon"]), int(r["year"]), WEATHER_REF)
                clusters.append((i, cc))
            for i, cc in clusters:
                df_proc.at[i, "climate_cluster"] = cc

        X_prepared = ensure_feature_frame(df_proc)
        cluster_vec = df_proc["climate_cluster"].astype(int).values
        yhat = predict_hybrid(X_prepared, cluster_vec)

        out = df_in.copy()
        out["pred_yield_per_hectare"] = yhat
        st.success("预测完成")
        st.dataframe(out.head(20))

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("下载预测结果", data=csv, file_name="predictions.csv", mime="text/csv")

# ========== 地图聚类可视化 ==========
elif mode == "气候城市地图":
    st.subheader("Weather City Map (colored by climate cluster)")
    year_sel = st.slider("Filter by year", min_value=int(WEATHER_REF["year"].min()), max_value=int(WEATHER_REF["year"].max()), value=int(WEATHER_REF["year"].min()), step=1)
    df_year = WEATHER_REF[WEATHER_REF["year"] == year_sel].copy()

    if df_year.empty:
        st.warning("该年份无数据")
    else:
        df_year["cluster"] = df_year["climate_cluster"].astype(int)
        st.write("Clusters in selected year:", df_year["cluster"].value_counts())

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_year,
            get_position='[cityLon, cityLat]',
            get_fill_color='[cluster * 60 + 50, 100, 180]',
            get_radius=40000,
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            longitude=df_year["cityLon"].mean(),
            latitude=df_year["cityLat"].mean(),
            zoom=4,
            pitch=0
        )

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"text": "City: {city}\nCluster: {cluster}"}
        ))
