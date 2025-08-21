# app.py
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.neighbors import BallTree
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_folium import st_folium
import folium

try:
    import shap
    from streamlit_shap import st_shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


st.set_page_config(page_title="Wheat Yield Predictor Pro", layout="wide")

# -------------------- 加载工件 --------------------
@st.cache_resource
def load_artifacts():
    global_model = joblib.load("model_global_xgb.joblib")
    specialists = joblib.load("model_specialists.joblib")  # {cluster: model}
    feat_list   = joblib.load("feature_list.joblib")
    weather_ref = pd.read_csv("weather_ref.csv")
    # 可选：城市坐标、残差分布
    try:
        city_coords = pd.read_csv("city_coords.csv")
    except Exception:
        city_coords = None
    try:
        residual_stats = joblib.load("residual_stats.joblib")  # {cluster: np.array([q2.5, q16, q50, q84, q97.5])}
    except Exception:
        residual_stats = None
    return global_model, specialists, feat_list, weather_ref, city_coords, residual_stats

global_model, specialists, FEATS, WEATHER_REF, CITY_COORDS, RESIDUAL = load_artifacts()

# -------------------- 工具函数 --------------------
def ensure_feature_frame(df_like: pd.DataFrame) -> pd.DataFrame:
    X = df_like.copy()
    for c in FEATS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEATS]
    return X.astype(float).fillna(X.median(numeric_only=True))

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

def predict_global(X_df: pd.DataFrame):
    return global_model.predict(X_df[FEATS].values)

def predict_hybrid(X_df: pd.DataFrame, cluster_vec: np.ndarray):
    yhat = global_model.predict(X_df[FEATS].values)
    for cid, model in specialists.items():
        mask = (cluster_vec == cid)
        if np.any(mask):
            yhat[mask] = model.predict(X_df.loc[mask, FEATS].values)
    return yhat

def predict_interval(yhat: np.ndarray, cluster_vec: np.ndarray, residual_stats):
    """基于 hold-out 残差分布估计预测区间（简单相加）。"""
    if residual_stats is None:
        return None
    lo95, lo68, med, hi68, hi95 = [], [], [], [], []
    for y, c in zip(yhat, cluster_vec):
        if (residual_stats is not None) and (int(c) in residual_stats):
            q = residual_stats[int(c)]
            lo95.append(y + q[0]); lo68.append(y + q[1]); med.append(y + q[2])
            hi68.append(y + q[3]); hi95.append(y + q[4])
        else:
            lo95.append(np.nan); lo68.append(np.nan); med.append(np.nan); hi68.append(np.nan); hi95.append(np.nan)
    return np.array(lo95), np.array(lo68), np.array(med), np.array(hi68), np.array(hi95)

# -------------------- 侧边栏 --------------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Single prediction", "Batch prediction (CSV)"])
    model_mode = st.radio("Model", ["Hybrid (recommended)", "Global only", "Both (compare)"], index=0)
    st.markdown("Features needed: **year, sown_area_hectare, cityLat, cityLon, climate_cluster**")
    st.caption("If climate_cluster is missing, the app can auto-detect by nearest weather city.")
    show_explain = st.checkbox("Show SHAP explanation (single prediction)", value=False)

st.title("Wheat Yield per Hectare — Hybrid (Global + Specialists)")

# -------------------- 单条预测 --------------------
if mode == "Single prediction":
    colA, colB = st.columns(2)
    with colA:
        year = st.number_input("Year", min_value=1990, max_value=2100, value=2018, step=1)
        sown_area = st.number_input("Sown area (hectare)", min_value=0.0, value=1000.0, step=10.0, format="%.3f")
        use_city = st.checkbox("Select by city name (auto-fill lat/lon)", value=CITY_COORDS is not None)
        if use_city and (CITY_COORDS is not None):
            city_name = st.selectbox("City", sorted(CITY_COORDS["city"].unique()))
            lat = float(CITY_COORDS.loc[CITY_COORDS["city"]==city_name, "cityLat"].iloc[0])
            lon = float(CITY_COORDS.loc[CITY_COORDS["city"]==city_name, "cityLon"].iloc[0])
            st.caption(f"Auto-filled: lat={lat:.6f}, lon={lon:.6f}")
        else:
            lat = st.number_input("Latitude (cityLat)", min_value=-90.0, max_value=90.0, value=34.750000, step=0.01, format="%.6f")
            lon = st.number_input("Longitude (cityLon)", min_value=-180.0, max_value=180.0, value=113.620000, step=0.01, format="%.6f")
    with colB:
        auto_cluster = st.checkbox("Auto-detect climate cluster by nearest weather city", value=True)
        manual_cluster = None
        if not auto_cluster:
            manual_cluster = st.number_input("Climate cluster (int)", min_value=0, value=0, step=1)

        if st.button("Detect cluster"):
            c, near_city = nearest_cluster(lat, lon, int(year), WEATHER_REF)
            if pd.isna(c):
                st.error("No weather reference found.")
            else:
                st.success(f"Detected cluster: {c}  (nearest weather city: {near_city})")

        if st.button("Predict"):
            c, _ = (nearest_cluster(lat, lon, int(year), WEATHER_REF) if auto_cluster else (manual_cluster, None))
            if pd.isna(c):
                st.error("Cannot detect cluster.")
            else:
                row = pd.DataFrame([{
                    "year": year,
                    "sown_area_hectare": sown_area,
                    "cityLat": lat,
                    "cityLon": lon,
                    "climate_cluster": int(c)
                }])
                Xp = ensure_feature_frame(row)

                if model_mode == "Global only":
                    y_g = predict_global(Xp)
                    st.subheader(f"Global prediction: {y_g[0]:.3f}")
                    y_show = y_g; cluster_vec = np.array([int(c)])
                elif model_mode == "Hybrid (recommended)":
                    y_h = predict_hybrid(Xp, np.array([int(c)]))
                    st.subheader(f"Hybrid prediction: {y_h[0]:.3f}")
                    y_show = y_h; cluster_vec = np.array([int(c)])
                else:
                    y_g = predict_global(Xp)
                    y_h = predict_hybrid(Xp, np.array([int(c)]))
                    st.subheader(f"Hybrid: {y_h[0]:.3f}   |   Global: {y_g[0]:.3f}")
                    st.caption(f"Diff (Hybrid - Global): {float(y_h[0]-y_g[0]):.3f}")
                    y_show = y_h; cluster_vec = np.array([int(c)])

                # 预测区间（若有 residual_stats）
                pi = predict_interval(y_show, cluster_vec, RESIDUAL)
                if pi is not None:
                    lo95, lo68, med, hi68, hi95 = pi
                    st.write(f"68% interval: [{lo68[0]:.3f}, {hi68[0]:.3f}]  |  95% interval: [{lo95[0]:.3f}, {hi95[0]:.3f}]")

                # 单点 SHAP
                if show_explain:
                    st.markdown("**Local explanation (SHAP values)**")
                    explainer = shap.TreeExplainer(global_model if (model_mode=="Global only") else (specialists.get(int(c), global_model)))
                    sv = explainer(Xp[FEATS])
                    st_shap(shap.plots.bar(sv, max_display=10), height=300)

    # 地图：当前点 + 最近气象站
    st.markdown("---")
    st.markdown("**Map**")
    m = folium.Map(location=[lat, lon], zoom_start=5)
    folium.Marker([lat, lon], tooltip="Input city", icon=folium.Icon(color="blue")).add_to(m)
    c_det, near_city = nearest_cluster(lat, lon, int(year), WEATHER_REF)
    if not pd.isna(c_det):
        row = WEATHER_REF[(WEATHER_REF["city"]==near_city) & (WEATHER_REF["year"].between(year-3, year+3))].head(1)
        if not row.empty:
            folium.Marker(
                [row["cityLat"].iloc[0], row["cityLon"].iloc[0]],
                tooltip=f"Nearest weather city: {near_city} (cluster {int(c_det)})",
                icon=folium.Icon(color="green")
            ).add_to(m)
    st_folium(m, width=900)

# -------------------- 批量预测 --------------------
if mode == "Batch prediction (CSV)":
    st.write("Upload CSV with columns: year, sown_area_hectare, cityLat, cityLon, [optional] climate_cluster")
    tmpl = pd.DataFrame({
        "year":[2018,2019],
        "sown_area_hectare":[1000,1200],
        "cityLat":[34.75, 36.06],
        "cityLon":[113.62, 120.38],
        "climate_cluster":[np.nan, np.nan]
    })
    st.download_button("Download template CSV", data=tmpl.to_csv(index=False).encode("utf-8"),
                        file_name="template.csv", mime="text/csv")

    up = st.file_uploader("Choose CSV", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        df_proc = df_in.copy()
        if "climate_cluster" not in df_proc.columns:
            df_proc["climate_cluster"] = np.nan

        miss = df_proc["climate_cluster"].isna()
        if miss.any():
            st.info(f"Auto-detecting clusters for {miss.sum()} rows...")
            det = []
            for i, r in df_proc.loc[miss, ["cityLat","cityLon","year"]].iterrows():
                cc, _ = nearest_cluster(float(r["cityLat"]), float(r["cityLon"]), int(r["year"]), WEATHER_REF)
                det.append((i, cc))
            for i, cc in det:
                df_proc.at[i, "climate_cluster"] = cc

        Xp = ensure_feature_frame(df_proc)
        cvec = df_proc["climate_cluster"].astype(int).values

        if model_mode == "Global only":
            y = predict_global(Xp)
            out = df_in.copy(); out["pred_global"] = y
        elif model_mode == "Hybrid (recommended)":
            y = predict_hybrid(Xp, cvec)
            out = df_in.copy(); out["pred_hybrid"] = y
        else:
            yg = predict_global(Xp)
            yh = predict_hybrid(Xp, cvec)
            out = df_in.copy()
            out["pred_global"] = yg; out["pred_hybrid"] = yh
            out["pred_diff"] = yh - yg

        st.success("Done.")
        st.dataframe(out.head(20))

        # 汇总：按 cluster 统计
        st.markdown("**Summary by cluster**")
        if model_mode == "Both (compare)":
            grp = out.assign(cluster=cvec).groupby("cluster").agg(
                n=("cluster","size"),
                pred_global_mean=("pred_global","mean"),
                pred_hybrid_mean=("pred_hybrid","mean")
            )
        elif model_mode == "Global only":
            grp = out.assign(cluster=cvec).groupby("cluster").agg(n=("cluster","size"), pred_mean=("pred_global","mean"))
        else:
            grp = out.assign(cluster=cvec).groupby("cluster").agg(n=("cluster","size"), pred_mean=("pred_hybrid","mean"))
        st.dataframe(grp)

        # 地图：批量点
        st.markdown("**Map**")
        m = folium.Map(location=[float(np.median(df_proc["cityLat"])), float(np.median(df_proc["cityLon"]))], zoom_start=5)
        for _, r in df_proc.iterrows():
            cc = int(r["climate_cluster"])
            color = ["red","blue","green","purple","orange"][cc % 5]
            folium.CircleMarker(
                [r["cityLat"], r["cityLon"]],
                radius=4, color=color, fill=True, fill_opacity=0.9,
                tooltip=f"cluster {cc}"
            ).add_to(m)
        st_folium(m, width=900)

        # 下载
        st.download_button("Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv", mime="text/csv")


