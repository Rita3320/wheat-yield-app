# app.py
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.neighbors import BallTree
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Optional imports (safe fallbacks) ----------------
# Map (folium) – optional
MAP_AVAILABLE = False
try:
    from streamlit_folium import st_folium
    import folium
    MAP_AVAILABLE = True
except Exception:
    MAP_AVAILABLE = False

# SHAP – optional
SHAP_AVAILABLE = False
try:
    import shap
    from streamlit_shap import st_shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="Wheat Yield Predictor", layout="wide")

# ---------------- Artifact loaders ----------------
@st.cache_resource
def load_artifacts():
    global_model = joblib.load("model_global_xgb.joblib")

    try:
        specialists = joblib.load("model_specialists.joblib")  # dict: {cluster: model}
        if not isinstance(specialists, dict):
            specialists = {}
    except Exception:
        specialists = {}

    feat_list = joblib.load("feature_list.joblib")             # list of feature columns
    weather_ref = pd.read_csv("weather_ref.csv")               # city,year,cityLat,cityLon,climate_cluster

    # Optional artifacts
    try:
        city_coords = pd.read_csv("city_coords.csv")           # city,cityLat,cityLon
    except Exception:
        city_coords = None

    try:
        residual_stats = joblib.load("residual_stats.joblib")  # {cluster: np.array([q2.5,q16,q50,q84,q97.5])}
    except Exception:
        residual_stats = None

    return global_model, specialists, feat_list, weather_ref, city_coords, residual_stats

global_model, specialists, FEATS, WEATHER_REF, CITY_COORDS, RESIDUAL = load_artifacts()

# ---------------- Utilities ----------------
def ensure_feature_frame(df_like: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected features exist and order matches FEATS."""
    X = df_like.copy()
    for c in FEATS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEATS]
    # simple numeric fill; customize if you use scalers/encoders at training
    return X.astype(float).fillna(X.median(numeric_only=True))

def nearest_cluster(lat: float, lon: float, year: int, weather_df: pd.DataFrame, max_year_span: int = 3):
    """Find nearest weather city in the same year; fallback to ±N years if needed."""
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
    _, idx = tree.query(np.radians([[lat, lon]]), k=1)
    j = int(idx[0, 0])
    row = df.iloc[j]
    return int(row["climate_cluster"]), row["city"]

def predict_global(X_df: pd.DataFrame) -> np.ndarray:
    return global_model.predict(X_df[FEATS].values)

def predict_hybrid(X_df: pd.DataFrame, cluster_vec: np.ndarray) -> np.ndarray:
    """Global everywhere, override with selected specialists where available."""
    yhat = global_model.predict(X_df[FEATS].values)
    for cid, model in specialists.items():
        mask = (cluster_vec == cid)
        if np.any(mask):
            yhat[mask] = model.predict(X_df.loc[mask, FEATS].values)
    return yhat

def predict_interval(yhat: np.ndarray, cluster_vec: np.ndarray, residual_stats):
    """Estimate prediction intervals using stored hold-out residual quantiles."""
    if residual_stats is None:
        return None
    lo95, lo68, med, hi68, hi95 = [], [], [], [], []
    for y, c in zip(yhat, cluster_vec):
        c = int(c)
        if c in residual_stats:
            q = residual_stats[c]  # [2.5,16,50,84,97.5] residuals
            lo95.append(y + q[0]); lo68.append(y + q[1]); med.append(y + q[2])
            hi68.append(y + q[3]); hi95.append(y + q[4])
        else:
            lo95.append(np.nan); lo68.append(np.nan); med.append(np.nan); hi68.append(np.nan); hi95.append(np.nan)
    return np.array(lo95), np.array(lo68), np.array(med), np.array(hi68), np.array(hi95)

# ---- Batch evaluation helpers ----
EVAL_TARGET_COLUMNS = ["actual_yield_per_hectare", "yield_per_hectare", "actual", "target"]

def find_target_column(df: pd.DataFrame):
    for c in EVAL_TARGET_COLUMNS:
        if c in df.columns:
            return c
    return None

def residual_histogram_series(residuals: np.ndarray, bins: int = 30) -> pd.DataFrame:
    res = residuals[~np.isnan(residuals)]
    if res.size == 0:
        return pd.DataFrame({"bin_mid": [], "count": []}).set_index("bin_mid")
    hist, bin_edges = np.histogram(res, bins=bins)
    mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return pd.DataFrame({"bin_mid": mids, "count": hist}).set_index("bin_mid")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Single prediction", "Batch prediction (CSV)"])
    model_mode = st.radio("Model", ["Hybrid (recommended)", "Global only", "Both (compare)"], index=0)
    st.markdown("Required columns: **year, sown_area_hectare, cityLat, cityLon, climate_cluster**")
    st.caption("If climate_cluster is missing, the app can auto-detect by nearest weather city.")
    show_explain = st.checkbox("Show SHAP explanation (single prediction)", value=False, disabled=not SHAP_AVAILABLE)
    if not SHAP_AVAILABLE:
        st.caption("SHAP not available. Add shap/numba/llvmlite in requirements.txt to enable.")

st.title("Wheat Yield per Hectare — Hybrid (Global + Specialists)")

# ---------------- Single prediction ----------------
if mode == "Single prediction":
    colA, colB = st.columns(2)
    with colA:
        year = st.number_input("Year", min_value=1990, max_value=2100, value=2018, step=1)
        sown_area = st.number_input("Sown area (hectare)", min_value=0.0, value=1000.0, step=10.0, format="%.3f")

        use_city = st.checkbox("Select by city name (auto-fill lat/lon)", value=(CITY_COORDS is not None))
        if use_city and (CITY_COORDS is not None):
            city_name = st.selectbox("City", sorted(CITY_COORDS["city"].unique()))
            lat = float(CITY_COORDS.loc[CITY_COORDS["city"] == city_name, "cityLat"].iloc[0])
            lon = float(CITY_COORDS.loc[CITY_COORDS["city"] == city_name, "cityLon"].iloc[0])
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
                st.error("No weather reference found for this location/year.")
            else:
                st.success(f"Detected cluster: {c}  (nearest weather city: {near_city})")

        y_show, cluster_vec = None, None
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
                    st.caption(f"Diff (Hybrid - Global): {float(y_h[0] - y_g[0]):.3f}")
                    y_show = y_h; cluster_vec = np.array([int(c)])

                # Prediction intervals (if residual stats available)
                pi = predict_interval(y_show, cluster_vec, RESIDUAL)
                if pi is not None:
                    lo95, lo68, med, hi68, hi95 = pi
                    st.write(f"68% interval: [{lo68[0]:.3f}, {hi68[0]:.3f}]  |  95% interval: [{lo95[0]:.3f}, {hi95[0]:.3f}]")

                # SHAP local explanation (optional)
                if show_explain and SHAP_AVAILABLE:
                    model_for_explain = global_model if (model_mode == "Global only") else specialists.get(int(c), global_model)
                    try:
                        explainer = shap.TreeExplainer(model_for_explain)
                        sv = explainer(Xp[FEATS])
                        st_shap(shap.plots.bar(sv, max_display=10), height=320)
                    except Exception as e:
                        st.warning(f"SHAP failed: {e}")

                # -------- What-if analysis (interactive) --------
                with st.expander("What-if analysis"):
                    st.caption("Adjust inputs and see how the prediction changes (cluster will re-detect).")
                    base_pred = float(y_show[0])

                    col_w1, col_w2 = st.columns(2)
                    with col_w1:
                        year_w = st.slider("Year (what-if)", int(year - 5), int(year + 5), int(year), step=1)
                        sown_area_w = st.slider("Sown area (hectare, what-if)",
                                                max(0.0, sown_area * 0.2), sown_area * 5.0, float(sown_area),
                                                step=10.0)
                    with col_w2:
                        lat_w = st.slider("Latitude (what-if)", -90.0, 90.0, float(lat), step=0.01)
                        lon_w = st.slider("Longitude (what-if)", -180.0, 180.0, float(lon), step=0.01)

                    c_w, _ = nearest_cluster(lat_w, lon_w, int(year_w), WEATHER_REF)
                    if pd.isna(c_w):
                        st.warning("No weather reference for what-if point.")
                    else:
                        row_w = pd.DataFrame([{
                            "year": year_w,
                            "sown_area_hectare": sown_area_w,
                            "cityLat": lat_w,
                            "cityLon": lon_w,
                            "climate_cluster": int(c_w)
                        }])
                        Xw = ensure_feature_frame(row_w)
                        if model_mode == "Global only":
                            y_w = predict_global(Xw)[0]
                        else:
                            y_w = predict_hybrid(Xw, np.array([int(c_w)]))[0]

                        st.write(f"Detected cluster for what-if: **{int(c_w)}**")
                        st.metric("What-if prediction", f"{y_w:.3f}", delta=float(y_w - base_pred))

    # Map for current point
    st.markdown("---")
    st.markdown("Map")
    if MAP_AVAILABLE:
        try:
            m = folium.Map(location=[lat, lon], zoom_start=5)
            folium.Marker([lat, lon], tooltip="Input city", icon=folium.Icon(color="blue")).add_to(m)
            c_det, near_city = nearest_cluster(lat, lon, int(year), WEATHER_REF)
            if not pd.isna(c_det):
                row = WEATHER_REF[(WEATHER_REF["city"] == near_city) &
                                  (WEATHER_REF["year"].between(year - 3, year + 3))].head(1)
                if not row.empty:
                    folium.Marker(
                        [row["cityLat"].iloc[0], row["cityLon"].iloc[0]],
                        tooltip=f"Nearest weather city: {near_city} (cluster {int(c_det)})",
                        icon=folium.Icon(color="green")
                    ).add_to(m)
            st_folium(m, width=900)
            # Download map as HTML
            html_map = m.get_root().render()
            st.download_button("Download map (HTML)", data=html_map, file_name="map.html", mime="text/html")
        except Exception as e:
            st.info(f"Map skipped: {e}")
    else:
        st.info("Map disabled (streamlit-folium not installed).")

# ---------------- Batch prediction ----------------
if mode == "Batch prediction (CSV)":
    st.write("Upload CSV with columns: year, sown_area_hectare, cityLat, cityLon, [optional] climate_cluster")

    # Downloadable template
    tmpl = pd.DataFrame({
        "year": [2018, 2019],
        "sown_area_hectare": [1000, 1200],
        "cityLat": [34.75, 36.06],
        "cityLon": [113.62, 120.38],
        "climate_cluster": [np.nan, np.nan]
    })
    st.download_button("Download template CSV",
                       data=tmpl.to_csv(index=False).encode("utf-8"),
                       file_name="template.csv",
                       mime="text/csv")

    up = st.file_uploader("Choose CSV", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        df_proc = df_in.copy()

        if "climate_cluster" not in df_proc.columns:
            df_proc["climate_cluster"] = np.nan

        # Auto-detect clusters where missing
        miss = df_proc["climate_cluster"].isna()
        if miss.any():
            st.info(f"Auto-detecting clusters for {miss.sum()} rows...")
            det = []
            for i, r in df_proc.loc[miss, ["cityLat", "cityLon", "year"]].iterrows():
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

        # Summary by cluster
        st.markdown("Summary by cluster")
        if model_mode == "Both (compare)":
            grp = out.assign(cluster=cvec).groupby("cluster").agg(
                n=("cluster", "size"),
                pred_global_mean=("pred_global", "mean"),
                pred_hybrid_mean=("pred_hybrid", "mean")
            )
        elif model_mode == "Global only":
            grp = out.assign(cluster=cvec).groupby("cluster").agg(
                n=("cluster", "size"),
                pred_mean=("pred_global", "mean")
            )
        else:
            grp = out.assign(cluster=cvec).groupby("cluster").agg(
                n=("cluster", "size"),
                pred_mean=("pred_hybrid", "mean")
            )
        st.dataframe(grp)

        # --- Optional evaluation if ground-truth is provided ---
        tcol = find_target_column(df_in)
        if tcol is not None:
            st.markdown("**Evaluation (ground truth detected)**")

            if model_mode == "Global only":
                y_pred = out["pred_global"].values
            elif model_mode == "Hybrid (recommended)":
                y_pred = out["pred_hybrid"].values
            else:
                y_pred = out["pred_hybrid"].values

            y_true = df_in[tcol].values.astype(float)
            mask_eval = ~np.isnan(y_true)
            if mask_eval.sum() > 0:
                rmse = float(np.sqrt(mean_squared_error(y_true[mask_eval], y_pred[mask_eval])))
                r2 = float(r2_score(y_true[mask_eval], y_pred[mask_eval]))
                st.write(f"Overall RMSE: **{rmse:.3f}**  |  R²: **{r2:.3f}**  (n={mask_eval.sum()})")

                # per-cluster metrics
                st.write("Per-cluster metrics:")
                rows = []
                for cid in sorted(np.unique(cvec)):
                    m = (cvec == cid) & mask_eval
                    if m.sum() > 0:
                        rows.append({
                            "cluster": int(cid),
                            "n": int(m.sum()),
                            "rmse": float(np.sqrt(mean_squared_error(y_true[m], y_pred[m]))),
                            "r2": float(r2_score(y_true[m], y_pred[m]))
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows).set_index("cluster"))

                # residual histogram
                res = y_true[mask_eval] - y_pred[mask_eval]
                st.write("Residual histogram")
                hist_df = residual_histogram_series(res, bins=30)
                st.bar_chart(hist_df)
            else:
                st.info(f"Column '{tcol}' has no numeric values to evaluate.")
        else:
            st.caption(f"No ground-truth column found. Add one of {EVAL_TARGET_COLUMNS} to your CSV to see evaluation.")

        # Map for batch
        st.markdown("Map")
        if MAP_AVAILABLE:
            try:
                m = folium.Map(
                    location=[float(np.median(df_proc["cityLat"])),
                              float(np.median(df_proc["cityLon"]))],
                    zoom_start=5
                )
                for _, r in df_proc.iterrows():
                    cc = int(r["climate_cluster"])
                    color = ["red", "blue", "green", "purple", "orange"][cc % 5]
                    folium.CircleMarker(
                        [r["cityLat"], r["cityLon"]],
                        radius=4, color=color, fill=True, fill_opacity=0.9,
                        tooltip=f"cluster {cc}"
                    ).add_to(m)
                st_folium(m, width=900)
                # Download map as HTML
                html_map = m.get_root().render()
                st.download_button("Download map (HTML)", data=html_map, file_name="map.html", mime="text/html")
            except Exception as e:
                st.info(f"Map skipped: {e}")
        else:
            st.info("Map disabled (streamlit-folium not installed).")

        # Download predictions
        st.download_button("Download predictions CSV",
                           data=out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv",
                           mime="text/csv")
