import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.neighbors import BallTree
import datetime
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Wheat Yield Predictor", layout="wide")

# ---------- Load resources ----------
@st.cache_resource
def load_artifacts():
    global_model = joblib.load("model_global_xgb.joblib")
    specialists = joblib.load("model_specialists.joblib")
    feat_list = joblib.load("feature_list.joblib")
    weather_ref = pd.read_csv("weather_ref.csv")
    return global_model, specialists, feat_list, weather_ref

global_model, specialists, FEATS, WEATHER_REF = load_artifacts()

# ---------- Utilities ----------
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

def generate_pdf_report(input_dict, pred_yield):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Wheat Yield Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Input Features:", styles['Heading2']))
    for key, value in input_dict.items():
        story.append(Paragraph(f"{key}: {value}", styles['Normal']))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Prediction Result:", styles['Heading2']))
    story.append(Paragraph(f"Predicted Yield: {pred_yield:.3f} tons/ha", styles['Normal']))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# ---------- UI ----------
st.title("Wheat Yield per Hectare — Hybrid (Global + Specialists)")

with st.sidebar:
    st.header("Prediction Mode")
    mode = st.radio("Select", ["Single prediction", "Batch prediction (CSV)"])
    st.markdown("Expected features: **year, sown_area_hectare, cityLat, cityLon, climate_cluster**")
    st.caption("If climate_cluster is not provided, the app can auto-detect by nearest weather city.")

# ---------- Single prediction ----------
if mode == "Single prediction":
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=1990, max_value=2100, value=2018, step=1)
        sown_area = st.number_input("Sown area (hectare)", min_value=0.0, value=1000.0, step=10.0, format="%.3f")
    with col2:
        lat = st.number_input("Latitude (cityLat)", min_value=-90.0, max_value=90.0, value=34.75, step=0.01, format="%.6f")
        lon = st.number_input("Longitude (cityLon)", min_value=-180.0, max_value=180.0, value=113.62, step=0.01, format="%.6f")

    auto_cluster = st.checkbox("Auto-detect climate cluster by nearest weather city", value=True)
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
        cluster = st.number_input("Climate cluster (integer)", min_value=0, value=0, step=1)

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
        st.subheader(f"Predicted Yield: {yhat[0]:.3f} tons/ha")
        st.caption("Feature vector used:")
        st.dataframe(X_prepared)

        pdf_data = generate_pdf_report({
            "year": year,
            "sown_area_hectare": sown_area,
            "cityLat": lat,
            "cityLon": lon,
            "climate_cluster": cluster
        }, yhat[0])

        st.download_button("Download PDF Report", data=pdf_data, file_name="prediction_report.pdf", mime="application/pdf")

# ---------- Batch prediction ----------
if mode == "Batch prediction (CSV)":
    st.write("Upload a CSV with columns: year, sown_area_hectare, cityLat, cityLon, [optional] climate_cluster")
    up = st.file_uploader("Choose CSV", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        df_proc = df_in.copy()

        if "climate_cluster" not in df_proc.columns:
            df_proc["climate_cluster"] = np.nan

        miss = df_proc["climate_cluster"].isna()
        if miss.any():
            st.info(f"Auto-detecting clusters for {miss.sum()} rows without climate_cluster...")
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
        st.success("Done.")
        st.dataframe(out.head(20))

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
