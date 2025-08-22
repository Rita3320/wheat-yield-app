import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime
from io import BytesIO
from sklearn.neighbors import BallTree
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pydeck as pdk
import plotly.express as px

st.set_page_config(page_title="Wheat Yield App", layout="wide")

# ---------- Load ----------
@st.cache_resource
def load_artifacts():
    global_model = joblib.load("model_global_xgb.joblib")
    specialists = joblib.load("model_specialists.joblib")
    feat_list = joblib.load("feature_list.joblib")
    weather_ref = pd.read_csv("weather_ref.csv")
    return global_model, specialists, feat_list, weather_ref

global_model, specialists, FEATS, WEATHER_REF = load_artifacts()

# ---------- Utils ----------
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
    return buffer.getvalue()

def cluster_insight(cluster_id):
    examples = {
        0: ["High humidity region. Irrigation timing is key.", "Apply nitrogen early in the season."],
        1: ["Moderate rainfall zone. Disease control is essential.", "Sow earlier to avoid dry season."],
        2: ["Hot and dry climate. Drought-resistant varieties perform better."],
        3: ["Cool and wet area. Monitor for fungal outbreaks.", "Use late-maturing wheat types."],
    }
    options = examples.get(cluster_id, ["General best practices apply."])
    return np.random.choice(options)

# ---------- UI ----------
st.title("Wheat Yield per Hectare — Hybrid Model")

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Climate Map", "Cluster Plot", "History"])

# ---------- Prediction Tab ----------
with tab1:
    with st.sidebar:
        st.header("Prediction Mode")
        mode = st.radio("Select", ["Single prediction", "Batch prediction (CSV)"])

    if mode == "Single prediction":
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Year", 1990, 2100, 2018)
            sown_area = st.number_input("Sown area (ha)", 0.0, 1e6, 1000.0, step=10.0)
        with col2:
            lat = st.number_input("Latitude", -90.0, 90.0, 34.75)
            lon = st.number_input("Longitude", -180.0, 180.0, 113.62)

        auto_cluster = st.checkbox("Auto-detect climate cluster", value=True)
        cluster = None
        if auto_cluster:
            if st.button("Detect cluster"):
                cluster, nearest = nearest_cluster(lat, lon, int(year), WEATHER_REF)
                if np.isnan(cluster):
                    st.error("No nearby weather station.")
                else:
                    st.success(f"Detected cluster: {cluster}")
        else:
            cluster = st.number_input("Cluster ID", 0, 10, 0)

        if st.button("Predict"):
            if cluster is None or pd.isna(cluster):
                cluster, _ = nearest_cluster(lat, lon, int(year), WEATHER_REF)

            X_row = pd.DataFrame([{
                "year": year, "sown_area_hectare": sown_area,
                "cityLat": lat, "cityLon": lon, "climate_cluster": cluster
            }])
            X_feat = ensure_feature_frame(X_row)
            yhat = predict_hybrid(X_feat, np.array([int(cluster)]))

            st.subheader(f"Predicted Yield: {yhat[0]:.3f} tons/ha")
            st.write("Insight:", cluster_insight(cluster))

            # 保存到 session state 历史
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Yield": round(yhat[0], 3),
                "Cluster": cluster
            })

            # PDF 下载
            pdf_data = generate_pdf_report({
                "year": year, "sown_area_hectare": sown_area,
                "cityLat": lat, "cityLon": lon, "climate_cluster": cluster
            }, yhat[0])
            st.download_button("Download PDF", data=pdf_data, file_name="report.pdf", mime="application/pdf")

    elif mode == "Batch prediction (CSV)":
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            df_in = pd.read_csv(up)
            df_proc = df_in.copy()

            if "climate_cluster" not in df_proc.columns:
                df_proc["climate_cluster"] = np.nan

            miss = df_proc["climate_cluster"].isna()
            if miss.any():
                clusters = []
                for (i, r) in df_proc.loc[miss, ["cityLat", "cityLon", "year"]].iterrows():
                    cc, _ = nearest_cluster(r["cityLat"], r["cityLon"], r["year"], WEATHER_REF)
                    clusters.append((i, cc))
                for i, cc in clusters:
                    df_proc.at[i, "climate_cluster"] = cc

            X_feat = ensure_feature_frame(df_proc)
            yhat = predict_hybrid(X_feat, df_proc["climate_cluster"].astype(int).values)
            df_out = df_in.copy()
            df_out["pred_yield"] = yhat
            st.dataframe(df_out.head(20))
            st.download_button("Download results", df_out.to_csv(index=False), file_name="predictions.csv")

# ---------- Map Tab ----------
with tab2:
    st.subheader("Weather Station Climate Clusters")
    map_df = WEATHER_REF.drop_duplicates(subset=["cityLat", "cityLon", "climate_cluster"])
    st.map(map_df.rename(columns={"cityLat": "lat", "cityLon": "lon"}))

# ---------- Cluster Plot Tab ----------
with tab3:
    st.subheader("Cluster Distribution")

    # 自定义颜色
    custom_colors = {
        "0": "#FF0000",  # 红
        "1": "#0066FF",  # 蓝
        "2": "#00CC66",  # 绿
        "3": "#FF9900",  # 橙
    }

    fig = px.scatter_mapbox(
        plot_df,
        lat="cityLat",
        lon="cityLon",
        color="climate_cluster",
        hover_name="city",
        mapbox_style="carto-positron",
        zoom=4,
        height=600,
        color_discrete_map=custom_colors,
    )
    st.plotly_chart(fig, use_container_width=True)



# ---------- History Tab ----------
with tab4:
    st.subheader("Your Prediction History")
    if "history" in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.info("No predictions yet.")



