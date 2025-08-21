import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import base64
from io import BytesIO
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import BallTree
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# 加载模型
model_global = joblib.load("model_global.pkl")
model_hybrid = joblib.load("model_hybrid.pkl")

st.set_page_config(page_title="Yield Prediction App", layout="wide")
st.title("Wheat Yield Prediction App")

st.sidebar.header("Prediction Mode")
mode = st.sidebar.radio("Choose prediction mode:", ["Single Input", "Batch Upload"])

# 单个预测输入表单
def single_input_form():
    st.subheader("Enter Climate and Location Data")
    temperature = st.number_input("Temperature (°C)", 10.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    wind = st.slider("Wind Speed (km/h)", 0, 50, 10)
    cluster = st.selectbox("Climate Cluster", [0, 1, 2, 3])

    if st.button("Predict Yield"):
        input_data = pd.DataFrame({
            "temperature": [temperature],
            "humidity": [humidity],
            "wind": [wind],
            "cluster": [cluster]
        })

        pred_global = model_global.predict(input_data.drop(columns="cluster"))
        pred_hybrid = model_hybrid.predict(input_data)

        st.success("Prediction Results")
        st.write("Global Model Prediction:", round(pred_global[0], 2), "tons/ha")
        st.write("Hybrid Model (with cluster) Prediction:", round(pred_hybrid[0], 2), "tons/ha")

        insight = f"Based on your input, the hybrid model predicts yield = {round(pred_hybrid[0], 2)} tons/ha in cluster {cluster}."
        st.info(insight)

        # PDF 下载按钮
        if st.button("Download Prediction as PDF"):
            pdf_bytes = generate_pdf_report(input_data, pred_global[0], pred_hybrid[0], insight)
            st.download_button(label="Download PDF Report", data=pdf_bytes, file_name="yield_prediction.pdf")

# 生成 PDF 报告
def generate_pdf_report(input_data, pred_g, pred_h, insight):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Wheat Yield Prediction Report</b>", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']),
        Spacer(1, 12),
        Paragraph("<b>Input Features:</b>", styles['Heading2'])
    ]

    for col in input_data.columns:
        story.append(Paragraph(f"{col}: {input_data[col].iloc[0]}", styles['Normal']))

    story.extend([
        Spacer(1, 12),
        Paragraph("<b>Prediction Results:</b>", styles['Heading2']),
        Paragraph(f"Global Model: {round(pred_g, 2)} tons/ha", styles['Normal']),
        Paragraph(f"Hybrid Model: {round(pred_h, 2)} tons/ha", styles['Normal']),
        Spacer(1, 12),
        Paragraph("<b>Insight:</b>", styles['Heading2']),
        Paragraph(insight, styles['Normal'])
    ])

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# 批量上传预测
def batch_upload_mode():
    st.subheader("Upload CSV File for Batch Prediction")
    template = pd.DataFrame({
        "temperature": [25.0],
        "humidity": [60],
        "wind": [10],
        "cluster": [0]
    })
    csv_buffer = BytesIO()
    template.to_csv(csv_buffer, index=False)
    st.download_button("Download Template CSV", data=csv_buffer.getvalue(), file_name="template.csv")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if not set(["temperature", "humidity", "wind", "cluster"]).issubset(df.columns):
            st.error("Missing required columns. Please use the template.")
            return

        pred_g = model_global.predict(df.drop(columns="cluster"))
        pred_h = model_hybrid.predict(df)

        df_result = df.copy()
        df_result["Global_Prediction"] = np.round(pred_g, 2)
        df_result["Hybrid_Prediction"] = np.round(pred_h, 2)
        df_result["Insight"] = df_result.apply(
            lambda row: f"Hybrid predicts {row['Hybrid_Prediction']} tons/ha in cluster {row['cluster']}", axis=1)

        st.success("Prediction completed.")
        st.dataframe(df_result.style.format("{:.2f}"))

        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results CSV", csv, file_name="batch_prediction_result.csv")

# 主流程判断
if mode == "Single Input":
    single_input_form()
else:
    batch_upload_mode()
