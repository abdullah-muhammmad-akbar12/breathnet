# ✅ Full Working Version of BreathNet App with Auto-Run Simulation

import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
import base64

# Title and Model
st.set_page_config(page_title="BreathNet", layout="centered")
st.title("🫁 BreathNet: AI-Powered Disease Prediction from VOCs")

# Load trained model
model = joblib.load("breathnet_model.pkl")

# 🧪 Load Simulated Sensor Data
st.sidebar.markdown("---")
st.sidebar.header("📂 Simulated Sensor Data")
uploaded_file = st.sidebar.file_uploader("Upload VOC CSV (simulated BME688)", type=["csv"])

# Initialize session state
if 'sensor_index' not in st.session_state:
    st.session_state.sensor_index = 0

# Auto-run toggle
auto_run = st.sidebar.checkbox("🔁 Auto-Run Every 10 Seconds")

# Fallback default input (from sliders)
st.sidebar.header("📊 Manual VOC Input (Optional)")
acetone = st.sidebar.slider("Acetone", 0.0, 2.0, 1.0, 0.01)
ethanol = st.sidebar.slider("Ethanol", 0.0, 0.5, 0.2, 0.01)
formaldehyde = st.sidebar.slider("Formaldehyde", 0.0, 0.1, 0.03, 0.001)
ammonia = st.sidebar.slider("Ammonia", 0.0, 0.1, 0.03, 0.001)
isoprene = st.sidebar.slider("Isoprene", 0.0, 1.5, 0.8, 0.01)
hydrogen_sulfide = st.sidebar.slider("Hydrogen Sulfide", 0.0, 0.1, 0.03, 0.001)
methanol = st.sidebar.slider("Methanol", 0.0, 0.1, 0.03, 0.001)
carbonyl_index = st.sidebar.slider("Carbonyl Index", 0.0, 0.3, 0.1, 0.01)

user_input = pd.DataFrame([{
    'Acetone': acetone,
    'Ethanol': ethanol,
    'Formaldehyde': formaldehyde,
    'Ammonia': ammonia,
    'Isoprene': isoprene,
    'Hydrogen Sulfide': hydrogen_sulfide,
    'Methanol': methanol,
    'Carbonyl_Index': carbonyl_index
}])

# Override input with simulated row if available
if uploaded_file:
    df_sensor = pd.read_csv(uploaded_file)
    if st.session_state.sensor_index < len(df_sensor):
        user_input = pd.DataFrame([df_sensor.iloc[st.session_state.sensor_index]])
        if auto_run:
            st.experimental_rerun()
    else:
        st.warning("🚫 No more rows left in uploaded CSV.")

    if auto_run:
        import time
        time.sleep(10)
        st.session_state.sensor_index += 1
    elif st.sidebar.button("➡️ Next Reading"):
        st.session_state.sensor_index += 1
        st.success(f"✅ Loaded row {st.session_state.sensor_index} from CSV")

# Prediction logic
if st.button("🔍 Predict Disease") or auto_run:
    probabilities = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)[0]

    st.session_state.prediction = prediction
    st.session_state.inputs = user_input.iloc[0].to_dict()
    st.session_state.probabilities = probabilities.tolist()

    st.success(f"🧬 Predicted Disease: **{prediction}**")

    # Bar chart for confidence
    st.subheader("📊 Prediction Confidence by Disease")
    prob_df = pd.DataFrame({
        'Disease': model.classes_,
        'Confidence': probabilities
    }).sort_values(by='Confidence', ascending=False)
    st.bar_chart(prob_df.set_index('Disease'))

    # Explainable AI
    st.subheader("🧠 Top Influencing VOCs (Feature Impact)")
    feature_impact = user_input.iloc[0].sort_values(ascending=False)
    st.bar_chart(feature_impact)

    top_feature = feature_impact.index[0]
    top_value = feature_impact.iloc[0]
    st.info(f"ℹ️ Most influencing compound: **{top_feature} = {top_value:.3f} ppm**")

# PDF generator

def create_pdf(prediction, inputs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "BreathNet Disease Prediction Report", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Predicted Disease: {prediction}", ln=True)
    pdf.cell(0, 10, "", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Input VOC Concentrations (ppm):", ln=True)
    pdf.set_font("Arial", '', 12)
    for key, value in inputs.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)
    file_path = "BreathNet_Report.pdf"
    pdf.output(file_path)
    return file_path

def get_pdf_download_link(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="BreathNet_Report.pdf">📥 Download PDF Report</a>'
    return href

# PDF download section
if 'prediction' in st.session_state and 'inputs' in st.session_state:
    if st.button("📥 Generate PDF Report"):
        file_path = create_pdf(st.session_state.prediction, st.session_state.inputs)
        st.markdown(get_pdf_download_link(file_path), unsafe_allow_html=True)
else:
    st.info("ℹ️ Please run a prediction first.")

# Current input
st.subheader("🔬 Current Input Data")
st.write(user_input)

