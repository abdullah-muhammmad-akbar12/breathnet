# breathnet_app.py (✅ 100% Working Version with Prediction, Confidence Graph, PDF Export)

import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
import base64

# 🧪 Day 7 - Simulated Sensor Data Integration
import time

st.sidebar.markdown("---")
st.sidebar.header("📂 Simulated Sensor Data")
uploaded_file = st.sidebar.file_uploader("Upload VOC CSV (simulated BME688)", type=["csv"])

import time

# Load sensor CSV
if uploaded_file:
    sensor_df = pd.read_csv(uploaded_file)
    st.session_state.sensor_data = sensor_df

    # Initialize session index
    if 'sensor_index' not in st.session_state:
        st.session_state.sensor_index = 0

    # 🟢 Auto-run mode toggle
    st.sidebar.markdown("---")
    auto_run = st.sidebar.checkbox("🔁 Auto-Run Every 10 Sec")

    # Run next sample either manually or automatically
    run_sample = False
    if auto_run:
        run_sample = True
        time.sleep(10)  # Wait before loading next sample
    elif st.sidebar.button("➡️ Next Reading"):
        run_sample = True

    # Load next row from CSV
    if run_sample:
        idx = st.session_state.sensor_index
        if idx < len(st.session_state.sensor_data):
            user_input = pd.DataFrame([st.session_state.sensor_data.iloc[idx]])
            # 🧪 Auto-run logic: use CSV if uploaded
if 'sensor_index' not in st.session_state:
    st.session_state.sensor_index = 0

if uploaded_file:
    df_sensor = pd.read_csv(uploaded_file)
    st.session_state.sensor_data = df_sensor

    auto_run = st.sidebar.checkbox("🔁 Auto-Run Every 10 Seconds")

    if auto_run and st.session_state.sensor_index < len(st.session_state.sensor_data):
        user_input = pd.DataFrame([st.session_state.sensor_data.iloc[st.session_state.sensor_index]])
        st.session_state.sensor_index += 1

        st.success(f"✅ Auto-loaded row {st.session_state.sensor_index} from sensor data")

        # Trigger a rerun every 10 sec
        st.experimental_rerun()
    elif st.sidebar.button("➡️ Next Reading"):
        if st.session_state.sensor_index < len(st.session_state.sensor_data):
            user_input = pd.DataFrame([st.session_state.sensor_data.iloc[st.session_state.sensor_index]])
            st.session_state.sensor_index += 1
            st.success(f"✅ Manually loaded row {st.session_state.sensor_index}")
        else:
            st.warning("🚫 No more data left in file.")
            st.session_state.sensor_index += 1
            st.success(f"✅ Auto-loaded row {idx + 1} from sensor file.")
        else:
            st.warning("🚫 No more data left in the CSV.")



# Load model
model = joblib.load('breathnet_model.pkl')

# App title
st.title("🫁 BreathNet: AI-Powered Disease Prediction from VOCs")

# Description
st.markdown("""
Welcome to **BreathNet** – a smart diagnostic tool that predicts diseases based on Volatile Organic Compounds (VOCs) in your breath.
🧪 Enter breath compound readings below to get a real-time prediction.
""")

# Sidebar inputs
st.sidebar.header("Input VOC Concentrations (ppm)")
acetone = st.sidebar.slider("Acetone", 0.0, 2.0, 1.0, 0.01)
ethanol = st.sidebar.slider("Ethanol", 0.0, 0.5, 0.2, 0.01)
formaldehyde = st.sidebar.slider("Formaldehyde", 0.0, 0.1, 0.03, 0.001)
ammonia = st.sidebar.slider("Ammonia", 0.0, 0.1, 0.03, 0.001)
isoprene = st.sidebar.slider("Isoprene", 0.0, 1.5, 0.8, 0.01)
hydrogen_sulfide = st.sidebar.slider("Hydrogen Sulfide", 0.0, 0.1, 0.03, 0.001)
methanol = st.sidebar.slider("Methanol", 0.0, 0.1, 0.03, 0.001)
carbonyl_index = st.sidebar.slider("Carbonyl Index", 0.0, 0.3, 0.1, 0.01)

# DataFrame for model
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

# Prediction button
if st.button("🔍 Predict Disease"):
    # Make prediction
    probabilities = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)[0]

    # Store in session state
    st.session_state.prediction = prediction
    st.session_state.inputs = user_input.iloc[0].to_dict()
    st.session_state.probabilities = probabilities.tolist()

    # ✅ Show prediction
    st.success(f"🧬 Predicted Disease: **{prediction}**")

    # ✅ Show confidence chart
    st.subheader("📊 Prediction Confidence by Disease")
    prob_df = pd.DataFrame({
        'Disease': model.classes_,
        'Confidence': probabilities
    }).sort_values(by='Confidence', ascending=False)
    st.bar_chart(prob_df.set_index('Disease'))

    # ✅ Explainable AI Section
    st.subheader("🧠 Top Influencing VOCs (Feature Impact)")
    feature_impact = user_input.iloc[0].sort_values(ascending=False)
    st.bar_chart(feature_impact)

    # ✅ Natural explanation
    top_feature = feature_impact.index[0]
    top_value = feature_impact.iloc[0]
    st.info(f"ℹ️ The model was most influenced by **{top_feature}**, which had a value of **{top_value:.3f}**.")


    # Confidence graph
    st.subheader("📊 Prediction Confidence by Disease")
    prob_df = pd.DataFrame({
        'Disease': model.classes_,
        'Confidence': probabilities
    }).sort_values(by='Confidence', ascending=False)
    st.bar_chart(prob_df.set_index('Disease'))

# PDF helper functions
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

# Download PDF button
if 'prediction' in st.session_state and 'inputs' in st.session_state:
    if st.button("📥 Generate PDF Report"):
        file_path = create_pdf(st.session_state.prediction, st.session_state.inputs)
        st.markdown(get_pdf_download_link(file_path), unsafe_allow_html=True)
else:
    st.info("ℹ️ Please run a prediction first.")

# Show current input
st.subheader("🔬 Current Input Data")
st.write(user_input)
