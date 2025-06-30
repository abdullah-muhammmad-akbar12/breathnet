# ✅ BreathNet App - Final with XAI + PDF + Logging

import streamlit as st
import pandas as pd
import joblib
import time
import base64
import csv
from datetime import datetime
from fpdf import FPDF

# Load model
model = joblib.load("breathnet_model.pkl")

# Setup
st.set_page_config(page_title="BreathNet", layout="centered")
st.title("🫁 BreathNet: AI-Powered Disease Prediction from VOCs")

# Upload CSV
st.sidebar.header("📂 Upload VOC CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# VOC Columns
expected_columns = ['Acetone', 'Ethanol', 'Formaldehyde', 'Ammonia',
                    'Isoprene', 'Hydrogen Sulfide', 'Methanol', 'Carbonyl_Index']

# Session state
if "sensor_index" not in st.session_state:
    st.session_state.sensor_index = 0
if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = None
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = time.time()
if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = False
if "auto_predict" not in st.session_state:
    st.session_state.auto_predict = False

# Manual input sliders
st.sidebar.header("Manual VOC Input")
manual_input = {
    col: st.sidebar.slider(
        col, 0.0, 2.0 if col == "Acetone" else 0.5,
        1.0 if col == "Acetone" else 0.03, 0.01
    ) for col in expected_columns
}
user_input = pd.DataFrame([manual_input])

# Auto-run checkbox
st.sidebar.markdown("---")
st.session_state.auto_mode = st.sidebar.checkbox("🔁 Auto-Run Every 10 Seconds")

# Load CSV into session
if uploaded_file:
    st.session_state.sensor_data = pd.read_csv(uploaded_file)

# Auto-run every 10s
if st.session_state.sensor_data is not None and st.session_state.auto_mode:
    now = time.time()
    if now - st.session_state.last_update_time >= 10:
        if st.session_state.sensor_index < len(st.session_state.sensor_data):
            user_input = pd.DataFrame([
                st.session_state.sensor_data.iloc[st.session_state.sensor_index]
            ])
            st.session_state.sensor_index += 1
            st.session_state.last_update_time = now
            st.session_state.auto_predict = True

# Manual step through
if st.sidebar.button("➡️ Load Next Sample"):
    if st.session_state.sensor_data is not None and st.session_state.sensor_index < len(st.session_state.sensor_data):
        user_input = pd.DataFrame([
            st.session_state.sensor_data.iloc[st.session_state.sensor_index]
        ])
        st.session_state.sensor_index += 1
        st.session_state.auto_predict = True

# 📄 PDF generator
def create_pdf(pred, inputs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "BreathNet Prediction Report", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Prediction: {pred}", ln=True)
    pdf.cell(0, 10, f"Time: {datetime.now()}", ln=True)
    pdf.ln()
    for k, v in inputs.items():
        pdf.cell(0, 10, f"{k}: {v}", ln=True)
    fname = "BreathNet_Report.pdf"
    pdf.output(fname)
    return fname

def get_download_link(fname):
    with open(fname, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{fname}">📥 Download PDF Report</a>'

# 🧠 Advanced Explanation Engine
compound_explanations = {
    "Acetone": "Acetone is elevated in diabetes and other metabolic conditions due to abnormal fat metabolism.",
    "Ethanol": "Ethanol can appear in breath due to alcohol intake or gut fermentation in liver disease.",
    "Formaldehyde": "Formaldehyde can indicate oxidative stress and cellular damage often found in cancer.",
    "Ammonia": "Ammonia levels rise in lung diseases like COPD and asthma due to protein breakdown and inflammation.",
    "Isoprene": "Isoprene is a byproduct of cholesterol metabolism and linked to cardiovascular or liver issues.",
    "Hydrogen Sulfide": "H2S may indicate gut flora imbalance or infections, sometimes tied to liver dysfunction.",
    "Methanol": "Methanol may indicate exposure to pollutants or liver problems due to detoxification failure.",
    "Carbonyl_Index": "Carbonyl compounds represent total oxidative stress — often elevated in multiple chronic conditions."
}

def explain_prediction(voc_data):
    top_features = sorted(voc_data.items(), key=lambda x: x[1], reverse=True)[:3]
    explanation = "### 🧠 Model Explanation:\n\n"
    for feat, val in top_features:
        note = compound_explanations.get(feat, "No explanation found.")
        explanation += f"- **{feat} = {val:.3f} ppm** → {note}\n"
    return explanation

# 🧾 Prediction logging
def log_prediction(voc_dict, prediction, prob_array):
    with open("prediction_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [now] + list(voc_dict.values()) + [prediction] + [f"{p:.4f}" for p in prob_array]
        writer.writerow(row)

# ✅ Prediction time
if st.button("🔍 Predict Disease") or st.session_state.auto_predict:
    st.session_state.auto_predict = False
    user_input = user_input[expected_columns]

    probs = model.predict_proba(user_input)[0]
    pred = model.predict(user_input)[0]

    st.success(f"🧬 Predicted Disease: **{pred}**")

    # Confidence chart
    st.subheader("📊 Confidence by Disease")
    prob_df = pd.DataFrame({
        "Disease": model.classes_,
        "Confidence": probs
    }).sort_values(by="Confidence", ascending=False)
    st.bar_chart(prob_df.set_index("Disease"))

    # Feature chart
    st.subheader("🧠 Feature Impact")
    feat_impact = user_input.iloc[0].sort_values(ascending=False)
    st.bar_chart(feat_impact)

    # Advanced explanation
    st.markdown(explain_prediction(user_input.iloc[0].to_dict()))

    # Initialize log file (once)
    if st.session_state.get("log_initialized") != True:
        with open("prediction_log.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp"] + expected_columns + ["Prediction"] + [f"Conf_{cls}" for cls in model.classes_])
        st.session_state["log_initialized"] = True

    # Save to log
    log_prediction(user_input.iloc[0].to_dict(), pred, probs)

    # PDF Button
    if st.button("📥 Generate PDF Report"):
        path = create_pdf(pred, user_input.iloc[0].to_dict())
        st.markdown(get_download_link(path), unsafe_allow_html=True)

# Show inputs
st.subheader("🔬 Current VOC Input")
st.write(user_input)



