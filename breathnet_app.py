# ✅ BreathNet Full Streamlit App (100% FIXED for Streamlit Cloud with Auto-Run)

import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
import base64
import time

# 🧠 Load AI Model
model = joblib.load("breathnet_model.pkl")

# 🌐 UI Setup
st.set_page_config(page_title="BreathNet", layout="centered")
st.title("🫁 BreathNet: AI-Powered Disease Prediction from VOCs")

# 📂 Upload CSV
st.sidebar.markdown("---")
st.sidebar.header("📂 Upload VOC CSV (Simulated Sensor)")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# 🌀 Initialize session state
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

# 🔁 Auto-run checkbox
st.sidebar.markdown("---")
st.session_state.auto_mode = st.sidebar.checkbox("🔁 Auto-Run Every 10 Seconds")

# 🧪 Manual input fallback
st.sidebar.header("Manual VOC Input (Optional)")
manual_input = {
    "Acetone": st.sidebar.slider("Acetone", 0.0, 2.0, 1.0, 0.01),
    "Ethanol": st.sidebar.slider("Ethanol", 0.0, 0.5, 0.2, 0.01),
    "Formaldehyde": st.sidebar.slider("Formaldehyde", 0.0, 0.1, 0.03, 0.001),
    "Ammonia": st.sidebar.slider("Ammonia", 0.0, 0.1, 0.03, 0.001),
    "Isoprene": st.sidebar.slider("Isoprene", 0.0, 1.5, 0.8, 0.01),
    "Hydrogen Sulfide": st.sidebar.slider("Hydrogen Sulfide", 0.0, 0.1, 0.03, 0.001),
    "Methanol": st.sidebar.slider("Methanol", 0.0, 0.1, 0.03, 0.001),
    "Carbonyl_Index": st.sidebar.slider("Carbonyl Index", 0.0, 0.3, 0.1, 0.01),
}
user_input = pd.DataFrame([manual_input])

# 📥 Load CSV if uploaded
if uploaded_file:
    st.session_state.sensor_data = pd.read_csv(uploaded_file)

# 🔁 Auto-run logic
if st.session_state.sensor_data is not None and st.session_state.auto_mode:
    current_time = time.time()
    if current_time - st.session_state.last_update_time >= 10:
        if st.session_state.sensor_index < len(st.session_state.sensor_data):
            user_input = pd.DataFrame([
                st.session_state.sensor_data.iloc[st.session_state.sensor_index]
            ])
            st.session_state.sensor_index += 1
            st.session_state.last_update_time = current_time
            st.session_state.auto_predict = True

# ➡️ Manual next row
if st.sidebar.button("➡️ Load Next Sample"):
    if st.session_state.sensor_data is not None and st.session_state.sensor_index < len(st.session_state.sensor_data):
        user_input = pd.DataFrame([
            st.session_state.sensor_data.iloc[st.session_state.sensor_index]
        ])
        st.session_state.sensor_index += 1
        st.session_state.auto_predict = True

# 🔍 Prediction logic
if st.button("🔍 Predict Disease") or st.session_state.auto_predict:
    st.session_state.auto_predict = False
    probs = model.predict_proba(user_input)[0]
    pred = model.predict(user_input)[0]

    st.success(f"🧬 Predicted Disease: **{pred}**")
    st.subheader("📊 Confidence")
    prob_df = pd.DataFrame({ "Disease": model.classes_, "Confidence": probs })
    st.bar_chart(prob_df.set_index("Disease"))

    st.subheader("🧠 Feature Impact")
    st.bar_chart(user_input.iloc[0].sort_values(ascending=False))

    top_feat = user_input.iloc[0].sort_values(ascending=False).index[0]
    st.info(f"ℹ️ Most influencing VOC: **{top_feat}**")

    # PDF generation
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

    if st.button("📥 Generate PDF Report"):
        path = create_pdf(pred, user_input.iloc[0].to_dict())
        st.markdown(get_download_link(path), unsafe_allow_html=True)

# 🔎 Show current VOCs
st.subheader("🔬 Current VOC Input")
st.write(user_input)

