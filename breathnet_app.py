# ✅ BreathNet Streamlit App (Ultimate Fixed Version)

import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
import base64
import time

# Load AI model
model = joblib.load("breathnet_model.pkl")

# Setup
st.set_page_config(page_title="BreathNet", layout="centered")
st.title("🫁 BreathNet: AI-Powered Disease Prediction from VOCs")

# Upload CSV
st.sidebar.markdown("---")
st.sidebar.header("📂 Upload VOC CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

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

# Expected feature order
expected_columns = ['Acetone', 'Ethanol', 'Formaldehyde', 'Ammonia',
                    'Isoprene', 'Hydrogen Sulfide', 'Methanol', 'Carbonyl_Index']

# Auto-run toggle
st.sidebar.markdown("---")
st.session_state.auto_mode = st.sidebar.checkbox("🔁 Auto-Run Every 10 Seconds")

# Sliders
st.sidebar.header("Manual VOC Input")
manual_input = {col: st.sidebar.slider(col, 0.0, 2.0 if col == "Acetone" else 0.5, 1.0 if col == "Acetone" else 0.03, 0.01)
                for col in expected_columns}
user_input = pd.DataFrame([manual_input])

# Load uploaded data
if uploaded_file:
    st.session_state.sensor_data = pd.read_csv(uploaded_file)

# Auto-run next row every 10 sec
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

# Manual next row
if st.sidebar.button("➡️ Load Next Sample"):
    if st.session_state.sensor_data is not None and st.session_state.sensor_index < len(st.session_state.sensor_data):
        user_input = pd.DataFrame([
            st.session_state.sensor_data.iloc[st.session_state.sensor_index]
        ])
        st.session_state.sensor_index += 1
        st.session_state.auto_predict = True

# Prediction block
if st.button("🔍 Predict Disease") or st.session_state.auto_predict:
    st.session_state.auto_predict = False

    # ✅ Column alignment fix
    user_input = user_input[expected_columns]

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

# Show current input
st.subheader("🔬 Current VOC Input")
st.write(user_input)


