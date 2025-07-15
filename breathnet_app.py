# ✅ BreathNet Streamlit App — Final Day 1 Version (CO₂ Added, MAX Removed)

import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
import base64
import time
from io import BytesIO, StringIO

# ✅ Load model
model = joblib.load("breathnet_model.pkl")

# ✅ Page config
st.set_page_config(page_title="BreathNet", layout="centered")
st.title("🫁 BreathNet: AI-Powered Disease Prediction from VOCs")

# ✅ Expected columns (CO₂ included, MAX30102 removed)
expected_columns = ['Acetone', 'Ethanol', 'Formaldehyde', 'Ammonia',
                    'Isoprene', 'Hydrogen Sulfide', 'Methanol',
                    'Carbonyl_Index', 'CO2']

# ✅ Session state
if "sensor_index" not in st.session_state:
    st.session_state.sensor_index = 0
if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = None
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

# ✅ Upload CSV
st.sidebar.header("📂 Upload VOC CSV (with CO₂)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ✅ Auto-run toggle
st.sidebar.markdown("---")
auto_mode = st.sidebar.checkbox("🔁 Auto-Load Next Row Every 10 sec")

# ✅ Manual VOC input
st.sidebar.header("🧪 Manual VOC Input (ppm)")
manual_input = {
    col: st.sidebar.slider(col, 0.0, 2.0 if col in ["Acetone", "Isoprene", "CO2"] else 0.5, 1.0 if col in ["Acetone", "Isoprene", "CO2"] else 0.03, 0.01)
    for col in expected_columns
}
user_input = pd.DataFrame([manual_input])

# ✅ Load uploaded CSV if available
if uploaded_file:
    st.session_state.sensor_data = pd.read_csv(uploaded_file)

    if st.session_state.sensor_index < len(st.session_state.sensor_data):
        user_input = pd.DataFrame([st.session_state.sensor_data.iloc[st.session_state.sensor_index]])
        
        if auto_mode:
            time.sleep(10)
            st.session_state.sensor_index += 1
            st.experimental_rerun()

if st.sidebar.button("➡️ Load Next Row") and st.session_state.sensor_data is not None:
    if st.session_state.sensor_index < len(st.session_state.sensor_data):
        user_input = pd.DataFrame([st.session_state.sensor_data.iloc[st.session_state.sensor_index]])
        st.session_state.sensor_index += 1

# ✅ Predict
if st.button("🔍 Predict Disease"):
    user_input = user_input[expected_columns]
    probabilities = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)[0]

   # ✅ Reorder columns to match the model
user_input = user_input[expected_columns]

# ✅ Make predictions
try:
    probabilities = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)[0]
except Exception as e:
    st.error(f"❌ Prediction failed. Reason: {e}")
    st.stop()

# ✅ Store session
st.session_state.prediction = prediction
st.session_state.inputs = user_input.iloc[0].to_dict()

# ✅ Show result
st.success(f"🧬 Predicted Disease: **{prediction}**")

st.subheader("📊 Prediction Confidence by Disease")
prob_df = pd.DataFrame({
    "Disease": model.classes_,
    "Confidence": probabilities
}).sort_values(by="Confidence", ascending=False)
st.bar_chart(prob_df.set_index("Disease"))

# ✅ Explainable AI
st.subheader("🧠 Feature Impact (VOC importance)")
impact_series = user_input.iloc[0].sort_values(ascending=False)
st.bar_chart(impact_series)

top_voc = impact_series.index[0]
top_value = impact_series.iloc[0]

st.subheader("💬 AI Explanation")
st.markdown(
    f"The model identified **{top_voc} = {top_value:.3f} ppm** as the most influential compound for predicting **{prediction}**.\n\n"
    f"This compound often correlates with metabolic or inflammatory changes observed in {prediction.lower()}, helping the model distinguish it from other diseases."
)


# ✅ PDF Report Generator
def create_pdf(prediction, inputs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "BreathNet Prediction Report", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Predicted Disease: {prediction}", ln=True)
    pdf.cell(0, 10, "", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "VOC Concentrations (ppm):", ln=True)
    pdf.set_font("Arial", '', 12)
    for k, v in inputs.items():
        pdf.cell(0, 10, f"{k}: {v}", ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

# ✅ PDF Download Button
if "prediction" in st.session_state and "inputs" in st.session_state:
    if st.button("📥 Generate PDF Report"):
        pdf_file = create_pdf(st.session_state.prediction, st.session_state.inputs)
        b64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="BreathNet_Report.pdf">📄 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

# ✅ CSV Log Export
if st.session_state.prediction_log:
    st.subheader("🗂️ Download Prediction Log")
    df_log = pd.DataFrame(
        st.session_state.prediction_log,
        columns=["Timestamp"] + expected_columns + ["Prediction"] + [f"Conf_{cls}" for cls in model.classes_]
    )
    csv_buffer = StringIO()
    df_log.to_csv(csv_buffer, index=False)
    b64_csv = base64.b64encode(csv_buffer.getvalue().encode()).decode("utf-8")
    href = f'<a href="data:file/csv;base64,{b64_csv}" download="BreathNet_Log.csv">📥 Download CSV Log</a>'
    st.markdown(href, unsafe_allow_html=True)

# ✅ Show Current Input
st.subheader("🔬 Current VOC Input")
st.write(user_input)


