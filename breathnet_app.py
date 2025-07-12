import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
import base64
import time
from io import BytesIO, StringIO

# ✅ Load trained model
model = joblib.load("breathnet_model.pkl")

# ✅ Extended feature set
expected_columns = ['Acetone', 'Ethanol', 'Formaldehyde', 'Ammonia',
                    'Isoprene', 'Hydrogen Sulfide', 'Methanol', 'Carbonyl_Index',
                    'MQ135_Value', 'Heart_Rate', 'Oxygen_Saturation']

# ✅ Page config
st.set_page_config(page_title="BreathNet", layout="centered")
st.title("🫁 BreathNet: AI-Powered Disease Prediction from VOCs + Bio Signals")

# ✅ Session state
if "sensor_index" not in st.session_state:
    st.session_state.sensor_index = 0
if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = None
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

# ✅ Upload CSV
st.sidebar.header("📂 Upload VOC CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ✅ Auto-run
st.sidebar.markdown("---")
auto_mode = st.sidebar.checkbox("🔁 Auto-Load Next Row Every 10 sec")

# ✅ Manual input sliders
st.sidebar.header("🧪 Manual Input (ppm + Bio)")
manual_input = {
    "Acetone": st.sidebar.slider("Acetone", 0.0, 2.0, 1.0, 0.01),
    "Ethanol": st.sidebar.slider("Ethanol", 0.0, 0.5, 0.2, 0.01),
    "Formaldehyde": st.sidebar.slider("Formaldehyde", 0.0, 0.1, 0.03, 0.001),
    "Ammonia": st.sidebar.slider("Ammonia", 0.0, 0.1, 0.03, 0.001),
    "Isoprene": st.sidebar.slider("Isoprene", 0.0, 1.5, 0.8, 0.01),
    "Hydrogen Sulfide": st.sidebar.slider("Hydrogen Sulfide", 0.0, 0.1, 0.03, 0.001),
    "Methanol": st.sidebar.slider("Methanol", 0.0, 0.1, 0.03, 0.001),
    "Carbonyl_Index": st.sidebar.slider("Carbonyl Index", 0.0, 0.3, 0.1, 0.01),
    "MQ135_Value": st.sidebar.slider("MQ135 VOC Index", 0.0, 1000.0, 300.0, 1.0),
    "Heart_Rate": st.sidebar.slider("Heart Rate (BPM)", 40, 160, 80, 1),
    "Oxygen_Saturation": st.sidebar.slider("SpO₂ (%)", 85, 100, 98, 1)
}
user_input = pd.DataFrame([manual_input])

# ✅ Load CSV if uploaded
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

# ✅ Prediction
if st.button("🔍 Predict Disease"):
    user_input = user_input[expected_columns]
    probabilities = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)[0]

    # ✅ Store results
    st.session_state.prediction = prediction
    st.session_state.inputs = user_input.iloc[0].to_dict()

    # ✅ Logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp] + list(user_input.iloc[0].values) + [prediction] + [f"{p:.4f}" for p in probabilities]
    st.session_state.prediction_log.append(row)

    # ✅ Results
    st.success(f"🧬 Predicted Disease: **{prediction}**")

    st.subheader("📊 Prediction Confidence")
    prob_df = pd.DataFrame({
        "Disease": model.classes_,
        "Confidence": probabilities
    }).sort_values(by="Confidence", ascending=False)
    st.bar_chart(prob_df.set_index("Disease"))

    # ✅ XAI Explanation
    st.subheader("🧠 Feature Impact")
    impact_series = user_input.iloc[0].sort_values(ascending=False)
    st.bar_chart(impact_series)

    top_voc = impact_series.index[0]
    top_value = impact_series.iloc[0]

    st.subheader("💬 AI Explanation")
    st.markdown(
        f"The model identified **{top_voc} = {top_value:.3f}** as the most influential feature leading to prediction of **{prediction}**."
    )

# ✅ PDF Export
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
    pdf.cell(0, 10, "VOC + Bio Signal Inputs:", ln=True)

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

# ✅ CSV Export
if st.session_state.prediction_log:
    st.subheader("📁 Prediction Log")
    df_log = pd.DataFrame(
        st.session_state.prediction_log,
        columns=["Timestamp"] + expected_columns + ["Prediction"] + [f"Conf_{cls}" for cls in model.classes_]
    )
    csv_buffer = StringIO()
    df_log.to_csv(csv_buffer, index=False)
    b64_csv = base64.b64encode(csv_buffer.getvalue().encode()).decode("utf-8")
    href = f'<a href="data:file/csv;base64,{b64_csv}" download="BreathNet_Log.csv">📥 Download CSV Log</a>'
    st.markdown(href, unsafe_allow_html=True)

# ✅ Show current input
st.subheader("🔬 Current VOC + Bio Input")
st.write(user_input)

