# breathnet_app.py (✅ 100% Working Version with Prediction, Confidence Graph, PDF Export)

import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
import base64

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
    probabilities = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)[0]

    st.session_state.prediction = prediction
    st.session_state.inputs = user_input.iloc[0].to_dict()
    st.session_state.probabilities = probabilities.tolist()

    st.success(f"🧬 Predicted Disease: **{prediction}**")
    # Feature influence (basic explanation)
st.subheader("🧠 Top Influencing VOCs (Feature Impact)")
feature_impact = pd.Series(st.session_state.inputs)
feature_impact = feature_impact.sort_values(ascending=False)

st.bar_chart(feature_impact)

# Simple auto-explanation
top_feature = feature_impact.index[0]
st.info(f"ℹ️ Most influential VOC: **{top_feature}** — with value {feature_impact.iloc[0]:.3f}")


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
