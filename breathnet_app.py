# breathnet_app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained AI model
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

# Create DataFrame
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

# 🧠 Store prediction + inputs in session state
if st.button("🔍 Predict Disease"):
    probabilities = model.predict_proba(user_input)[0]
    prediction = model.predict(user_input)[0]

    # Store results
    st.session_state.prediction = prediction
    st.session_state.inputs = user_input.iloc[0].to_dict()

    # Show results
    st.success(f"🧬 Predicted Disease: **{prediction}**")

    # Confidence chart
    st.subheader("📊 Prediction Confidence by Disease")
    prob_df = pd.DataFrame({
        'Disease': model.classes_,
        'Confidence': probabilities
    }).sort_values(by='Confidence', ascending=False)

    st.bar_chart(prob_df.set_index('Disease'))


    # Show prediction
    st.success(f"🧬 Predicted Disease: **{prediction}**")

    if 'prediction' in st.session_state and 'inputs' in st.session_state:
    if st.button("📥 Generate PDF Report"):
        file_path = create_pdf(st.session_state.prediction, st.session_state.inputs)
        st.markdown(get_pdf_download_link(file_path), unsafe_allow_html=True)
else:
    st.info("ℹ️ Please run a prediction first.")



    # Plot confidence graph
    st.subheader("📊 Prediction Confidence by Disease")
    prob_df = pd.DataFrame({
        'Disease': model.classes_,
        'Confidence': probabilities
    }).sort_values(by='Confidence', ascending=False)

    st.bar_chart(prob_df.set_index('Disease'))

# Show data
st.subheader("🔬 Current Input Data")
st.write(user_input)
