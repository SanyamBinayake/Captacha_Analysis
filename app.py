import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the pre-trained model (ensure you have trained and saved it before running this)
model = joblib.load('path_to_your_model.joblib')

# Set up Streamlit app
st.title("Live Session Classification")

# Input fields
col1, col2 = st.columns(2)

with col1:
    mouse_movements = st.number_input("Mouse Movements", min_value=0, max_value=1000, value=50)
    keyboard_inputs = st.number_input("Keyboard Inputs", min_value=0, max_value=500, value=20)

with col2:
    time_on_page = st.number_input("Time on Page (seconds)", min_value=0, max_value=600, value=60)

# Button to classify session
if st.button("Classify Session"):
    input_data = np.array([[mouse_movements, keyboard_inputs, time_on_page]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    result_color = "#2ca02c" if prediction == 0 else "#d62728"
    st.markdown(f"""
    <div style='background-color: {result_color}; color: white; padding: 10px; border-radius: 5px;'>
    <h3>Prediction: {'Human' if prediction == 0 else 'Bot'}</h3>
    <p>Probability of being a bot: {probability:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='stAlert'>
    <strong>Interpretation:</strong>
    <ul>
    <li>This tool allows you to input session data and see whether our model classifies it as a bot or human session.</li>
    <li>The probability gives an idea of how confident the model is in its prediction.</li>
    <li>Experimenting with different input values can help understand the model's decision boundaries.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
