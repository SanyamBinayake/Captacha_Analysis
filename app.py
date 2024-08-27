import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import random
from faker import Faker

# Initialize Faker
fake = Faker()
Faker.seed(0)

# Set page config
st.set_page_config(page_title="Live Session Classification", layout="wide")

# Generate sample data for training
def generate_sample_data():
    np.random.seed(42)
    num_samples = 1000
    
    data = {
        'mouse_movements': np.random.randint(0, 1000, num_samples),
        'keyboard_inputs': np.random.randint(0, 500, num_samples),
        'time_on_page': np.random.randint(0, 600, num_samples),
        'js_enabled': np.random.choice([0, 1], num_samples),
        'cookie_enabled': np.random.choice([0, 1], num_samples),
        'is_bot': np.random.choice([0, 1], num_samples)
    }
    
    df = pd.DataFrame(data)
    return df

# Train a RandomForest model
def train_model(df):
    features = ['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'cookie_enabled']
    X = df[features]
    y = df['is_bot']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save the model
    joblib.dump(model, 'model.joblib')
    
    return model, report

# Main app
def main():
    st.title("Live Session Classification")

    # Load or train the model
    try:
        model = joblib.load('model.joblib')
    except FileNotFoundError:
        st.write("Pre-trained model not found. Training a new model...")
        df = generate_sample_data()
        model, report = train_model(df)
        st.write("Model trained and saved.")
        st.code(report, language='text')
    else:
        st.write("Pre-trained model loaded.")

    st.subheader("Classify a Live Session")

    col1, col2, col3 = st.columns(3)
    with col1:
        mouse_movements = st.number_input("Mouse Movements", min_value=0, max_value=1000, value=50)
        keyboard_inputs = st.number_input("Keyboard Inputs", min_value=0, max_value=500, value=20)
    with col2:
        time_on_page = st.number_input("Time on Page (seconds)", min_value=0, max_value=600, value=60)
        js_enabled = st.checkbox("JavaScript Enabled", value=True)
    with col3:
        cookie_enabled = st.checkbox("Cookies Enabled", value=True)

    if st.button("Classify Session"):
        input_data = np.array([[mouse_movements, keyboard_inputs, time_on_page, 
                                int(js_enabled), int(cookie_enabled)]])
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

if __name__ == "__main__":
    main()
