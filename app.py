import streamlit as st
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import threading
import uvicorn
import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Set the number of records to generate
num_users = 1000
num_sessions = 5000

# Define FastAPI app
app = FastAPI()

# Define a data model for incoming requests
class UserData(BaseModel):
    mouse_movements: list
    keystrokes: list
    aadharNumber: str

# Define a FastAPI route to accept data
@app.post("/predict/")
async def predict(data: UserData):
    # Extract features from the incoming data
    mouse_movements = len(data.mouse_movements)
    keyboard_inputs = len(data.keystrokes)
    time_on_page = max(m['timestamp'] for m in data.mouse_movements) - min(m['timestamp'] for m in data.mouse_movements)
    js_enabled = True  # Assuming JS is always enabled for simplicity

    # Prepare input data for the model
    input_data = np.array([[mouse_movements, keyboard_inputs, time_on_page, int(js_enabled)]])
    
    # Load the trained model (assuming it's saved as 'model.pkl')
    model = joblib.load('model.pkl')
    
    # Make a prediction
    prediction = model.predict(input_data)[0]
    
    return {"signal": prediction}

# Integrate FastAPI with Streamlit
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Start FastAPI in a separate thread
threading.Thread(target=run_fastapi).start()

# Streamlit part of the application
st.set_page_config(page_title="ML-Enhanced Passive CAPTCHA Solution", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stAlert {
        background-color: #e6f3ff;
        border: 1px solid #1f77b4;
        border-radius: 5px;
        padding: 10px;
    }
    .stMetric {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .plot-container {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("ML-Enhanced Passive CAPTCHA Solution for UIDAI")

# Generate User Data
@st.cache_data
def generate_user_data():
    return pd.DataFrame({
        'user_id': range(1, num_users + 1),
        'browser': [random.choice(['Chrome', 'Firefox', 'Safari', 'Edge']) for _ in range(num_users)],
        'operating_system': [random.choice(['Windows', 'MacOS', 'Linux', 'iOS', 'Android']) for _ in range(num_users)],
        'screen_resolution': [random.choice(['1920x1080', '1366x768', '1440x900', '2560x1440']) for _ in range(num_users)],
        'language': [random.choice(['en-US', 'es-ES', 'fr-FR', 'de-DE', 'zh-CN']) for _ in range(num_users)],
    })

# Generate Session Data
@st.cache_data
def generate_session_data(users_df):
    def generate_mouse_movements():
        return random.randint(0, 100) if random.random() < 0.9 else random.randint(500, 1000)

    def generate_keyboard_inputs():
        return random.randint(0, 50) if random.random() < 0.9 else random.randint(200, 500)

    def generate_time_on_page():
        return random.randint(5, 300) if random.random() < 0.9 else random.randint(1, 5)

    sessions = pd.DataFrame({
        'session_id': range(1, num_sessions + 1),
        'user_id': [random.choice(users_df['user_id']) for _ in range(num_sessions)],
        'timestamp': [fake.date_time_this_year() for _ in range(num_sessions)],
        'ip_address': [fake.ipv4() for _ in range(num_sessions)],
        'mouse_movements': [generate_mouse_movements() for _ in range(num_sessions)],
        'keyboard_inputs': [generate_keyboard_inputs() for _ in range(num_sessions)],
        'time_on_page': [generate_time_on_page() for _ in range(num_sessions)],
        'js_enabled': [random.choice([True, False]) for _ in range(num_sessions)],
    })
    
    sessions['is_bot'] = ((sessions['mouse_movements'] > 500) | 
                          (sessions['keyboard_inputs'] > 200) | 
                          (sessions['time_on_page'] < 5)).astype(int)
    
    return sessions

# Train ML model
@st.cache_resource
def train_model(sessions_df):
    features = ['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled']
    X = sessions_df[features]
    y = sessions_df['is_bot']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Save the model
    joblib.dump(model, 'model.pkl')
    
    return model, classification_report(y_test, y_pred)

# Main Streamlit app
def main():
    # Generate data
    users_df = generate_user_data()
    sessions_df = generate_session_data(users_df)
    
    # Train ML model
    model, classification_report_text = train_model(sessions_df)
    
    st.subheader("Live Session Classification")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mouse_movements = st.number_input("Mouse Movements", min_value=0, max_value=1000, value=50)
        keyboard_inputs = st.number_input("Keyboard Inputs", min_value=0, max_value=500, value=20)
    
    with col2:
        time_on_page = st.number_input("Time on Page (seconds)", min_value=0, max_value=600, value=60)
        js_enabled = True  # Default value
    
    if st.button("Classify Session"):
        input_data = np.array([[mouse_movements, keyboard_inputs, time_on_page, int(js_enabled)]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        result_color = "#2ca02c" if prediction == 0 else "#d62728"
        st.markdown(f"""
        <div style='background-color: {result_color}; color: white; padding: 10px; border-radius: 5px;'>
        <h3>Prediction: {'Human' if prediction == 0 else 'Bot'}</h3>
        <p>Probability of being a bot: {probability:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
