import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
from faker import Faker

# Set page config
st.set_page_config(page_title="ML-Enhanced Passive CAPTCHA Solution", layout="wide")

# Initialize Faker
fake = Faker()
Faker.seed(0)

# Set the number of records to generate
num_users = 1000
num_sessions = 5000

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
        'cookie_enabled': [random.choice([True, False]) for _ in range(num_sessions)],
    })
    
    sessions['is_bot'] = ((sessions['mouse_movements'] > 500) | 
                          (sessions['keyboard_inputs'] > 200) | 
                          (sessions['time_on_page'] < 5)).astype(int)
    
    return sessions

# Train ML model
@st.cache_resource
def train_model(sessions_df):
    features = ['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'cookie_enabled']
    X = sessions_df[features]
    y = sessions_df['is_bot']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return model, classification_report(y_test, y_pred)

# Main app
def main():
    st.title("ML-Enhanced Passive CAPTCHA Solution for UIDAI")

    # Generate data
    users_df = generate_user_data()
    sessions_df = generate_session_data(users_df)
    
    # Train ML model
    model, classification_report_text = train_model(sessions_df)
    
    st.header("Machine Learning Insights")
    
    col1, col2 = st.columns(2)
    
    # with col1:
    #     st.subheader("Model Performance")
    #     st.code(classification_report_text, language='text')
    
    # with col2:
    #     st.markdown("""
    #     <div class='stAlert'>
    #     <strong>Interpretation:</strong>
    #     <ul>
    #     <li>High precision reduces false positives, ensuring we don't wrongly label human users as bots.</li>
    #     <li>High recall ensures we're catching most of the actual bot sessions.</li>
    #     <li>The F1-score balances precision and recall, giving an overall measure of the model's performance.</li>
    #     </ul>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # st.subheader("Feature Importance")
    # feature_importance = pd.DataFrame({
    #     'feature': ['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'cookie_enabled'],
    #     'importance': model.feature_importances_
    # }).sort_values('importance', ascending=False)
    
    # fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
    #                         title="Feature Importance for Bot Detection",
    #                         labels={'importance': 'Importance Score', 'feature': 'Feature'},
    #                         color='importance',
    #                         color_continuous_scale=px.colors.sequential.Viridis)
    # fig_importance.update_layout(plot_bgcolor='white')
    # st.plotly_chart(fig_importance, use_container_width=True, config={'displayModeBar': False})
    
    # st.markdown("""
    # <div class='stAlert'>
    # <strong>Insights:</strong>
    # <ul>
    # <li>Features with high importance are the most crucial for distinguishing between bots and humans.</li>
    # <li>This information can guide further refinement of the passive CAPTCHA system, focusing on the most relevant features.</li>
    # <li>Less important features might be candidates for removal to simplify the model and improve performance.</li>
    # </ul>
    # </div>
    # """, unsafe_allow_html=True)

    # st.subheader("Live Session Classification")
    # col1, col2, col3 = st.columns(3)
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
