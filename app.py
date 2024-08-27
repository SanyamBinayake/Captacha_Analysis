import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import random

# Set page config
st.set_page_config(page_title="ML-Enhanced Passive CAPTCHA Solution", layout="wide")

# Set a consistent color palette
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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

# Initialize Faker
fake = Faker()
Faker.seed(0)

# Set the number of records to generate
num_users = 1000
num_sessions = 5000



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
    })
    
    sessions['is_bot'] = ((sessions['mouse_movements'] > 500) | 
                          (sessions['keyboard_inputs'] > 200) | 
                          (sessions['time_on_page'] < 5)).astype(int)
    
    return sessions

# Train ML model
@st.cache_resource
def train_model(sessions_df):
    features = ['mouse_movements', 'keyboard_inputs', 'time_on_page']
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



    # Sidebar
    st.sidebar.title("Settings")
    date_range = st.sidebar.date_input("Select Date Range", 
                                       [pd.Timestamp.now() - pd.Timedelta(days=30), pd.Timestamp.now()],
                                       min_value=pd.Timestamp.now() - pd.Timedelta(days=365),
                                       max_value=pd.Timestamp.now())
    
    # Generate data
    users_df = generate_user_data()
    sessions_df = generate_session_data(users_df)
    
    # Train ML model
    model, classification_report = train_model(sessions_df)
    
    # Filter data based on sidebar inputs
    sessions_df = sessions_df[(sessions_df['timestamp'].dt.date >= date_range[0]) & (sessions_df['timestamp'].dt.date <= date_range[1])]

    # Main content
    # tab0, tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Overview", "User Profiles", "Session Analysis", "ML Insights"])

   

    with tab4:
        st.header("Machine Learning Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            st.code(classification_report, language='text')
        
        with col2:
            st.markdown("""
            <div class='stAlert'>
            <strong>Interpretation:</strong>
            <ul>
            <li>High precision reduces false positives, ensuring we don't wrongly label human users as bots.</li>
            <li>High recall ensures we're catching most of the actual bot sessions.</li>
            <li>The F1-score balances precision and recall, giving an overall measure of the model's performance.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': ['mouse_movements', 'keyboard_inputs', 'time_on_page'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title="Feature Importance for Bot Detection",
                                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                                color='importance',
                                color_continuous_scale=px.colors.sequential.Viridis)
        fig_importance.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_importance, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        <div class='stAlert'>
        <strong>Insights:</strong>
        <ul>
        <li>Features with high importance are the most crucial for distinguishing between bots and humans.</li>
        <li>This information can guide further refinement of the passive CAPTCHA system, focusing on the most relevant features.</li>
        <li>Less important features might be candidates for removal to simplify the model and improve performance.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Live Session Classification")
        col1, col2, col3 = st.columns(3)
        with col1:
            mouse_movements = st.number_input("Mouse Movements", min_value=0, max_value=1000, value=50)
            keyboard_inputs = st.number_input("Keyboard Inputs", min_value=0, max_value=500, value=20)
        with col2:
            time_on_page = st.number_input("Time on Page (seconds)", min_value=0, max_value=600, value=60)
           
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

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This demo app showcases how to use machine learning for a passive CAPTCHA solution to differentiate between bots and human users. 
    
    In a real-world scenario, such a system would need to be thoroughly tested, regularly updated, and integrated with other security measures to ensure robust protection against bot attacks while maintaining a smooth user experience.
    """)

if __name__ == "__main__":
    main()
