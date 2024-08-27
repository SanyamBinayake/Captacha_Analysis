import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Set page config
st.set_page_config(page_title="Session Analysis", layout="wide")

# Set a consistent color palette
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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
    st.title("Session Analysis")

    # Generate data
    users_df = generate_user_data()
    sessions_df = generate_session_data(users_df)
    
    # Train ML model
    model, classification_report = train_model(sessions_df)

    # Main content
    tab3 = st.container()

    with tab3:
        st.header("Session Analysis")
        
        # Mouse movements vs Keyboard inputs
        fig_mouse_keyboard = px.scatter(sessions_df, x='mouse_movements', y='keyboard_inputs', 
                                        color='is_bot', title="Mouse Movements vs Keyboard Inputs",
                                        labels={'mouse_movements': 'Mouse Movements', 'keyboard_inputs': 'Keyboard Inputs'},
                                        color_discrete_map={0: color_palette[0], 1: color_palette[1]})
        fig_mouse_keyboard.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_mouse_keyboard, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        <div class='stAlert'>
        <strong>Insights:</strong>
        <ul>
        <li>Human users typically show a balance between mouse movements and keyboard inputs.</li>
        <li>Bots might show unusual patterns, such as very high mouse movements with low keyboard inputs or vice versa.</li>
        <li>Clusters in this plot can help identify different types of bot behavior.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Time on page distribution
        fig_time = px.histogram(sessions_df, x='time_on_page', color='is_bot', 
                                title="Time on Page Distribution",
                                labels={'time_on_page': 'Time on Page (seconds)', 'count': 'Number of Sessions'},
                                color_discrete_map={0: color_palette[2], 1: color_palette[3]},
                                barmode='overlay')
        fig_time.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_time, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        <div class='stAlert'>
        <strong>Insights:</strong>
        <ul>
        <li>Human users typically spend varying amounts of time on a page, often following a normal distribution.</li>
        <li>Bots might show very short page times or unusually long times.</li>
        <li>This information can be crucial for setting thresholds in the passive CAPTCHA system.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Session Data Sample")
        st.dataframe(sessions_df.head(10))

if __name__ == "__main__":
    main()
