import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import random

# Set page config
st.set_page_config(page_title="ML-Enhanced Passive CAPTCHA Solution", layout="wide")

# Set a consistent color palette
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Custom CSS
st.markdown("""
    <style>
        /* General app styles */
        body {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4 {
            color: black;
        }
        p, li, strong {
            color: black;
        }

        /* Specific styles for custom elements */
        .custom-info {
            background-color: #E6F3FF;
            padding: 20px;
            border-radius: 10px;
        }
        .custom-info h3 {
            color: #1f77b4;
        }
        .stAlert {
            background-color: transparent;
            border: 1px solid #1f77b4;
            border-radius: 5px;
            padding: 10px;
            color: black;
        }
        .stMetric, .plot-container, .your-custom-class {
            background-color: transparent;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .section-title {
            font-size: 24px;
            color: #1f77b4;
            margin-bottom: 10px;
        }
        .subsection-title {
            font-size: 20px;
            color: #ff7f0e;
            margin-bottom: 10px;
        }
        .note {
            border: 1px solid #1f77b4;
            border-radius: 5px;
            padding: 10px;
            color: black;
            background-color: transparent;
        }
    </style>
""", unsafe_allow_html=True)

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
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Overview", "User Profiles", "Session Analysis", "ML Insights"])

    with tab0:
        st.header("Problem Statement")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                """
                <div class='custom-info'>
                <h3>Quick Info</h3>
                <p><strong>ID:</strong> 1672</p>
                <p><strong>Organization:</strong> Ministry of Electronics and Information Technology</p>
                <p><strong>Department:</strong> Co-ordination Division</p>
                <p><strong>Category:</strong> Software</p>
                <p><strong>Theme:</strong> Smart Automation</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div class='problem-statement'>
                <h3 class='subsection-title'>Develop a ML Model based solution to refine CAPTCHA</h3>
                <h4 class='subsection-title'>Background:</h4>
                <p>UIDAI aims to remove traditional CAPTCHA from its portals to improve user experience. Instead, a passive solution is needed to differentiate between bots and human users.</p>
                <h4 class='subsection-title'>Key Requirements:</h4>
                <ul>
                <li>Develop a passive approach using environmental parameters and AI/ML.</li>
                <li>Capture browser context and analyze with backend ML models.</li>
                <li>Protect backend APIs from DoS/DDoS vulnerabilities.</li>
                <li>Minimize human interaction for better user experience.</li>
                <li>Ensure compliance with UIDAI's privacy policies.</li>
                </ul>
                <h4 class='subsection-title'>Expected Solution:</h4>
                <p>A complete solution with frontend and backend design, corresponding code, and ML model to demonstrate the passive CAPTCHA approach.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab1:
        st.header("Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users_df))
        with col2:
            st.metric("Total Sessions", len(sessions_df))
        with col3:
            st.metric("Human Sessions", len(sessions_df[sessions_df['is_bot'] == 0]))
        with col4:
            st.metric("Bot Sessions", len(sessions_df[sessions_df['is_bot'] == 1]))
        
        st.markdown("---")
        
        # Sessions over time
        fig_sessions = px.line(sessions_df.groupby(sessions_df['timestamp'].dt.date).size().reset_index(name='count'), 
                               x='timestamp', y='count', title="Sessions Over Time",
                               labels={'timestamp': 'Date', 'count': 'Number of Sessions'})
        fig_sessions.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', title_font=dict(size=24, color='black'))
        st.plotly_chart(fig_sessions, use_container_width=True)
    
    with tab2:
        st.header("User Profiles")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_browser = px.pie(users_df, names='browser', title="Browsers Used")
            fig_browser.update_layout(legend_title_text='Browser', legend_font=dict(size=12, color='black'),
                                      plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', title_font=dict(size=24, color='black'))
            st.plotly_chart(fig_browser, use_container_width=True)
        
        with col2:
            fig_os = px.pie(users_df, names='operating_system', title="Operating Systems Used")
            fig_os.update_layout(legend_title_text='Operating System', legend_font=dict(size=12, color='black'),
                                 plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', title_font=dict(size=24, color='black'))
            st.plotly_chart(fig_os, use_container_width=True)
    
    with tab3:
        st.header("Session Analysis")
        
        fig_mouse = px.histogram(sessions_df, x='mouse_movements', nbins=30, color='is_bot', 
                                 title="Mouse Movements by User Type", color_discrete_map={0: 'blue', 1: 'red'},
                                 labels={'mouse_movements': 'Number of Mouse Movements', 'is_bot': 'Is Bot'})
        fig_mouse.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', title_font=dict(size=24, color='black'))
        st.plotly_chart(fig_mouse, use_container_width=True)
    
        fig_keyboard = px.histogram(sessions_df, x='keyboard_inputs', nbins=30, color='is_bot', 
                                    title="Keyboard Inputs by User Type", color_discrete_map={0: 'blue', 1: 'red'},
                                    labels={'keyboard_inputs': 'Number of Keyboard Inputs', 'is_bot': 'Is Bot'})
        fig_keyboard.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', title_font=dict(size=24, color='black'))
        st.plotly_chart(fig_keyboard, use_container_width=True)

    with tab4:
        st.header("ML Insights")
        st.text("Model Classification Report")
        st.text(classification_report)
        
        st.markdown("---")
        
        st.markdown(
            """
            <div class='note'>
            <strong>Note:</strong> The above metrics are based on the simulated data. In a real-world scenario, data collection would involve 
            gathering real-time interactions from users across various contexts.
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
