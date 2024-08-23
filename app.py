import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

# Generate User Data
@st.cache
def generate_user_data():
    return pd.DataFrame({
        'user_id': range(1, num_users + 1),
        'browser': [random.choice(['Chrome', 'Firefox', 'Safari', 'Edge']) for _ in range(num_users)],
        'operating_system': [random.choice(['Windows', 'MacOS', 'Linux', 'iOS', 'Android']) for _ in range(num_users)],
        'screen_resolution': [random.choice(['1920x1080', '1366x768', '1440x900', '2560x1440']) for _ in range(num_users)],
        'language': [random.choice(['en-US', 'es-ES', 'fr-FR', 'de-DE', 'zh-CN']) for _ in range(num_users)],
    })

# Generate Session Data
@st.cache
def generate_session_data(users_df):
    def generate_mouse_movements():
        return random.randint(0, 100) if random.random() < 0.9 else random.randint(500, 1000)

    def generate_keyboard_inputs():
        return random.randint(0, 50) if random.random() < 0.9 else random.randint(200, 500)

    def generate_time_on_page():
        return random.randint(5, 300) if random.random() < 0.9 else random.randint(1, 5)

    def generate_zoom_level():
        return random.choice([100, 125, 150, 175, 200])

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
        'zoom_level': [generate_zoom_level() for _ in range(num_sessions)],
    })
    
    sessions['is_bot'] = ((sessions['mouse_movements'] > 500) | 
                          (sessions['keyboard_inputs'] > 200) | 
                          (sessions['time_on_page'] < 5)).astype(int)
    
    return sessions

# Train ML model
@st.cache
def train_model(sessions_df):
    features = ['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'cookie_enabled', 'zoom_level']
    X = sessions_df[features]
    y = sessions_df['is_bot']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return model, report

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
    model, classification_report_text = train_model(sessions_df)
    
    # Filter data based on sidebar inputs
    sessions_df = sessions_df[(sessions_df['timestamp'].dt.date >= date_range[0]) & (sessions_df['timestamp'].dt.date <= date_range[1])]

    # Main content
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Overview", "User Profiles", "Session Analysis", "ML Insights"])

    with tab0:
        st.header("Problem Statement")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f"""
                <div style='background-color: #E6F3FF; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #1f77b4;'>Quick Info</h3>
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
                f"""
                <div style='background-color: #FFF5E6; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #ff7f0e;'>Develop a ML Model based solution to refine CAPTCHA</h3>
                <h4 style='color: #ff7f0e;'>Background:</h4>
                <p>UIDAI aims to remove traditional CAPTCHA from its portals to improve user experience. Instead, a passive solution is needed to differentiate between bots and human users.</p>
                <h4 style='color: #ff7f0e;'>Key Requirements:</h4>
                <ul>
                <li>Develop a passive approach using environmental parameters and AI/ML.</li>
                <li>Capture browser context and analyze with backend ML models.</li>
                <li>Protect backend APIs from DoS/DDoS vulnerabilities.</li>
                <li>Minimize human interaction for better user experience.</li>
                <li>Ensure compliance with UIDAI's privacy policies.</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab1:
        st.header("Overview")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://example.com/overview_image.png", caption="System Overview", use_column_width=True)
        
        with col2:
            st.write("### Architecture Diagram")
            st.image("https://example.com/architecture_diagram.png", use_column_width=True)

    with tab2:
        st.header("User Profiles")
        
        st.dataframe(users_df)

        browser_counts = users_df['browser'].value_counts()
        st.bar_chart(browser_counts, use_container_width=True)
        st.write("### Browser Distribution")

        os_counts = users_df['operating_system'].value_counts()
        st.bar_chart(os_counts, use_container_width=True)
        st.write("### Operating System Distribution")

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
        
        # Zoom Level distribution
        fig_zoom = px.histogram(sessions_df, x='zoom_level', color='is_bot', 
                                title="Zoom Level Distribution",
                                labels={'zoom_level': 'Zoom Level (%)', 'count': 'Number of Sessions'},
                                color_discrete_map={0: color_palette[2], 1: color_palette[3]},
                                barmode='overlay')
        fig_zoom.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_zoom, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        <div class='stAlert'>
        <strong>Insights:</strong>
        <ul>
        <li>Different zoom levels can affect how users interact with the page.</li>
        <li>Analyze how zoom levels impact the bot classification to adjust the CAPTCHA thresholds accordingly.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Session Data Sample")
        st.dataframe(sessions_df.head(10))

    with tab4:
        st.header("Machine Learning Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            st.code(classification_report_text, language='text')
        
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
            'feature': ['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'cookie_enabled', 'zoom_level'],
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
            js_enabled = st.checkbox("JavaScript Enabled", value=True)
        with col3:
            cookie_enabled = st.checkbox("Cookies Enabled", value=True)
            zoom_level = st.selectbox("Zoom Level (%)", options=[100, 125, 150, 175, 200])

        if st.button("Classify Session"):
            input_data = np.array([[mouse_movements, keyboard_inputs, time_on_page, 
                                    int(js_enabled), int(cookie_enabled), zoom_level]])
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            result_color = "#2ca02c" if prediction == 0 else "#d62728"
            result_text = "Human" if prediction == 0 else "Bot"
            
            st.markdown(f"""
            <div style='color: {result_color}; font-size: 24px;'>
            <strong>Classification Result:</strong> {result_text}
            </div>
            <div>
            <strong>Probability of being a bot:</strong> {probability:.2f}
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
