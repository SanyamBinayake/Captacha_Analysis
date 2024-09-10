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
        # The rest of the content for tab0 goes here...

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
                               labels={'timestamp': 'Date', 'count': 'Number of Sessions'},
                               color_discrete_sequence=[color_palette[0]])
        fig_sessions.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_sessions, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        <div class='stAlert'>
        <strong>Insights:</strong>
        <ul>
        <li>Look for unusual patterns or spikes in activity that might indicate bot attacks.</li>
        <li>Regular patterns might represent normal human traffic patterns.</li>
        <li>Sudden drops could suggest technical issues or changes in bot behavior.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Bot vs Human sessions over time
        bot_sessions = sessions_df[sessions_df['is_bot'] == 1].groupby(sessions_df['timestamp'].dt.date).size().reset_index(name='bot_count')
        human_sessions = sessions_df[sessions_df['is_bot'] == 0].groupby(sessions_df['timestamp'].dt.date).size().reset_index(name='human_count')
        combined_sessions = pd.merge(bot_sessions, human_sessions, on='timestamp', how='outer').fillna(0)
        
        fig_bot_human = go.Figure()
        fig_bot_human.add_trace(go.Scatter(x=combined_sessions['timestamp'], y=combined_sessions['bot_count'], name='Bot Sessions', line=dict(color=color_palette[1])))
        fig_bot_human.add_trace(go.Scatter(x=combined_sessions['timestamp'], y=combined_sessions['human_count'], name='Human Sessions', line=dict(color=color_palette[2])))
        fig_bot_human.update_layout(title='Bot vs Human Sessions Over Time', xaxis_title='Date', yaxis_title='Number of Sessions', plot_bgcolor='white')
        st.plotly_chart(fig_bot_human, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        <div class='stAlert'>
        <strong>Insights:</strong>
        <ul>
        <li>Compare the trends of bot and human sessions to identify unusual patterns.</li>
        <li>A sudden increase in bot sessions might indicate a new bot attack strategy.</li>
        <li>Consistent levels of human sessions suggest normal user behavior.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.header("User Profiles")
        col1, col2 = st.columns(2)
        
        with col1:
            # Browser distribution
            fig_browsers = px.pie(users_df['browser'].value_counts().reset_index(), 
                                  values='count', names='browser', title="Browser Distribution",
                                  color_discrete_sequence=color_palette)
            fig_browsers.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_browsers, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # Operating System distribution
            fig_os = px.pie(users_df['operating_system'].value_counts().reset_index(), 
                            values='count', names='operating_system', title="Operating System Distribution",
                            color_discrete_sequence=color_palette)
            fig_os.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_os, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        <div class='stAlert'>
        <strong>Insights:</strong>
        <ul>
        <li>Unusual browser or OS distributions might indicate bot activity.</li>
        <li>This information can help tailor the passive CAPTCHA solution for different environments.</li>
        <li>Consider focusing on the most common browsers and operating systems for initial implementation.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("User Data Sample")
        st.dataframe(users_df.head(100))

    with tab3:
        st.header("Session Analysis")
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
        st.dataframe(sessions_df.head(100))

    with tab4:
        
        st.markdown("---")
        
        # Predict whether a session is likely a bot or not
        st.subheader("Session Prediction")
        user_input = {
            'mouse_movements': st.slider("Mouse Movements", min_value=0, max_value=1000, value=50),
            'keyboard_inputs': st.slider("Keyboard Inputs", min_value=0, max_value=500, value=20),
            'time_on_page': st.slider("Time on Page (seconds)", min_value=1, max_value=300, value=50),
            'js_enabled': st.selectbox("JavaScript Enabled", [True])
        }

        session_input = pd.DataFrame([user_input])
        prediction_proba = model.predict_proba(session_input)[0]
        is_bot = model.predict(session_input)[0]

        # Classification output based on prediction probability
        if prediction_proba[1] < 0.3:
            st.success(f"""
                Classification: Human
                
                The system has classified this session as a Human.
                Probability of being a bot: {prediction_proba[1]:.2f}
            """)
        elif 0.3 <= prediction_proba[1] <= 0.7:
            st.warning(f"""
                Classification: Confused
                
                The system is unsure whether this session is a bot or a human.
                Probability of being a bot: {prediction_proba[1]:.2f}
            """)
        else:
            st.error(f"""
                Classification: Bot
                
                The system has classified this session as a Bot.
                Probability of being a bot: {prediction_proba[1]:.2f}
            """)

if __name__ == '__main__':
    main()
