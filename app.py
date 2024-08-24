import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

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

# Load static data
@st.cache_data
def load_static_data():
    # Load user data
    users_df = pd.read_csv('static_user_data.csv')
    
    # Load session data
    sessions_df = pd.read_csv('static_session_data.csv')
    
    return users_df, sessions_df

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
    
    # Load static data
    users_df, sessions_df = load_static_data()
    
    # Train ML model
    model, classification_report = train_model(sessions_df)
    
    # Filter data based on sidebar inputs
    sessions_df['timestamp'] = pd.to_datetime(sessions_df['timestamp'])
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
                <h4 style='color: #ff7f0e;'>Expected Solution:</h4>
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
        
        # User activity distribution
        user_activity = users_df.groupby('user_id').agg({
            'mouse_movements': 'sum',
            'keyboard_inputs': 'sum',
            'time_on_page': 'sum',
            'js_enabled': 'mean'
        }).reset_index()
        
        fig_user_activity = px.histogram(user_activity, x='mouse_movements', nbins=50, title="Mouse Movements Distribution",
                                         labels={'mouse_movements': 'Mouse Movements'},
                                         color_discrete_sequence=[color_palette[3]])
        fig_user_activity.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_user_activity, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")
        
        fig_user_activity_keyboard = px.histogram(user_activity, x='keyboard_inputs', nbins=50, title="Keyboard Inputs Distribution",
                                                  labels={'keyboard_inputs': 'Keyboard Inputs'},
                                                  color_discrete_sequence=[color_palette[4]])
        fig_user_activity_keyboard.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_user_activity_keyboard, use_container_width=True, config={'displayModeBar': False})

    with tab3:
        st.header("Session Analysis")
        
        # Analyze session characteristics
        session_characteristics = sessions_df[['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'is_bot']]
        st.write(session_characteristics.describe())
        
        st.markdown("---")
        
        # Feature importance from the model
        feature_importances = model.feature_importances_
        feature_names = session_characteristics.columns[:-1]
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        fig_feature_importance = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance",
                                        labels={'Feature': 'Feature', 'Importance': 'Importance'},
                                        color_discrete_sequence=[color_palette[5]])
        fig_feature_importance.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_feature_importance, use_container_width=True, config={'displayModeBar': False})

    with tab4:
        st.header("ML Insights")
        
        st.subheader("Model Performance")
        st.write("### Classification Report")
        st.text(classification_report)
        
        st.subheader("Model Usage")
        st.write("To use this model for predictions, ensure you have the following features:")
        st.write(" - mouse_movements")
        st.write(" - keyboard_inputs")
        st.write(" - time_on_page")
        st.write(" - js_enabled")

        # Example prediction
        example_input = {
            'mouse_movements': [500],
            'keyboard_inputs': [300],
            'time_on_page': [120],
            'js_enabled': [1]
        }
        example_df = pd.DataFrame(example_input)
        prediction = model.predict(example_df)[0]
        st.write(f"Example Prediction: {'Bot' if prediction else 'Human'}")

if __name__ == "__main__":
    main()
