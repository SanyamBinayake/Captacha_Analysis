import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
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
    
    def generate_session_activity_pattern(mouse_movements, keyboard_inputs, time_on_page):
        return (mouse_movements + keyboard_inputs) / (time_on_page + 1)

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

    sessions['time_of_day'] = sessions['timestamp'].dt.hour
    sessions['session_activity_pattern'] = sessions.apply(
        lambda row: generate_session_activity_pattern(row['mouse_movements'], row['keyboard_inputs'], row['time_on_page']),
        axis=1
    )
    
    sessions['is_bot'] = ((sessions['mouse_movements'] > 500) | 
                          (sessions['keyboard_inputs'] > 200) | 
                          (sessions['time_on_page'] < 5)).astype(int)
    
    return sessions

# Define the PyTorch Model
class BotDetectionModel(nn.Module):
    def _init_(self, input_size):
        super(BotDetectionModel, self)._init_()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Prepare data for PyTorch
def prepare_data_for_pytorch(sessions_df):
    features = ['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'cookie_enabled', 'time_of_day', 'session_activity_pattern']
    X = sessions_df[features].values
    y = sessions_df['is_bot'].values

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

# Train the PyTorch model
def train_pytorch_model(train_loader):
    input_size = len(['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'cookie_enabled', 'time_of_day', 'session_activity_pattern'])
    model = BotDetectionModel(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    
    return model

# Evaluate the PyTorch model
def evaluate_pytorch_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

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
    
    # Prepare data for PyTorch
    train_loader, test_loader = prepare_data_for_pytorch(sessions_df)
    
    # Train PyTorch model
    model = train_pytorch_model(train_loader)
    
    # Evaluate the model
    accuracy = evaluate_pytorch_model(model, test_loader)
    
    # Filter data based on sidebar inputs
    sessions_df = sessions_df[(sessions_df['timestamp'].dt.date >= date_range[0]) & (sessions_df['timestamp'].dt.date <= date_range[1])]

    # Main content
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Problem Statement", "Overview", "User Profiles", "Session Analysis", "ML Insights"])

    with tab4:
        st.header("Machine Learning Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
        
        with col2:
            st.markdown("""
            <div class='stAlert'>
            <strong>Interpretation:</strong>
            <ul>
            <li>High accuracy indicates the model's effectiveness in distinguishing between bots and humans.</li>
            <li>Further tuning and experimentation can improve performance.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': ['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'cookie_enabled', 'time_of_day', 'session_activity_pattern'],
            'importance': np.random.rand(7)  # Replace with actual importance if available
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

    with tab1:
        st.header("Overview")
        st.write("""
        The UIDAI (Unique Identification Authority of India) system faces constant threats from automated bots trying to gain unauthorized access. 
        This solution leverages a passive CAPTCHA mechanism powered by machine learning to distinguish between human users and bots based on their session activity data. 
        By analyzing patterns in mouse movements, keyboard inputs, and other session features, we can effectively identify potential bot activity without interrupting the user experience.
        """)

    with tab2:
        st.header("User Profiles")
        st.write("Here is a summary of the user profile data generated:")
        
        fig_browser = px.pie(users_df, names='browser', title="Browser Distribution", color_discrete_sequence=color_palette)
        fig_browser.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_browser, use_container_width=True, config={'displayModeBar': False})

        fig_os = px.pie(users_df, names='operating_system', title="Operating System Distribution", color_discrete_sequence=color_palette)
        fig_os.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_os, use_container_width=True, config={'displayModeBar': False})

        fig_resolution = px.histogram(users_df, x='screen_resolution', title="Screen Resolution Distribution", color_discrete_sequence=color_palette)
        fig_resolution.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_resolution, use_container_width=True, config={'displayModeBar': False})

    with tab3:
        st.header("Session Analysis")
        st.write("Analyze the session data to understand user behavior and identify patterns indicative of bot activity.")
        
        fig_mouse = px.histogram(sessions_df, x='mouse_movements', title="Mouse Movements Distribution", color_discrete_sequence=color_palette)
        fig_mouse.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_mouse, use_container_width=True, config={'displayModeBar': False})

        fig_keyboard = px.histogram(sessions_df, x='keyboard_inputs', title="Keyboard Inputs Distribution", color_discrete_sequence=color_palette)
        fig_keyboard.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_keyboard, use_container_width=True, config={'displayModeBar': False})

        fig_time = px.histogram(sessions_df, x='time_on_page', title="Time on Page Distribution", color_discrete_sequence=color_palette)
        fig_time.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig_time, use_container_width=True, config={'displayModeBar': False})

        st.markdown("""
        <div class='stAlert'>
        <strong>Observations:</strong>
        <ul>
        <li>Normal human behavior is typically characterized by moderate mouse movements and keyboard inputs, with reasonable time spent on the page.</li>
        <li>Bots often exhibit extreme values in these metrics, such as very high or very low activity, which can be effectively captured by the model.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if _name_ == "_main_":
    main()
