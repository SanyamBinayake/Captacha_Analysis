import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="ML-Enhanced Passive CAPTCHA Solution", layout="wide")

# Set a consistent color palette
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Custom CSS for styling
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
    .small-input .stNumberInput input {
    padding: 5px;
    font-size: 12px;
    width: 50px;
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

# Load data from CSV file
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Train ML model
@st.cache_resource
def train_model(sessions_df):
    features = ['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled']
    X = sessions_df[features]
    y = sessions_df['is_bot']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    mse = mean_squared_error(y_test, y_prob)
    r2 = r2_score(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        "MSE": mse,
        "R2": r2,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    return model, metrics

# Main app
def main():
    st.title("ML-Enhanced Passive CAPTCHA Solution for UIDAI")

    # Directly load the CSV file
    file_name = 'generated_data.csv'
    
    try:
        sessions_df = load_data(file_name)

        # Ensure the necessary columns are present
        required_columns = ['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'is_bot']
        if all(col in sessions_df.columns for col in required_columns):
            # Train ML model
            model, metrics = train_model(sessions_df)

            # Display metrics
            st.subheader("Model Performance Metrics")
            st.write(f"Mean Squared Error: {metrics['MSE']:.2f}")
            st.write(f"R^2 Score: {metrics['R2']:.2f}")
            st.write(f"AUC Score: {metrics['AUC']:.2f}")
            st.write(f"Precision: {metrics['Precision']:.2f}")
            st.write(f"Recall: {metrics['Recall']:.2f}")
            st.write(f"F1 Score: {metrics['F1 Score']:.2f}")

            # Main content
            st.subheader("Live Session Classification")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="small-input">', unsafe_allow_html=True)
                mouse_movements = st.number_input("Mouse Movements", min_value=0, max_value=1000, value=50)
                keyboard_inputs = st.number_input("Keyboard Inputs", min_value=0, max_value=500, value=20)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="small-input">', unsafe_allow_html=True)
                time_on_page = st.number_input("Time on Page (seconds)", min_value=0, max_value=600, value=60)
                st.markdown('</div>', unsafe_allow_html=True)
                js_enabled = True  # JavaScript enabled by default

            if st.button("Classify Session"):
                input_data = np.array([[mouse_movements, keyboard_inputs, time_on_page, int(js_enabled)]])
                prediction_proba = model.predict_proba(input_data)[0][1]
                
                if prediction_proba < 0.3:
                    st.success(f"""
                        **Classification: Human**
                        
                        The system has classified this session as a **Human**.
                        Probability of being a bot: {prediction_proba:.2f}
                    """)
                elif 0.3 <= prediction_proba <= 0.7:
                    st.warning(f"""
                        **Classification: Confused**
                        
                        The system is unsure whether this session is a bot or a human.
                        Probability of being a bot: {prediction_proba:.2f}
                    """)
                else:
                    st.error(f"""
                        **Classification: Bot**
                        
                        The system has classified this session as a **Bot**.
                        Probability of being a bot: {prediction_proba:.2f}
                    """)
        else:
            st.error("The uploaded CSV file does not contain the required columns.")
    except FileNotFoundError:
        st.error(f"The file {file_name} was not found. Please make sure the file is in the correct directory.")

if __name__ == '__main__':
    main()
