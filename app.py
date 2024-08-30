import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from faker import Faker

# Initialize Faker
fake = Faker()

# Generate synthetic data
def generate_synthetic_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        mouse_movements = np.random.randint(0, 1000)
        keyboard_inputs = np.random.randint(0, 500)
        time_on_page = np.random.randint(0, 600)
        js_enabled = np.random.choice([0, 1])
        is_bot = np.random.choice([0, 1], p=[0.7, 0.3])  # Assuming 30% bots, 70% humans

        data.append([mouse_movements, keyboard_inputs, time_on_page, js_enabled, is_bot])
    
    df = pd.DataFrame(data, columns=['mouse_movements', 'keyboard_inputs', 'time_on_page', 'js_enabled', 'is_bot'])
    return df

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
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    return model, metrics

# Main app
def main():
    st.title("ML-Enhanced Passive CAPTCHA Solution for UIDAI")

    # Generate synthetic data
    sessions_df = generate_synthetic_data()

    # Train ML model
    model, metrics = train_model(sessions_df)

    # Display metrics
    st.subheader("Model Performance Metrics")
    st.write(f"Accuracy: {metrics['Accuracy']:.2f}")

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
    
if __name__ == '__main__':
    main()
