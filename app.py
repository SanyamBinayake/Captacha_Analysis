import streamlit as st
import pandas as pd

def load_static_data():
    try:
        users_df = pd.read_csv('static_user_data.csv')
        sessions_df = pd.read_csv('static_session_data.csv')
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Print out the first few rows to understand the data structure
    st.write("Users DataFrame preview:")
    st.write(users_df.head())
    st.write("Sessions DataFrame preview:")
    st.write(sessions_df.head())

    try:
        sessions_df['timestamp'] = pd.to_datetime(sessions_df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        sessions_df = sessions_df.dropna(subset=['timestamp'])
    except KeyError as e:
        st.error(f"Error in timestamp column: {e}")
        sessions_df['timestamp'] = pd.NaT
    
    return users_df, sessions_df

def main():
    st.title("ML-Enhanced Passive CAPTCHA Solution")
    
    # Load data
    users_df, sessions_df = load_static_data()
    
    if users_df.empty or sessions_df.empty:
        st.warning("Data not available. Please check the files.")
        return
    
    # Continue with the rest of your Streamlit app logic
    st.write("Users Data:")
    st.write(users_df)
    
    st.write("Sessions Data:")
    st.write(sessions_df)
    
if __name__ == "__main__":
    main()
