import streamlit as st
import openai
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_gsheets import GSheetsConnection

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Growth GPT Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Pro" look
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stDataFrame {
        border: 1px solid #333;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MEMORY FUNCTIONS ---
def load_memory():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="ChatHistory", usecols=[0, 1, 2], ttl=0)
        if df.empty or "role" not in df.columns:
            return []
        return df.to_dict("records")
    except Exception:
        return []

def save_memory(role, content):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        try:
            existing_data = conn.read(worksheet="ChatHistory", usecols=[0, 1, 2], ttl=0)
        except:
            existing_data = pd.DataFrame(columns=["timestamp", "role", "content"])
        
        new_row = pd
