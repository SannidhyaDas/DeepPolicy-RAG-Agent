import os
import streamlit as st
from dotenv import load_dotenv

# Try loading from local .env file first (for local development)
load_dotenv()

def get_api_key(key_name: str) -> str:
    """
    Fetches API keys securely. 
    Checks Streamlit secrets first (for cloud deployment), 
    then falls back to local environment variables.
    """
    # 1. Check Streamlit Secrets (Cloud)
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except FileNotFoundError:
        pass # Not running in Streamlit environment or secrets.toml is missing

    # 2. Check Local Environment Variables (.env)
    api_key = os.getenv(key_name)
    
    if not api_key:
        raise ValueError(f"CRITICAL ERROR: {key_name} is missing. Please add it to .env or Streamlit Secrets.")
    
    return api_key

# Centralized constants
GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY")
GROQ_API_KEY = get_api_key("GROQ_API_KEY")
TAVILY_API_KEY = get_api_key("TAVILY_API_KEY")
# OPENAI_API_KEY = get_api_key("OPENAI_API_KEY") # Uncomment if using OpenAI