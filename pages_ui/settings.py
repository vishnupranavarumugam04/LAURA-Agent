"""Settings Page - API Key Management"""
import streamlit as st
import os
import time
from utils.llm import SimpleLLM
import google.generativeai as genai


def render():
    st.title("⚙️ Settings")
    
    current_key = os.getenv("GOOGLE_API_KEY", "")
    
    # API Key Status
    if current_key:
        st.success(f"✅ API Key Configured: ...{current_key[-8:]}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Key", use_container_width=True):
                with st.spinner("Testing..."):
                    success, message = SimpleLLM.test_api_key(current_key)
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)
        
        with col2:
            if st.button("Clear Key", use_container_width=True):
                os.environ["GOOGLE_API_KEY"] = ""
                # Clear cached API key from genai
                try:
                    genai.configure(api_key="")
                except:
                    pass
                st.info("Key cleared")
                st.rerun()
    else:
        st.warning("⚠️ No API Key Configured")
    
    st.divider()
    
    # Add new API Key
    st.subheader("Configure API Key")
    
    with st.expander("How to get API Key", expanded=(not current_key)):
        st.markdown("""
        1. Visit: [Google AI Studio](https://aistudio.google.com/app/apikey)
        2. Click "Create API Key"
        3. Copy the complete key (starts with 'AIza...')
        4. Paste it below
        """)
    
    with st.form("api_key_form"):
        api_key_input = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIza...",
        )
        
        submitted = st.form_submit_button("Save & Test", type="primary", use_container_width=True)
    
    if submitted:
        if not api_key_input:
            st.error("Please enter API key")
        elif len(api_key_input) < 30:
            st.error("API key too short")
        elif not api_key_input.startswith("AIza"):
            st.error("Invalid format - should start with 'AIza'")
        else:
            with st.spinner("Validating..."):
                success, message = SimpleLLM.test_api_key(api_key_input)
            
            if success:
                st.success(message)
                os.environ["GOOGLE_API_KEY"] = api_key_input
                genai.configure(api_key=api_key_input)
                st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.error(message)
    
    st.divider()
    
    # Info Section
    st.subheader("Free Tier Limits")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("• 15 requests/min\n• 1,500 requests/day\n• Cost: FREE")
    
    with col2:
        st.info("• Models: Gemini 2.5, 2.0\n• Text & Image Support\n• All Features Included")

