"""
LAURA v1.0 : The Adaptive Life Concierge Agent
Multi-Agent AI System with Predictive ML Models
"""
import streamlit as st
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import pages
from pages_ui import (
    dashboard, 
    scheduler, 
    study_mate, 
    buddy, 
    profile, 
    settings,
    advanced_analytics
)

# Page config
st.set_page_config(
    page_title="LAURA v1.0",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global dark theme CSS with enhanced UI
st.markdown("""
    <style>
    /* Force dark theme globally */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Dark metric containers everywhere */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #CCCCCC !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
        border: 2px solid #444444 !important;
        padding: 20px !important;
        border-radius: 15px !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 24px rgba(0,0,0,0.6) !important;
    }
    
    /* Dark info/warning boxes */
    .stAlert {
        background-color: #1E1E1E !important;
        border: 1px solid #444 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar metrics dark */
    [data-testid="stSidebar"] [data-testid="metric-container"] {
        background-color: #2D2D2D !important;
        border: 1px solid #444 !important;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 8px 16px rgba(255,75,75,0.3) !important;
    }
    
    /* Enhanced dividers */
    hr {
        border-color: #444 !important;
        margin: 20px 0 !important;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px 0;
        margin-bottom: 10px;
    }
    .logo-text {
        font-size: 42px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .version-badge {
        background: linear-gradient(135deg, #FF4B4B 0%, #C73E3E 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
        margin-left: 10px;
        box-shadow: 0 2px 8px rgba(255,75,75,0.4);
    }
    
    /* API Warning Banner */
    /* API Warning Banner */
    .api-warning-banner {
        background: linear-gradient(135deg, #FF4B4B 0%, #C73E3E 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: 700;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(199,62,62,0.36);
        animation: bannerIn 600ms cubic-bezier(.2,.9,.3,1) both;
    }

    /* Slide-in with subtle bounce */
    @keyframes bannerIn {
        0% { transform: translateY(-18px) scale(0.98); opacity: 0; }
        60% { transform: translateY(6px) scale(1.02); opacity: 1; }
        100% { transform: translateY(0) scale(1); opacity: 1; }
    }

    /* Shimmer sweep to add polish */
    .api-warning-banner::after {
        content: '';
        position: absolute;
        left: -40%;
        top: 0;
        width: 40%;
        height: 100%;
        background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.18), rgba(255,255,255,0.06));
        transform: skewX(-20deg);
        animation: shimmer 2.2s ease-in-out infinite;
        pointer-events: none;
        opacity: 0.9;
    }

    @keyframes shimmer {
        0% { left: -40%; }
        50% { left: 100%; }
        100% { left: 100%; }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_001"
    from utils.db import SimpleDB, create_demo_user
    db = SimpleDB()
    if not db.get_user(st.session_state.user_id):
        create_demo_user(db, st.session_state.user_id)

# Check API key status
api_key_configured = bool(os.getenv("GOOGLE_API_KEY"))

# Initialize first_visit flag
if "first_visit" not in st.session_state:
    st.session_state.first_visit = True
    if not api_key_configured:
        st.session_state.show_api_setup = True
    else:
        st.session_state.show_api_setup = False
else:
    if "show_api_setup" not in st.session_state:
        st.session_state.show_api_setup = False

# Sidebar navigation
with st.sidebar:
    # Logo with version badge
    st.markdown("""
        <div class="logo-container">
            <span class="logo-text">LAURA</span>
            <span class="version-badge">v1.0</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: #888; font-size: 12px; margin-top: -10px;'>The Adaptive Life Concierge Agent</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #667eea; font-size: 11px; font-weight: 600;'>Multi-Agent AI | Predictive ML</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # API Status indicator
    if api_key_configured:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); 
                        padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 15px;
                        box-shadow: 0 4px 12px rgba(0,184,148,0.3);">
                <span style="color: white; font-weight: 600;">✅ API Key Active</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #FF4B4B 0%, #C73E3E 100%); 
                        padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 15px;
                        box-shadow: 0 4px 12px rgba(255,75,75,0.3);">
                <span style="color: white; font-weight: 600;">⚠️ API Key Required</span>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🔑 Configure API Key", use_container_width=True, type="primary"):
            st.session_state.show_api_setup = True
            st.rerun()
    
    st.divider()
    
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Dashboard", 
            "Scheduler", 
            "Study Mate", 
            "Buddy", 
            "Advanced Analytics",
            "Profile", 
            "Settings"
        ],
        icons=[
            "speedometer2", 
            "calendar2", 
            "book", 
            "robot", 
            "graph-up-arrow",
            "person", 
            "gear"
        ],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#FFFFFF", "font-size": "16px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "left", 
                "margin":"2px 0",
                "color": "#FFFFFF",
                "background-color": "transparent",
                "border-radius": "8px",
                "padding": "10px 15px",
                "transition": "all 0.3s ease"
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "color": "#FFFFFF",
                "font-weight": "600",
                "box-shadow": "0 4px 12px rgba(102,126,234,0.4)"
            },
        }
    )
    
    st.divider()
    
    # System status with dark theme
    st.markdown("### 📋 System Status")
    
    # Custom dark metrics for sidebar
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 12px; border-radius: 10px; text-align: center;
                        box-shadow: 0 4px 12px rgba(102,126,234,0.3);">
                <div style="color: #E0E0E0; font-size: 11px; font-weight: 500;">Agents</div>
                <div style="color: #FFFFFF; font-size: 24px; font-weight: 900;">6</div>
                <div style="color: #B8F5B8; font-size: 10px; font-weight: 600;">Active</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #FF4B4B 0%, #C73E3E 100%); 
                        padding: 12px; border-radius: 10px; text-align: center;
                        box-shadow: 0 4px 12px rgba(255,75,75,0.3);">
                <div style="color: #E0E0E0; font-size: 11px; font-weight: 500;">ML Models</div>
                <div style="color: #FFFFFF; font-size: 24px; font-weight: 900;">4</div>
                <div style="color: #B8F5B8; font-size: 10px; font-weight: 600;">Trained</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: #00b894; font-size: 12px; font-weight: 600; margin-top: 10px;'>✅ All systems operational</p>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("<p style='text-align: center; color: #888; font-size: 11px;'>Kaggle Capstone Project</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #667eea; font-size: 11px; font-weight: 600;'>Multi-Agent AI System</p>", unsafe_allow_html=True)

# Show API setup modal on first visit or when requested
if st.session_state.show_api_setup and not api_key_configured:
    st.markdown("""
        <div class="api-warning-banner">
            ⚠️ API KEY REQUIRED - Please configure your Google Gemini API key to unlock all features!
        </div>
    """, unsafe_allow_html=True)
    
    # Force navigation to Settings
    selected = "Settings"

# Show warning banner for locked features
if not api_key_configured and selected in ["Study Mate", "Buddy"]:
    st.markdown("""
        <div class="api-warning-banner">
            🔒 This feature requires API Key configuration. Please go to Settings to set up your key.
        </div>
    """, unsafe_allow_html=True)

# Route to pages
if selected == "Dashboard":
    dashboard.render()
elif selected == "Scheduler":
    scheduler.render()
elif selected == "Study Mate":
    if not api_key_configured:
        st.error("🔒 API Key Required")
        st.warning("Please configure your Google Gemini API key in Settings to use Study Mate.")
        if st.button("Go to Settings", type="primary"):
            st.session_state.show_api_setup = True
            st.rerun()
    else:
        study_mate.render()
elif selected == "Buddy":
    if not api_key_configured:
        st.error("🔒 API Key Required")
        st.warning("Please configure your Google Gemini API key in Settings to use Buddy.")
        if st.button("Go to Settings", type="primary"):
            st.session_state.show_api_setup = True
            st.rerun()
    else:
        buddy.render()
elif selected == "Advanced Analytics":
    advanced_analytics.render()
elif selected == "Profile":
    profile.render()
elif selected == "Settings":
    settings.render()

# Footer with enhanced styling
st.divider()
st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); 
                border-radius: 10px; margin-top: 20px;">
        <p style="color: #667eea; font-weight: 700; font-size: 16px; margin-bottom: 8px;">
            LAURA v1.0: The Adaptive Life Concierge Agent
        </p>
        <p style="color: #888; font-size: 12px; margin-bottom: 5px;">
            Powered by Google Gemini API | Custom ML Models
        </p>
        <p style="color: #667eea; font-size: 11px; font-weight: 600;">
            First Multi-Agent Productivity AI
        </p>
    </div>
""", unsafe_allow_html=True)