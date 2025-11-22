"""Profile Page"""
import streamlit as st
import os
from utils.db import SimpleDB


def render():
    st.title("👤 Profile")
    
    db = SimpleDB()
    user_id = st.session_state.get("user_id", "user_001")
    
    user = db.get_user(user_id)
    
    # Edit button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("User Information")
    with col2:
        if st.button("✏️ Edit Profile", type="primary"):
            st.session_state.edit_profile = not st.session_state.get("edit_profile", False)
            st.rerun()
    
    if user:
        if st.session_state.get("edit_profile", False):
            # Edit mode
            st.info("Edit your profile information")
            with st.form("edit_profile_form"):
                new_name = st.text_input("Name", value=user.get('name', ''))
                new_email = st.text_input("Email", value=user.get('email', ''))
                new_bio = st.text_area("Bio", value=user.get('bio', ''), height=100)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("💾 Save Changes", type="primary"):
                        db.update_user(user_id, new_name, new_email, new_bio)
                        st.success("Profile updated successfully!")
                        st.session_state.edit_profile = False
                        st.rerun()
                with col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.edit_profile = False
                        st.rerun()
        else:
            # View mode
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {user.get('name', 'N/A')}")
                st.write(f"**Email:** {user.get('email', 'N/A')}")
            with col2:
                st.write(f"**Bio:** {user.get('bio', 'N/A')}")
                st.write(f"**Joined:** {user.get('created_at', 'N/A')}")

    st.divider()
    
    st.divider()
    
    st.subheader("Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    tasks = db.get_tasks(user_id)
    metrics = db.get_metrics(user_id)
    logs = db.get_agent_logs(user_id)
    
    with col1:
        st.metric("Tasks", len(tasks))
    with col2:
        st.metric("Metrics Logged", len(metrics))
    with col3:
        st.metric("Agent Actions", len(logs))
    with col4:
        st.metric("Active", "Yes")
