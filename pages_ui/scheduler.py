"""Scheduler Page"""
import streamlit as st
from datetime import datetime
from utils.db import SimpleDB


def render():
    st.title("📅 Scheduler")
    
    db = SimpleDB()
    user_id = st.session_state.get("user_id", "user_001")
    
    # Create new task
    st.subheader("Create New Task")
    
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Task Title", placeholder="e.g., Study Physics")
        start_time = st.time_input("Start Time", value=datetime.now().time())
    
    with col2:
        priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        duration = st.number_input("Duration (minutes)", 15, 480, 60, 15)
    
    description = st.text_area("Description", placeholder="Add details...")
    
    if st.button("Add Task", type="primary"):
        if title:
            # Calculate end time
            from datetime import timedelta, datetime as dt
            start_dt = dt.combine(dt.today(), start_time)
            end_dt = start_dt + timedelta(minutes=duration)
            
            task_id = f"task_{int(datetime.now().timestamp())}"
            db.add_task(
                user_id, task_id, title, description,
                priority, start_time.strftime("%H:%M"), end_dt.time().strftime("%H:%M")
            )
            st.success("Task added!")
            st.rerun()
    
    st.divider()
    
    # Display tasks
    st.subheader("Your Tasks")
    tasks = db.get_tasks(user_id)
    
    if tasks:
        # Separate by status
        pending_tasks = [t for t in tasks if t.get("status") == "pending"]
        completed_tasks = [t for t in tasks if t.get("status") == "completed"]
        
        # Show pending tasks
        if pending_tasks:
            st.markdown("### 📋 Pending Tasks")
            for task in pending_tasks:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{task['title']}**")
                    st.caption(f"{task['start_time']} - {task['end_time']} | {task['priority']}")
                    if task['description']:
                        st.caption(task['description'])
                
                with col2:
                    # Mark as completed button
                    if st.button("✅ Complete", key=f"complete_{task['id']}"):
                        db.update_task_status(task['id'], 'completed')
                        st.success("Task marked as completed!")
                        st.rerun()
                
                with col3:
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{task['id']}"):
                        db.delete_task(task["id"])
                        st.success("Task deleted!")
                        st.rerun()
                
                st.divider()
        
        # Show completed tasks in collapsible section
        if completed_tasks:
            with st.expander(f"✅ Completed Tasks ({len(completed_tasks)})", expanded=False):
                for task in completed_tasks:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"~~**{task['title']}**~~")  # Strikethrough
                        st.caption(f"{task['start_time']} - {task['end_time']} | {task['priority']}")
                        if task['description']:
                            st.caption(task['description'])
                    
                    with col2:
                        # Mark as pending button
                        if st.button("🔄 Reopen", key=f"reopen_{task['id']}"):
                            db.update_task_status(task['id'], 'pending')
                            st.success("Task reopened!")
                            st.rerun()
                    
                    with col3:
                        # Delete button
                        if st.button("🗑️ Delete", key=f"delete_comp_{task['id']}"):
                            db.delete_task(task["id"])
                            st.success("Task deleted!")
                            st.rerun()
                    
                    st.divider()
    else:
        st.info("No tasks yet. Create one above!")