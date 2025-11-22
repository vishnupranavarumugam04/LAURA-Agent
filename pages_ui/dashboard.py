"""Dashboard Page - Dynamic Agent-Based Metrics"""
import streamlit as st
from utils.db import SimpleDB
from datetime import datetime, timedelta
import random


def calculate_dynamic_metrics(user_id: str, db: SimpleDB):
    """Calculate dynamic metrics based on user activity"""
    
    # Get user activity data
    tasks = db.get_tasks(user_id)
    agent_logs = db.get_agent_logs(user_id)
    
    # Base metrics
    base_sleep = 75
    base_focus = 80
    base_stress = 40
    base_productivity = 75
    
    # Adjust based on tasks
    completed_tasks = [t for t in tasks if t.get("status") == "completed"]
    pending_tasks = [t for t in tasks if t.get("status") == "pending"]
    
    # Task completion affects metrics
    completion_rate = len(completed_tasks) / len(tasks) if tasks else 0.5
    
    # Calculate time-based adjustments
    current_hour = datetime.now().hour
    
    # Sleep quality (higher in morning if well-rested, lower at night)
    if 6 <= current_hour <= 10:
        sleep_bonus = 15
    elif 22 <= current_hour or current_hour <= 5:
        sleep_bonus = -10
    else:
        sleep_bonus = 0
    
    # Focus (peaks mid-morning, declines evening)
    if 9 <= current_hour <= 11:
        focus_bonus = 20
    elif 14 <= current_hour <= 16:
        focus_bonus = 10
    elif 18 <= current_hour <= 22:
        focus_bonus = -15
    else:
        focus_bonus = 0
    
    # Stress (increases with pending high-priority tasks)
    high_priority_pending = len([t for t in pending_tasks if t.get("priority") == "High"])
    stress_adjustment = min(high_priority_pending * 8, 30)
    
    # Productivity (based on completion rate and agent activity)
    productivity_bonus = completion_rate * 15
    agent_activity_bonus = min(len(agent_logs) * 0.5, 10)
    
    # Calculate final metrics
    sleep = min(100, max(0, base_sleep + sleep_bonus + random.randint(-5, 5)))
    focus = min(100, max(0, base_focus + focus_bonus + (completion_rate * 10) + random.randint(-3, 3)))
    stress = min(100, max(0, base_stress + stress_adjustment - (completion_rate * 10) + random.randint(-5, 5)))
    productivity = min(100, max(0, base_productivity + productivity_bonus + agent_activity_bonus + random.randint(-3, 3)))
    
    # Calculate trends
    sleep_trend = "+" if sleep_bonus >= 0 else "-"
    focus_trend = "+" if focus_bonus >= 0 else "-"
    stress_trend = "-" if completion_rate > 0.5 else "+"
    productivity_trend = "+" if completion_rate > 0.5 else "-"
    
    return {
        "Sleep": {"value": sleep, "trend": sleep_trend, "delta": abs(sleep_bonus)},
        "Focus": {"value": focus, "trend": focus_trend, "delta": abs(focus_bonus)},
        "Stress": {"value": stress, "trend": stress_trend, "delta": abs(stress_adjustment)},
        "Productivity": {"value": productivity, "trend": productivity_trend, "delta": abs(int(productivity_bonus))}
    }


def get_metric_emoji(metric_name: str, value: float):
    """Get appropriate emoji based on metric and value"""
    emojis = {
        "Sleep": "😴" if value < 60 else "😊" if value < 80 else "🌟",
        "Focus": "😵" if value < 60 else "🎯" if value < 80 else "🔥",
        "Stress": "😰" if value > 70 else "😌" if value > 40 else "🎉",
        "Productivity": "📉" if value < 60 else "📈" if value < 80 else "🚀"
    }
    return emojis.get(metric_name, "📊")


def get_metric_color(metric_name: str, value: float):
    """Get color based on metric value"""
    if metric_name == "Stress":
        # Inverted: lower is better
        if value > 70:
            return "#FF4B4B"
        elif value > 40:
            return "#FFA500"
        else:
            return "#00b894"
    else:
        # Normal: higher is better
        if value < 60:
            return "#FF4B4B"
        elif value < 80:
            return "#FFA500"
        else:
            return "#00b894"


def render():
    st.title("📊 Dashboard")
    
    # Initialize database
    db = SimpleDB()
    user_id = st.session_state.get("user_id", "user_001")
    
    # Calculate dynamic metrics
    metrics = calculate_dynamic_metrics(user_id, db)
    
    # Display metrics with enhanced HTML cards
    st.subheader("🎯 Real-Time Metrics")
    st.caption("Powered by Multi-Agent AI System - Updates based on your activity")
    
    # Create 4 columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metric_configs = [
        ("Sleep", col1, "Sleep quality based on time and rest patterns"),
        ("Focus", col2, "Cognitive performance and concentration"),
        ("Stress", col3, "Stress level from task load and deadlines"),
        ("Productivity", col4, "Overall productivity and task completion")
    ]
    
    for metric_name, col, description in metric_configs:
        metric_data = metrics[metric_name]
        value = metric_data["value"]
        trend = metric_data["trend"]
        delta = metric_data["delta"]
        
        emoji = get_metric_emoji(metric_name, value)
        color = get_metric_color(metric_name, value)
        trend_symbol = "▲" if trend == "+" else "▼"
        trend_color = "#00b894" if (trend == "+" and metric_name != "Stress") or (trend == "-" and metric_name == "Stress") else "#FF4B4B"
        
        with col:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); 
                            padding: 25px; border-radius: 15px; border: 2px solid {color};
                            box-shadow: 0 8px 20px rgba(0,0,0,0.5); 
                            transition: transform 0.3s ease, box-shadow 0.3s ease;
                            cursor: pointer;"
                     onmouseover="this.style.transform='translateY(-8px)'; this.style.boxShadow='0 12px 30px rgba(0,0,0,0.7)';"
                     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 8px 20px rgba(0,0,0,0.5)';">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div style="color: #CCCCCC; font-size: 14px; font-weight: 600;">{emoji} {metric_name}</div>
                        <div style="background: {color}; color: white; padding: 3px 8px; border-radius: 8px; font-size: 11px; font-weight: 700;">
                            LIVE
                        </div>
                    </div>
                    <div style="color: #FFFFFF; font-size: 42px; font-weight: 900; margin: 15px 0;">
                        {value:.0f}<span style="font-size: 24px; color: #888;">%</span>
                    </div>
                    <div style="color: {trend_color}; font-size: 14px; font-weight: 600;">
                        {trend_symbol} {delta}% from baseline
                    </div>
                    <div style="color: #888; font-size: 11px; margin-top: 8px; font-style: italic;">
                        {description}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # AI Agent Activity Section
    st.subheader("🤖 AI Agent Activity")
    st.caption("Real-time monitoring of multi-agent system")
    
    logs = db.get_agent_logs(user_id)
    
    if logs:
        # Show only recent logs (last 10)
        recent_logs = logs[:10]
        
        for log in recent_logs:
            agent_name = log['agent_name']
            action = log['action']
            details = log['details']
            timestamp = log.get('timestamp', 'Unknown')
            
            # Agent-specific colors
            agent_colors = {
                "Coordinator": "#667eea",
                "Task Agent": "#00b894",
                "Schedule Agent": "#FFA500",
                "Health Agent": "#FF4B4B",
                "Learning Agent": "#764ba2",
                "Insight Agent": "#00cec9"
            }
            
            # Find matching color
            color = "#667eea"  # default
            for key in agent_colors:
                if key in agent_name:
                    color = agent_colors[key]
                    break
            
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1E1E1E 0%, #2d2d2d 100%); 
                            padding: 15px; border-radius: 12px; margin-bottom: 12px; 
                            border-left: 5px solid {color};
                            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                            transition: transform 0.2s ease;"
                     onmouseover="this.style.transform='translateX(5px)';"
                     onmouseout="this.style.transform='translateX(0)';">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="color: {color}; font-weight: 700; font-size: 15px;">
                            🤖 {agent_name}: {action}
                        </div>
                        <div style="color: #888; font-size: 11px;">
                            {timestamp}
                        </div>
                    </div>
                    <div style="color: #CCCCCC; font-size: 13px; line-height: 1.6;">
                        {details}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #2d3436 0%, #1a1a1a 100%); 
                        padding: 20px; border-radius: 12px; text-align: center;
                        border: 2px dashed #444;">
                <p style="color: #888; font-size: 14px; margin: 0;">
                    🤖 No agent activity yet. Create tasks to activate the multi-agent system!
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Today's Tasks Section
    st.subheader("📅 Today's Schedule")
    tasks = db.get_tasks(user_id)
    
    if tasks:
        # Separate by status
        pending = [t for t in tasks if t.get("status") == "pending"]
        completed = [t for t in tasks if t.get("status") == "completed"]
        
        # Show stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 15px; border-radius: 10px; text-align: center;
                            box-shadow: 0 4px 12px rgba(102,126,234,0.3);">
                    <div style="color: white; font-size: 28px; font-weight: 900;">{len(tasks)}</div>
                    <div style="color: #E0E0E0; font-size: 13px;">Total Tasks</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%); 
                            padding: 15px; border-radius: 10px; text-align: center;
                            box-shadow: 0 4px 12px rgba(255,165,0,0.3);">
                    <div style="color: white; font-size: 28px; font-weight: 900;">{len(pending)}</div>
                    <div style="color: #FFE5CC; font-size: 13px;">Pending</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); 
                            padding: 15px; border-radius: 10px; text-align: center;
                            box-shadow: 0 4px 12px rgba(0,184,148,0.3);">
                    <div style="color: white; font-size: 28px; font-weight: 900;">{len(completed)}</div>
                    <div style="color: #E0FFF8; font-size: 13px;">Completed</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display tasks
        for task in tasks[:8]:  # Show first 8 tasks
            status_icon = "✅" if task["status"] == "completed" else "⏳"
    
            priority_colors = {
                "High": "#FF4B4B",
                "Medium": "#FFA500",
                "Low": "#00b894"
            }
            priority_color = priority_colors.get(task["priority"], "#888888")
    
    # Task completion styling
            opacity = "0.6" if task["status"] == "completed" else "1.0"
            text_decoration = "line-through" if task["status"] == "completed" else "none"
    
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1E1E1E 0%, #2d2d2d 100%); 
                            padding: 15px; border-radius: 12px; margin-bottom: 10px; 
                            border-left: 5px solid {priority_color};
                            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                            opacity: {opacity};
                            transition: transform 0.2s ease, opacity 0.2s ease;"
                    onmouseover="this.style.transform='translateX(5px)'; this.style.opacity='1.0';"
                    onmouseout="this.style.transform='translateX(0)'; this.style.opacity='{opacity}';">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="color: #FFFFFF; font-weight: 700; font-size: 15px; 
                                        text-decoration: {text_decoration}; margin-bottom: 5px;">
                                {status_icon} {task['title']}
                            </div>
                            <div style="color: #AAAAAA; font-size: 12px;">
                                <span style="background: {priority_color}; color: white; padding: 2px 8px; 
                                            border-radius: 6px; font-size: 11px; font-weight: 600;">
                                    {task['priority']}
                                </span>
                                {f" • ⏰ {task['start_time']} - {task['end_time']}" if task["start_time"] else ""}
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #2d3436 0%, #1a1a1a 100%); 
                        padding: 30px; border-radius: 12px; text-align: center;
                        border: 2px dashed #444;">
                <div style="font-size: 48px; margin-bottom: 15px;">📅</div>
                <p style="color: #888; font-size: 16px; margin: 0 0 10px 0;">
                    No tasks scheduled yet
                </p>
                <p style="color: #667eea; font-size: 14px; margin: 0;">
                    Go to Scheduler to create your first task!
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Refresh button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Refresh Metrics", use_container_width=True, type="primary"):
            st.rerun()