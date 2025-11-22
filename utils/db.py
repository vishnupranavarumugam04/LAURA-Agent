"""
Simple Database Handler - No Threading Issues
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional


class SimpleDB:
    """Simple SQLite database handler - works reliably"""
    
    def __init__(self, db_path: str = "laura.db"):
        self.db_path = db_path
        if db_path != ":memory:":
            self._init_db()
        else:
            # For in-memory, initialize on first use
            if not hasattr(SimpleDB, '_mem_initialized'):
                self._init_db()
                SimpleDB._mem_initialized = True
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                bio TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                description TEXT,
                status TEXT DEFAULT 'pending',
                priority TEXT,
                start_time TEXT,
                end_time TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                metric_type TEXT,
                value REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Agent logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                agent_name TEXT,
                action TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Chat table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                role TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_conn(self):
        """Get fresh database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # User operations
    def create_user(self, user_id: str, name: str, email: str = "", bio: str = ""):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO users (id, name, email, bio) VALUES (?, ?, ?, ?)",
            (user_id, name, email, bio)
        )
        conn.commit()
        conn.close()
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def update_user(self, user_id: str, name: str = None, email: str = None, bio: str = None):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        user = self.get_user(user_id)
        if not user:
            conn.close()
            return
        
        name = name if name is not None else user['name']
        email = email if email is not None else user['email']
        bio = bio if bio is not None else user['bio']
        
        cursor.execute(
            "UPDATE users SET name = ?, email = ?, bio = ? WHERE id = ?",
            (name, email, bio, user_id)
        )
        conn.commit()
        conn.close()
    
    # Task operations
    def add_task(self, user_id: str, task_id: str, title: str, desc: str = "", 
                 priority: str = "Medium", start: str = "", end: str = ""):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO tasks (id, user_id, title, description, priority, start_time, end_time)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (task_id, user_id, title, desc, priority, start, end)
        )
        conn.commit()
        conn.close()
    
    def get_tasks(self, user_id: str) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def delete_task(self, task_id: str):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        conn.close()
    
    def update_task_status(self, task_id: str, status: str):
        """Update task status (pending/completed)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            (status, task_id)
        )
        conn.commit()
        conn.close()

    # Metrics operations
    def add_metric(self, user_id: str, metric_type: str, value: float):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO metrics (user_id, metric_type, value) VALUES (?, ?, ?)",
            (user_id, metric_type, value)
        )
        conn.commit()
        conn.close()
    
    def get_metrics(self, user_id: str) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM metrics WHERE user_id = ? ORDER BY recorded_at DESC LIMIT 100",
                      (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # Agent logs
    def add_agent_log(self, user_id: str, agent: str, action: str, details: str = ""):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO agent_logs (user_id, agent_name, action, details) VALUES (?, ?, ?, ?)",
            (user_id, agent, action, details)
        )
        conn.commit()
        conn.close()
    
    def get_agent_logs(self, user_id: str) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM agent_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
                      (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # Chat operations
    def add_chat(self, user_id: str, role: str, message: str):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_messages (user_id, role, message) VALUES (?, ?, ?)",
            (user_id, role, message)
        )
        conn.commit()
        conn.close()
    
    def get_chat(self, user_id: str) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chat_messages WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
                      (user_id,))
        rows = cursor.fetchall()
        conn.close()
        result = [dict(row) for row in rows]
        result.reverse()
        return result
    


def create_demo_user(db: SimpleDB, user_id: str):
    """Create demo user with sample data"""
    db.create_user(user_id, "Demo User", "demo@laura.ai", "LAURA Demo")
    
    # Add demo metrics
    db.add_metric(user_id, "Sleep", 78)
    db.add_metric(user_id, "Focus", 92)
    db.add_metric(user_id, "Stress", 35)
    db.add_metric(user_id, "Productivity", 88)
    
    # Add demo tasks
    db.add_task(user_id, "task_1", "Physics Study", "Chapter 5 Review", "High", "09:00", "10:30")
    db.add_task(user_id, "task_2", "Team Meeting", "Project Discussion", "Medium", "14:00", "15:00")
    db.add_task(user_id, "task_3", "Code Review", "PR #234", "High", "16:00", "16:45")
    
    # Add demo agent logs
    db.add_agent_log(user_id, "Task Agent", "Analyzed", "Physics study - High priority")
    db.add_agent_log(user_id, "Schedule Agent", "Scheduled", "9:00 AM - Peak focus time")
    db.add_agent_log(user_id, "Coordinator", "Approved", "Task scheduled successfully")
