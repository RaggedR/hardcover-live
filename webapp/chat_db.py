"""
Chat database module for Book Friends webapp.
Uses SQLite for message storage.
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

# Store chat database in the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "chat.db")


def init_db():
    """Create the messages table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_user_id INTEGER NOT NULL,
                to_user_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                read INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation ON messages(from_user_id, to_user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_to_user ON messages(to_user_id, read)")
        conn.commit()


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def send_message(from_user_id: int, to_user_id: int, message: str) -> tuple:
    """Send a message and return (message_id, timestamp)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO messages (from_user_id, to_user_id, message, timestamp) VALUES (?, ?, ?, ?)",
            (from_user_id, to_user_id, message, timestamp)
        )
        conn.commit()
        return cursor.lastrowid, timestamp


def get_conversation(user1_id: int, user2_id: int, since: str = None) -> list:
    """Get all messages between two users, optionally since a timestamp."""
    with get_db() as conn:
        if since:
            rows = conn.execute("""
                SELECT id, from_user_id, to_user_id, message, timestamp, read
                FROM messages
                WHERE ((from_user_id = ? AND to_user_id = ?)
                    OR (from_user_id = ? AND to_user_id = ?))
                  AND timestamp > ?
                ORDER BY timestamp ASC
            """, (user1_id, user2_id, user2_id, user1_id, since)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, from_user_id, to_user_id, message, timestamp, read
                FROM messages
                WHERE (from_user_id = ? AND to_user_id = ?)
                   OR (from_user_id = ? AND to_user_id = ?)
                ORDER BY timestamp ASC
            """, (user1_id, user2_id, user2_id, user1_id)).fetchall()

        return [dict(row) for row in rows]


def mark_as_read(user_id: int, from_user_id: int):
    """Mark all messages from a specific user as read."""
    with get_db() as conn:
        conn.execute(
            "UPDATE messages SET read = 1 WHERE to_user_id = ? AND from_user_id = ? AND read = 0",
            (user_id, from_user_id)
        )
        conn.commit()


def get_inbox(user_id: int) -> list:
    """Get all conversations for a user with the latest message preview."""
    with get_db() as conn:
        # Get all unique conversation partners
        rows = conn.execute("""
            SELECT
                CASE
                    WHEN from_user_id = ? THEN to_user_id
                    ELSE from_user_id
                END as other_user_id,
                MAX(timestamp) as last_timestamp
            FROM messages
            WHERE from_user_id = ? OR to_user_id = ?
            GROUP BY other_user_id
            ORDER BY last_timestamp DESC
        """, (user_id, user_id, user_id)).fetchall()

        conversations = []
        for row in rows:
            other_id = row['other_user_id']

            # Get the latest message
            latest = conn.execute("""
                SELECT message, from_user_id, timestamp
                FROM messages
                WHERE (from_user_id = ? AND to_user_id = ?)
                   OR (from_user_id = ? AND to_user_id = ?)
                ORDER BY timestamp DESC
                LIMIT 1
            """, (user_id, other_id, other_id, user_id)).fetchone()

            # Count unread messages
            unread = conn.execute("""
                SELECT COUNT(*) as count
                FROM messages
                WHERE to_user_id = ? AND from_user_id = ? AND read = 0
            """, (user_id, other_id)).fetchone()['count']

            conversations.append({
                'other_user_id': other_id,
                'last_message': latest['message'] if latest else '',
                'last_message_from_me': latest['from_user_id'] == user_id if latest else False,
                'last_timestamp': latest['timestamp'] if latest else '',
                'unread_count': unread
            })

        return conversations


def get_unread_count(user_id: int) -> int:
    """Get total unread message count for a user."""
    with get_db() as conn:
        result = conn.execute(
            "SELECT COUNT(*) as count FROM messages WHERE to_user_id = ? AND read = 0",
            (user_id,)
        ).fetchone()
        return result['count']
