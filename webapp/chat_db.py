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
        # Direct messages
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

        # Group chats (book clubs)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS group_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER NOT NULL,
                book_title TEXT NOT NULL,
                created_by INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_group_book ON group_chats(book_id)")

        # Group chat members
        conn.execute("""
            CREATE TABLE IF NOT EXISTS group_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (group_id) REFERENCES group_chats(id),
                UNIQUE(group_id, user_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_group_members ON group_members(group_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_groups ON group_members(user_id)")

        # Group messages
        conn.execute("""
            CREATE TABLE IF NOT EXISTS group_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                from_user_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (group_id) REFERENCES group_chats(id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_group_messages ON group_messages(group_id, timestamp)")

        # Book club invitations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS invitations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_user_id INTEGER NOT NULL,
                to_user_id INTEGER NOT NULL,
                book_id INTEGER NOT NULL,
                book_title TEXT NOT NULL,
                group_id INTEGER,
                message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                seen INTEGER DEFAULT 0,
                FOREIGN KEY (group_id) REFERENCES group_chats(id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_invitations_to ON invitations(to_user_id, seen)")

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


# ============================================================================
# GROUP CHAT FUNCTIONS
# ============================================================================

def create_group_chat(book_id: int, book_title: str, created_by: int) -> int:
    """Create a new group chat for a book. Returns the group_id."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO group_chats (book_id, book_title, created_by) VALUES (?, ?, ?)",
            (book_id, book_title, created_by)
        )
        group_id = cursor.lastrowid
        # Add creator as first member
        conn.execute(
            "INSERT INTO group_members (group_id, user_id) VALUES (?, ?)",
            (group_id, created_by)
        )
        conn.commit()
        return group_id


def get_group_chat_for_book(book_id: int) -> dict:
    """Get the group chat for a book, if it exists."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM group_chats WHERE book_id = ? ORDER BY created_at DESC LIMIT 1",
            (book_id,)
        ).fetchone()
        return dict(row) if row else None


def join_group_chat(group_id: int, user_id: int) -> bool:
    """Add a user to a group chat. Returns True if successful."""
    with get_db() as conn:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO group_members (group_id, user_id) VALUES (?, ?)",
                (group_id, user_id)
            )
            conn.commit()
            return True
        except:
            return False


def leave_group_chat(group_id: int, user_id: int):
    """Remove a user from a group chat."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM group_members WHERE group_id = ? AND user_id = ?",
            (group_id, user_id)
        )
        conn.commit()


def is_group_member(group_id: int, user_id: int) -> bool:
    """Check if a user is a member of a group chat."""
    with get_db() as conn:
        result = conn.execute(
            "SELECT 1 FROM group_members WHERE group_id = ? AND user_id = ?",
            (group_id, user_id)
        ).fetchone()
        return result is not None


def get_group_members(group_id: int) -> list:
    """Get all members of a group chat."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT user_id, joined_at FROM group_members WHERE group_id = ? ORDER BY joined_at",
            (group_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def send_group_message(group_id: int, from_user_id: int, message: str) -> tuple:
    """Send a message to a group chat. Returns (message_id, timestamp)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO group_messages (group_id, from_user_id, message, timestamp) VALUES (?, ?, ?, ?)",
            (group_id, from_user_id, message, timestamp)
        )
        conn.commit()
        return cursor.lastrowid, timestamp


def get_group_messages(group_id: int, since: str = None, limit: int = 100) -> list:
    """Get messages from a group chat."""
    with get_db() as conn:
        if since:
            rows = conn.execute("""
                SELECT id, from_user_id, message, timestamp
                FROM group_messages
                WHERE group_id = ? AND timestamp > ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (group_id, since, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, from_user_id, message, timestamp
                FROM group_messages
                WHERE group_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (group_id, limit)).fetchall()
            rows = list(reversed(rows))  # Reverse to get oldest first
        return [dict(row) for row in rows]


def get_user_groups(user_id: int) -> list:
    """Get all group chats a user is a member of."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT gc.id, gc.book_id, gc.book_title, gc.created_at,
                   (SELECT COUNT(*) FROM group_members WHERE group_id = gc.id) as member_count,
                   (SELECT MAX(timestamp) FROM group_messages WHERE group_id = gc.id) as last_message_at
            FROM group_chats gc
            JOIN group_members gm ON gc.id = gm.group_id
            WHERE gm.user_id = ?
            ORDER BY last_message_at DESC NULLS LAST, gc.created_at DESC
        """, (user_id,)).fetchall()
        return [dict(row) for row in rows]


def get_group_by_id(group_id: int) -> dict:
    """Get group chat details by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM group_chats WHERE id = ?",
            (group_id,)
        ).fetchone()
        return dict(row) if row else None


# ============================================================================
# INVITATION FUNCTIONS
# ============================================================================

def send_invitation(from_user_id: int, to_user_id: int, book_id: int, book_title: str, group_id: int = None, message: str = None) -> int:
    """Send a book club invitation. Returns invitation ID."""
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO invitations (from_user_id, to_user_id, book_id, book_title, group_id, message)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (from_user_id, to_user_id, book_id, book_title, group_id, message)
        )
        conn.commit()
        return cursor.lastrowid


def get_user_invitations(user_id: int, unseen_only: bool = False) -> list:
    """Get all invitations for a user."""
    with get_db() as conn:
        if unseen_only:
            rows = conn.execute("""
                SELECT * FROM invitations
                WHERE to_user_id = ? AND seen = 0
                ORDER BY created_at DESC
            """, (user_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM invitations
                WHERE to_user_id = ?
                ORDER BY created_at DESC
                LIMIT 50
            """, (user_id,)).fetchall()
        return [dict(row) for row in rows]


def mark_invitation_seen(invitation_id: int):
    """Mark an invitation as seen."""
    with get_db() as conn:
        conn.execute(
            "UPDATE invitations SET seen = 1 WHERE id = ?",
            (invitation_id,)
        )
        conn.commit()


def mark_all_invitations_seen(user_id: int):
    """Mark all invitations for a user as seen."""
    with get_db() as conn:
        conn.execute(
            "UPDATE invitations SET seen = 1 WHERE to_user_id = ?",
            (user_id,)
        )
        conn.commit()


def get_unseen_invitation_count(user_id: int) -> int:
    """Get count of unseen invitations for a user."""
    with get_db() as conn:
        result = conn.execute(
            "SELECT COUNT(*) as count FROM invitations WHERE to_user_id = ? AND seen = 0",
            (user_id,)
        ).fetchone()
        return result['count']
