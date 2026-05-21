"""
schema.py — SQLite Database Schema & Connection Management

Creates and manages the SQLite database for storing chat history.
SQLite is the perfect choice here — zero-config, file-based, ACID-compliant,
and fast enough for single-user chat history.

Schema:
    chat_history table:
        - id          INTEGER PRIMARY KEY  (auto-increment row identifier)
        - session_id  TEXT                 (groups messages by conversation)
        - role        TEXT                 (either 'user' or 'assistant')
        - content     TEXT                 (the actual message text)
        - timestamp   TEXT                 (ISO-8601 format for sorting)

Design Notes:
    - session_id acts like a partition key — we always query by session, 
      so we index on it for O(log N) lookups instead of full table scans.
    - Timestamps use ISO-8601 strings rather than Unix epochs because 
      SQLite's datetime functions work natively with them, and they're 
      human-readable when inspecting the DB directly.
    - We use WAL (Write-Ahead Logging) journal mode for better concurrent 
      read performance — the agent can read history while we're writing 
      new messages without blocking.
"""

import os
import sqlite3
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default path: project_root/database/chat_history.db
DEFAULT_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database",
)
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "chat_history.db")


# ---------------------------------------------------------------------------
# Schema Definition
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

# Index on session_id for fast lookups — queries always filter by session
CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id
ON chat_history (session_id);
"""

# Composite index for the most common query pattern: 
# "get last N messages for a session, ordered by time"
CREATE_COMPOSITE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chat_history_session_timestamp
ON chat_history (session_id, timestamp DESC);
"""


# ---------------------------------------------------------------------------
# Connection & Initialization
# ---------------------------------------------------------------------------

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Create a connection to the SQLite database.
    
    If the database file doesn't exist, it will be created automatically.
    The database directory is also created if needed.
    
    Configuration:
        - row_factory = sqlite3.Row  →  results behave like dicts
        - journal_mode = WAL         →  concurrent reads during writes
        - foreign_keys = ON          →  enforce referential integrity
    
    Args:
        db_path: Path to the SQLite database file.
    
    Returns:
        sqlite3.Connection configured for our use case.
    """
    # Ensure the database directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    # Row factory: access columns by name (row["content"]) instead of index
    conn.row_factory = sqlite3.Row
    
    # Enable WAL mode for better read concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    
    # Enforce foreign key constraints (good practice even if unused now)
    conn.execute("PRAGMA foreign_keys=ON;")
    
    return conn


def initialize_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Create the chat_history table and indexes if they don't already exist.
    
    This is idempotent — safe to call multiple times. Uses 
    CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS, so 
    re-running on an existing database is a no-op.
    
    Args:
        db_path: Path to the SQLite database file.
    """
    conn = get_connection(db_path)
    
    try:
        cursor = conn.cursor()
        
        # Create the table
        cursor.execute(CREATE_TABLE_SQL)
        
        # Create indexes for query performance
        cursor.execute(CREATE_INDEX_SQL)
        cursor.execute(CREATE_COMPOSITE_INDEX_SQL)
        
        conn.commit()
        print(f"  ✅ Database initialized: {db_path}")
        
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_db_stats(db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Get basic statistics about the database.
    
    Returns:
        Dictionary with keys:
            - total_messages:  Total number of messages stored.
            - total_sessions:  Number of unique sessions.
            - db_size_kb:      Database file size in kilobytes.
    """
    conn = get_connection(db_path)
    
    try:
        cursor = conn.cursor()
        
        # Total messages
        cursor.execute("SELECT COUNT(*) FROM chat_history;")
        total_messages = cursor.fetchone()[0]
        
        # Total unique sessions
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM chat_history;")
        total_sessions = cursor.fetchone()[0]
        
        # File size
        db_size_kb = os.path.getsize(db_path) / 1024 if os.path.exists(db_path) else 0
        
        return {
            "total_messages": total_messages,
            "total_sessions": total_sessions,
            "db_size_kb": round(db_size_kb, 2),
        }
        
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI: Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  === Database Schema Initialization ===\n")
    initialize_database()
    
    stats = get_db_stats()
    print(f"  📊 Stats: {stats}")
    print(f"\n  Database location: {DEFAULT_DB_PATH}")
    print("  Done!\n")
