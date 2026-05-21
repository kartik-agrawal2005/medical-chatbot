"""
crud.py — Chat History CRUD Operations

Provides the interface between the agent and the SQLite database.
All chat history operations go through this module.

Operations:
    CREATE  →  save_message()           Store a new user/assistant message
    READ    →  get_last_n_messages()    Retrieve last N messages for context
    READ    →  get_full_session()       Get complete session history
    READ    →  get_all_sessions()       List all sessions with previews
    DELETE  →  delete_session()         Remove a session's history
    DELETE  →  clear_all_history()      Wipe the entire database

Why Last 5 Messages?
    LLaMA-3.1-8B has an 8K context window. Each message averages ~100 tokens,
    so 5 messages ≈ 500 tokens of history — leaving plenty of room for the 
    system prompt (~200 tokens), retrieved chunks (~600 tokens), and the
    model's response. This is a deliberate trade-off: enough context for 
    conversational continuity without starving the model of generation space.
    
    Think of it like a sliding window over a stream — we maintain a fixed-size 
    buffer of recent context, similar to how TCP uses a sliding window for 
    flow control.
"""

import uuid
from datetime import datetime, timezone

from memory.schema import get_connection, initialize_database, DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------

def save_message(
    session_id: str,
    role: str,
    content: str,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Save a single message to the chat history.
    
    Each message is tagged with a session_id (to group conversations)
    and a role (to distinguish user queries from assistant responses).
    
    Args:
        session_id: UUID string identifying the conversation session.
        role:       Either 'user' or 'assistant'.
        content:    The message text.
        db_path:    Path to the SQLite database file.
    
    Returns:
        The auto-generated row ID of the inserted message.
    
    Raises:
        ValueError: If role is not 'user' or 'assistant'.
    """
    if role not in ("user", "assistant"):
        raise ValueError(
            f"Invalid role: '{role}'. Must be 'user' or 'assistant'."
        )
    
    if not content.strip():
        raise ValueError("Message content cannot be empty.")
    
    conn = get_connection(db_path)
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_history (session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?);
            """,
            (
                session_id,
                role,
                content.strip(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cursor.lastrowid
        
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------

def get_last_n_messages(
    session_id: str,
    n: int = 5,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Retrieve the last N messages for a given session.
    
    This is the primary read operation — called before every agent 
    invocation to provide conversational context. The messages are 
    returned in chronological order (oldest first) so they can be 
    directly appended to the prompt.
    
    Implementation Detail:
        We use a subquery to get the last N rows (ordered by timestamp 
        DESC), then reverse the order in the outer query. This is more 
        efficient than fetching all rows and slicing in Python, especially 
        as the history grows.
        
        It's similar to how you'd find the K-th largest element — 
        rather than sorting everything, you use a targeted approach.
    
    Args:
        session_id: UUID string identifying the conversation session.
        n:          Number of recent messages to retrieve (default: 5).
        db_path:    Path to the SQLite database file.
    
    Returns:
        List of dicts with keys: 'role', 'content', 'timestamp'
        Ordered chronologically (oldest → newest).
    """
    conn = get_connection(db_path)
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT role, content, timestamp
            FROM (
                SELECT id, role, content, timestamp
                FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            )
            ORDER BY timestamp ASC, id ASC;
            """,
            (session_id, n),
        )
        
        rows = cursor.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]
        
    finally:
        conn.close()


def get_full_session(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Retrieve the complete message history for a session.
    
    Unlike get_last_n_messages(), this returns ALL messages — useful 
    for displaying full chat history in the frontend.
    
    Args:
        session_id: UUID string identifying the conversation session.
        db_path:    Path to the SQLite database file.
    
    Returns:
        List of dicts with keys: 'id', 'role', 'content', 'timestamp'
        Ordered chronologically (oldest → newest).
    """
    conn = get_connection(db_path)
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, role, content, timestamp
            FROM chat_history
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC;
            """,
            (session_id,),
        )
        
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]
        
    finally:
        conn.close()


def get_all_sessions(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    List all conversation sessions with a preview.
    
    Returns each session with:
        - session_id:     The unique session identifier
        - message_count:  Total messages in the session
        - first_message:  Preview of the first user message (truncated)
        - last_active:    Timestamp of the most recent message
    
    Useful for the frontend sidebar to display a list of past 
    conversations, similar to ChatGPT's sidebar.
    
    Args:
        db_path: Path to the SQLite database file.
    
    Returns:
        List of dicts, ordered by last activity (most recent first).
    """
    conn = get_connection(db_path)
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                session_id,
                COUNT(*) as message_count,
                MIN(CASE WHEN role = 'user' THEN content END) as first_message,
                MAX(timestamp) as last_active
            FROM chat_history
            GROUP BY session_id
            ORDER BY last_active DESC;
            """,
        )
        
        rows = cursor.fetchall()
        return [
            {
                "session_id": row["session_id"],
                "message_count": row["message_count"],
                "first_message": (
                    row["first_message"][:80] + "..."
                    if row["first_message"] and len(row["first_message"]) > 80
                    else row["first_message"]
                ),
                "last_active": row["last_active"],
            }
            for row in rows
        ]
        
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------

def delete_session(
    session_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Delete all messages for a given session.
    
    Args:
        session_id: UUID string identifying the conversation session.
        db_path:    Path to the SQLite database file.
    
    Returns:
        Number of messages deleted.
    """
    conn = get_connection(db_path)
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM chat_history WHERE session_id = ?;",
            (session_id,),
        )
        conn.commit()
        return cursor.rowcount
        
    finally:
        conn.close()


def clear_all_history(db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Delete ALL chat history from the database.
    
    ⚠️  This is destructive — use with caution.
    
    Args:
        db_path: Path to the SQLite database file.
    
    Returns:
        Number of messages deleted.
    """
    conn = get_connection(db_path)
    
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history;")
        conn.commit()
        return cursor.rowcount
        
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helper: Generate Session ID
# ---------------------------------------------------------------------------

def create_session_id() -> str:
    """
    Generate a new unique session ID.
    
    Uses UUID4 (random) — collision probability is astronomically low
    (2^-122 ≈ 10^-37). Safe for our use case.
    
    Returns:
        A UUID4 string like '550e8400-e29b-41d4-a716-446655440000'.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# CLI: Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  === Chat History CRUD Test ===\n")
    
    # Initialize the database
    initialize_database()
    
    # Create a test session
    test_session = create_session_id()
    print(f"  📝 Test session: {test_session[:8]}...\n")
    
    # Simulate a conversation
    messages = [
        ("user", "What are the symptoms of diabetes?"),
        ("assistant", "Common symptoms of diabetes include increased thirst (polydipsia), frequent urination (polyuria), unexplained weight loss, fatigue, and blurred vision. Type 1 diabetes symptoms tend to develop quickly, while Type 2 symptoms may develop gradually."),
        ("user", "What about treatment options?"),
        ("assistant", "Treatment depends on the type. Type 1 requires insulin therapy. Type 2 management typically starts with lifestyle modifications (diet, exercise) and may include oral medications like metformin. Regular blood glucose monitoring is essential for both types."),
        ("user", "Is it hereditary?"),
        ("assistant", "Yes, genetics play a role. Type 2 diabetes has a stronger hereditary component — if a parent has it, your risk increases significantly. Type 1 also has genetic factors, but environmental triggers (possibly viral infections) are thought to initiate the autoimmune response."),
    ]
    
    for role, content in messages:
        msg_id = save_message(test_session, role, content)
        print(f"  💾 Saved [{role:>9}] → id={msg_id}")
    
    # Test: Get last 3 messages (what the agent would see)
    print(f"\n  --- Last 3 messages (agent context) ---")
    recent = get_last_n_messages(test_session, n=3)
    for msg in recent:
        preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
        print(f"  [{msg['role']:>9}] {preview}")
    
    # Test: Get all sessions
    print(f"\n  --- All sessions ---")
    sessions = get_all_sessions()
    for s in sessions:
        print(f"  🗂️  {s['session_id'][:8]}... | {s['message_count']} msgs | {s['first_message']}")
    
    # Cleanup test data
    deleted = delete_session(test_session)
    print(f"\n  🗑️  Cleaned up: {deleted} test messages deleted")
    print("  Done!\n")
