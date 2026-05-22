"""
state.py — LangGraph Agent State Definition

Defines the TypedDict that flows through the LangGraph StateGraph.
Every node in the graph reads from and writes to this shared state — it's 
the single source of truth for the agent's current execution context.

LangGraph State Model:
    Unlike a simple function pipeline (input → process → output), LangGraph 
    uses a shared state object that gets passed between nodes. Each node can 
    read any field and write back updates. The graph runtime handles merging.

    Think of it like a blackboard architecture (from AI textbooks):
        - The state is the "blackboard"
        - Each node is a "knowledge source" that reads/writes to it
        - The graph controls which knowledge source runs next

State Fields:
    - messages:         Chat history (last N messages from SQLite)
    - user_query:       The current user input
    - retrieved_chunks: Documents fetched from FAISS by the retrieval tool
    - agent_response:   The final response to return to the user
    - session_id:       Links to SQLite for persistence
    - tool_calls:       Tracks which tools the agent decided to use
    - iteration:        Loop counter to prevent infinite ReAct cycles
"""

from typing import TypedDict


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Shared state for the ReAct agent graph.
    
    This TypedDict defines the schema of data flowing through the graph.
    LangGraph uses Python's typing system to validate state transitions.
    
    Flow:
        1. Frontend populates: messages, user_query, session_id
        2. Reasoning node reads: messages, user_query, retrieved_chunks
        3. Tool node writes: retrieved_chunks
        4. Response node writes: agent_response
    """
    
    # --- Input Fields (set before graph invocation) ---
    
    # Chat history: list of {"role": "user"|"assistant", "content": "..."}
    # Retrieved from SQLite's last N messages for this session
    messages: list[dict]
    
    # The current user query (the latest message being processed)
    user_query: str
    
    # Session identifier for SQLite persistence
    session_id: str
    
    # --- Internal Fields (managed by graph nodes) ---
    
    # Documents retrieved from FAISS by the search_medical_kb tool
    # Each dict: {"text": "...", "source_file": "...", "score": float}
    retrieved_chunks: list[dict]
    
    # Whether the agent decided to use the retrieval tool
    # Values: "search_kb" | "respond" | None
    tool_decision: str | None
    
    # Iteration counter to cap the ReAct loop
    # Prevents infinite cycles if the agent keeps calling tools
    # Max iterations: 3 (enough for most medical queries)
    iteration: int
    
    # --- Output Fields (returned after graph execution) ---
    
    # The agent's final response text to display to the user
    agent_response: str


# ---------------------------------------------------------------------------
# State Initialization Helper
# ---------------------------------------------------------------------------

def create_initial_state(
    user_query: str,
    session_id: str,
    messages: list[dict] | None = None,
) -> AgentState:
    """
    Create a fresh AgentState for a new query.
    
    This factory function ensures all fields are properly initialized 
    before the graph starts executing. Avoids KeyError from missing fields.
    
    Args:
        user_query: The user's current message.
        session_id: The session ID for SQLite lookups.
        messages:   Prior chat history (from get_last_n_messages).
                    Defaults to empty list if None.
    
    Returns:
        An initialized AgentState dict ready for graph invocation.
    """
    return AgentState(
        messages=messages or [],
        user_query=user_query,
        session_id=session_id,
        retrieved_chunks=[],
        tool_decision=None,
        iteration=0,
        agent_response="",
    )
