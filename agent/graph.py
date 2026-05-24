"""
graph.py — ReAct Agent Graph (LangGraph StateGraph)

Wires the full ReAct loop using LangGraph's StateGraph:

    START → reason → should_use_tool?
              ↓ yes               ↓ no
          tool_execute         respond
              ↓
          reason  ← (loop back with Observation)

This is an explicit cyclic graph — not a simple chain. The agent can 
loop back from tool_execute to reason multiple times (up to MAX_ITERATIONS)
before finally responding. This is what makes it "agentic" — it decides 
its own control flow at runtime.

LangGraph Concepts:
    - StateGraph:       Defines the graph structure (nodes + edges)
    - Nodes:            Python functions that transform the state
    - Conditional Edge: Runtime branching based on state values
    - Compile:          Converts the graph definition into a runnable

The graph uses Groq's API for fast LLM inference with LLaMA-3.1-8B.
Groq runs models on their LPU hardware at ~500 tokens/sec — fast enough
for real-time chat without GPU costs.
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq

from agent.state import AgentState, create_initial_state
from agent.tools import get_medical_kb_tool
from agent.prompts import (
    SYSTEM_PROMPT,
    build_agent_prompt,
    build_tool_decision_prompt,
)
from memory.crud import get_last_n_messages, save_message

# Load environment variables (.env file)
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Max ReAct loop iterations (prevent infinite tool-calling cycles)
MAX_ITERATIONS = 3

# Groq client — initialized lazily
_groq_client = None


def _get_groq_client() -> Groq:
    """Get or create the Groq API client."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call LLaMA-3.1-8B via Groq API.
    
    Args:
        system_prompt: The system-level instructions.
        user_prompt:   The user-level prompt (query + context).
    
    Returns:
        The model's response text.
    """
    client = _get_groq_client()
    
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,     # Low temperature for factual medical responses
        max_tokens=1024,     # Enough for detailed medical explanations
        top_p=0.9,
    )
    
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

def reason_node(state: AgentState) -> AgentState:
    """
    Node 1: Reasoning — Decide whether to use the search tool or respond.
    
    The agent analyzes the user's query and decides:
        - SEARCH: <query>  →  route to tool_execute node
        - RESPOND           →  route to respond node
    
    This is the "Thought" step in the ReAct framework.
    """
    print(f"  🧠 Reasoning (iteration {state['iteration']})...")
    
    # Build the decision prompt
    decision_prompt = build_tool_decision_prompt(
        user_query=state["user_query"],
        chat_history=state["messages"],
    )
    
    # If we already have retrieved chunks (from a previous iteration),
    # include them so the agent knows it already searched
    if state["retrieved_chunks"]:
        decision_prompt += "\n\nNote: You already searched and found relevant information. You should RESPOND now unless you need different information."
    
    # Ask the LLM to decide
    decision = _call_llm(SYSTEM_PROMPT, decision_prompt)
    print(f"     → Decision: {decision[:80]}")
    
    # Parse the decision
    decision_upper = decision.strip().upper()
    
    if decision_upper.startswith("SEARCH:") or decision_upper.startswith("SEARCH "):
        # Extract the search query from "SEARCH: diabetes symptoms"
        search_query = re.sub(r'^SEARCH[:\s]+', '', decision.strip(), flags=re.IGNORECASE).strip()
        # Clean up quotes if present
        search_query = search_query.strip('"').strip("'")
        
        state["tool_decision"] = "search_kb"
        state["user_query_for_search"] = search_query if search_query else state["user_query"]
    else:
        state["tool_decision"] = "respond"
    
    state["iteration"] += 1
    return state


def tool_execute_node(state: AgentState) -> AgentState:
    """
    Node 2: Tool Execution — Run the FAISS retrieval tool.
    
    Searches the medical knowledge base using the query determined 
    by the reasoning node. The results are stored in state for the 
    response node to use.
    
    This is the "Action" + "Observation" step in ReAct.
    """
    search_query = state.get("user_query_for_search", state["user_query"])
    print(f"  🔍 Searching KB: \"{search_query}\"")
    
    tool = get_medical_kb_tool()
    results = tool.search(search_query)
    
    # Store structured results in state
    state["retrieved_chunks"] = results
    
    # Log what was found
    print(f"     → Retrieved {len(results)} chunks")
    for i, chunk in enumerate(results):
        preview = chunk.get("text", "")[:60]
        score = chunk.get("score", 0)
        print(f"       [{i+1}] (score: {score:.2f}) {preview}...")
    
    return state


def respond_node(state: AgentState) -> AgentState:
    """
    Node 3: Response Generation — Synthesize the final answer.
    
    Takes the user's query, chat history, and any retrieved context,
    then generates a comprehensive medical response via the LLM.
    
    This is the final "Response" step in ReAct.
    """
    print(f"  💬 Generating response...")
    
    # Format retrieved context (if any)
    retrieved_context = ""
    if state["retrieved_chunks"]:
        chunks_text = []
        for chunk in state["retrieved_chunks"]:
            source = chunk.get("source_file", "unknown")
            text = chunk.get("text", "")
            chunks_text.append(f"[Source: {source}]\n{text}")
        retrieved_context = "\n\n---\n\n".join(chunks_text)
    
    # Build the full prompt
    user_prompt = build_agent_prompt(
        user_query=state["user_query"],
        chat_history=state["messages"],
        retrieved_context=retrieved_context,
    )
    
    # Generate the response
    response = _call_llm(SYSTEM_PROMPT, user_prompt)
    
    state["agent_response"] = response
    print(f"     → Response generated ({len(response)} chars)")
    
    return state


# ---------------------------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------------------------

def should_use_tool(state: AgentState) -> str:
    """
    Conditional edge: Route to tool_execute or respond.
    
    Returns:
        "tool_execute" if the agent wants to search the KB.
        "respond" if the agent is ready to answer directly.
    """
    # Safety: cap iterations to prevent infinite loops
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"  ⚠️  Max iterations ({MAX_ITERATIONS}) reached — forcing response")
        return "respond"
    
    if state["tool_decision"] == "search_kb":
        return "tool_execute"
    else:
        return "respond"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph():
    """
    Build the ReAct agent graph.
    
    Graph Structure:
        START → reason → [conditional] → tool_execute → reason (loop)
                                       → respond → END
    
    Note: We implement this as a simple function-based pipeline rather 
    than using langgraph's StateGraph directly. This gives us the same 
    ReAct behavior with fewer dependencies — the graph is just a while 
    loop with routing logic.
    
    If you later want the full LangGraph visualization and debugging tools,
    the node functions (reason_node, tool_execute_node, respond_node) are 
    already compatible — just wire them into a StateGraph.
    
    Returns:
        A callable that takes an AgentState and returns the final state.
    """
    def run_graph(state: AgentState) -> AgentState:
        """Execute the ReAct loop."""
        
        while True:
            # Step 1: Reason — decide what to do
            state = reason_node(state)
            
            # Step 2: Route — tool or respond?
            route = should_use_tool(state)
            
            if route == "tool_execute":
                # Step 3a: Execute tool → loop back to reason
                state = tool_execute_node(state)
                # Continue the loop (reason again with new context)
                continue
            else:
                # Step 3b: Generate response → done
                state = respond_node(state)
                break
        
        return state
    
    return run_graph


# ---------------------------------------------------------------------------
# High-Level API
# ---------------------------------------------------------------------------

def chat(
    user_query: str,
    session_id: str,
    save_to_memory: bool = True,
) -> str:
    """
    Process a user query through the full agent pipeline.
    
    This is the main entry point — call this from the frontend.
    
    Pipeline:
        1. Load chat history from SQLite
        2. Create initial state
        3. Run the ReAct graph
        4. Save messages to SQLite
        5. Return the response
    
    Args:
        user_query:     The user's message.
        session_id:     Session ID for history persistence.
        save_to_memory: Whether to save messages to SQLite (default: True).
    
    Returns:
        The agent's response text.
    """
    print(f"\n{'='*60}")
    print(f"  📨 New query: \"{user_query[:50]}...\"")
    print(f"  🔑 Session: {session_id[:8]}...")
    print(f"{'='*60}")
    
    # Step 1: Load recent chat history
    messages = get_last_n_messages(session_id, n=5)
    print(f"  📜 Loaded {len(messages)} messages from history")
    
    # Step 2: Create initial state
    state = create_initial_state(
        user_query=user_query,
        session_id=session_id,
        messages=messages,
    )
    
    # Step 3: Run the ReAct graph
    graph = build_graph()
    final_state = graph(state)
    
    # Step 4: Save to memory
    if save_to_memory:
        save_message(session_id, "user", user_query)
        save_message(session_id, "assistant", final_state["agent_response"])
        print(f"  💾 Messages saved to SQLite")
    
    print(f"{'='*60}\n")
    
    return final_state["agent_response"]


# ---------------------------------------------------------------------------
# CLI: Interactive test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from memory.schema import initialize_database
    from memory.crud import create_session_id
    
    print("\n" + "="*60)
    print("  🏥 Medical Chatbot — CLI Test Mode")
    print("  Type 'quit' to exit, 'new' for a new session")
    print("="*60 + "\n")
    
    # Initialize database
    initialize_database()
    
    # Create a session
    session_id = create_session_id()
    print(f"  Session: {session_id[:8]}...\n")
    
    while True:
        try:
            query = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye! 👋")
            break
        
        if not query:
            continue
        
        if query.lower() == "quit":
            print("  Goodbye! 👋")
            break
        
        if query.lower() == "new":
            session_id = create_session_id()
            print(f"  🆕 New session: {session_id[:8]}...\n")
            continue
        
        try:
            response = chat(query, session_id)
            print(f"\n  Assistant: {response}\n")
        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")
