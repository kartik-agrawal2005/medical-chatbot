"""
streamlit_app.py — Streamlit Chat Frontend

The user-facing interface for the Medical Chatbot. Provides a ChatGPT-like
experience with:
    - Real-time chat with the ReAct agent
    - Session management (new chat, switch sessions)
    - Message history persistence via SQLite
    - Medical disclaimer banner
    - Responsive layout with sidebar

Streamlit Concepts:
    - st.session_state:  Dict that persists across reruns (button clicks, 
                         form submits). We store session_id and messages here.
    - st.chat_message:   Renders a message bubble with an avatar.
    - st.chat_input:     Fixed input bar at the bottom of the page.
    - st.sidebar:        Collapsible panel for session management.
    - st.spinner:        Loading indicator while the agent is thinking.
"""

import sys
import os

# Ensure project root is on Python path (needed when Streamlit runs from app/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from memory.schema import initialize_database
from memory.crud import (
    create_session_id,
    get_full_session,
    get_all_sessions,
    save_message,
    delete_session,
    get_last_n_messages,
)
from agent.graph import chat


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .main-header p {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0;
    }
    
    /* Disclaimer banner */
    .disclaimer-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #e74c3c44;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
        color: #ccc;
        text-align: center;
    }
    
    .disclaimer-banner strong {
        color: #e74c3c;
    }
    
    /* Session item in sidebar */
    .session-item {
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.3rem;
        cursor: pointer;
        transition: background 0.2s;
        border: 1px solid transparent;
    }
    
    .session-item:hover {
        background: #ffffff10;
        border-color: #667eea44;
    }
    
    .session-preview {
        font-size: 0.8rem;
        color: #999;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Sidebar title */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem 0;
        font-size: 0.75rem;
        color: #666;
        border-top: 1px solid #333;
        margin-top: 2rem;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------

@st.cache_resource
def init_db():
    """Initialize database once (cached across reruns)."""
    initialize_database()
    return True

init_db()


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize session state variables on first run."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = create_session_id()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

init_session_state()


# ---------------------------------------------------------------------------
# Sidebar: Session Management
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar with session management controls."""
    with st.sidebar:
        st.markdown('<div class="sidebar-title">💬 Chat Sessions</div>', 
                    unsafe_allow_html=True)
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.session_id = create_session_id()
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # List existing sessions
        sessions = get_all_sessions()
        
        if sessions:
            st.markdown(f"**Recent Conversations** ({len(sessions)})")
            
            for session in sessions:
                sid = session["session_id"]
                preview = session.get("first_message", "New conversation") or "New conversation"
                msg_count = session.get("message_count", 0)
                
                # Highlight active session
                is_active = sid == st.session_state.session_id
                label = f"{'🔵 ' if is_active else '💭 '}{preview[:35]}..."
                
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    if st.button(
                        label,
                        key=f"session_{sid}",
                        use_container_width=True,
                        disabled=is_active,
                    ):
                        # Switch to this session
                        st.session_state.session_id = sid
                        st.session_state.messages = get_full_session(sid)
                        st.rerun()
                
                with col2:
                    if st.button(
                        "🗑️",
                        key=f"delete_{sid}",
                        help="Delete this conversation",
                    ):
                        delete_session(sid)
                        if sid == st.session_state.session_id:
                            st.session_state.session_id = create_session_id()
                            st.session_state.messages = []
                        st.rerun()
        else:
            st.caption("No conversations yet. Start chatting!")
        
        # Sidebar footer
        st.divider()
        st.caption("🏥 Medical Chatbot v0.4.0")
        st.caption("Powered by LLaMA-3.1 + FAISS")

render_sidebar()


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Medical Chatbot</h1>
    <p>AI-powered medical assistant with RAG-based knowledge retrieval</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer-banner">
    <strong>⚠️ Medical Disclaimer:</strong> This chatbot provides general health 
    information for educational purposes only. It is <strong>not</strong> a substitute 
    for professional medical advice. Always consult a qualified healthcare provider 
    for medical decisions.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load & Display Chat History
# ---------------------------------------------------------------------------

def load_chat_history():
    """Load chat history from SQLite if not already in session state."""
    if not st.session_state.messages:
        history = get_full_session(st.session_state.session_id)
        if history:
            st.session_state.messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in history
            ]

load_chat_history()

# Display all messages
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🏥"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# Chat Input & Agent Response
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask me about medical conditions, symptoms, treatments..."):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Generate agent response
    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Searching medical knowledge base..."):
            try:
                response = chat(
                    user_query=prompt,
                    session_id=st.session_state.session_id,
                    save_to_memory=True,
                )
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_msg = f"⚠️ An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


# ---------------------------------------------------------------------------
# Welcome Message (empty state)
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Try asking:**
        > What are the symptoms of diabetes?
        """)
    
    with col2:
        st.markdown("""
        **💊 Or ask about:**
        > How is hypertension treated?
        """)
    
    with col3:
        st.markdown("""
        **🧠 Or explore:**
        > What causes migraine headaches?
        """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit • LLaMA-3.1-8B via Groq • FAISS HNSW • SQLite<br>
    © 2026 Kartik Agrawal
</div>
""", unsafe_allow_html=True)
