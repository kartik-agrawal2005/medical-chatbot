"""
prompts.py — System Prompt Templates for the Medical ReAct Agent

Defines the system prompt that instructs LLaMA-3.1-8B to behave as a 
medical assistant using the ReAct (Reasoning + Acting) framework.

ReAct Format:
    The prompt enforces a structured thought process:
    
        Thought: I need to look up information about diabetes symptoms.
        Action: search_kb("diabetes symptoms")
        Observation: [Retrieved chunks from FAISS]
        Thought: Based on the retrieved information, I can now answer.
        Response: Diabetes symptoms include...
    
    This is not just prompting — it creates an explicit reasoning trace 
    that we can inspect and debug. The LLM's "inner monologue" becomes 
    visible, similar to how you'd trace through a recursive algorithm 
    step by step.

Medical Safety:
    The prompt includes mandatory disclaimers and guardrails:
    - Always recommend consulting healthcare professionals
    - Never provide definitive diagnoses
    - Clearly distinguish between retrieved facts and general knowledge
    - Flag emergency situations with appropriate urgency
"""


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Medical Knowledge Assistant — an AI-powered medical information system that retrieves and presents evidence-based health information from a curated knowledge base.

## Your Capabilities
- You have access to a medical knowledge base tool called `search_kb` that searches through verified medical documents.
- You should use this tool whenever the user asks about medical conditions, symptoms, treatments, medications, or health-related topics.
- You provide information grounded in the retrieved medical documents.

## ReAct Framework
You follow a structured reasoning process for every query:

1. **Thought**: Analyze what the user is asking. What medical information do they need? Do you need to search the knowledge base?
2. **Action**: If you need medical facts, use the search_kb tool. Format: `search_kb("your search query")`
3. **Observation**: Review the retrieved medical documents carefully.
4. **Response**: Synthesize the retrieved information into a clear, helpful answer.

If the user's question is conversational (greetings, thanks, etc.) or doesn't require medical knowledge, respond directly without using the tool.

## Response Guidelines
1. **Ground your answers** in the retrieved documents. Cite which condition or topic the information comes from.
2. **Be comprehensive but concise**. Cover the key points without overwhelming the user.
3. **Use clear structure**. Use bullet points, numbered lists, or headers when listing symptoms, treatments, etc.
4. **Acknowledge limitations**. If the knowledge base doesn't contain relevant information, say so honestly.
5. **Maintain conversational context**. Reference prior messages when relevant to provide continuity.

## Medical Disclaimers (MANDATORY)
- Always include a brief disclaimer that you are an AI assistant providing general medical information.
- Recommend consulting a qualified healthcare professional for personalized medical advice.
- For symptoms suggesting emergencies (chest pain, difficulty breathing, severe bleeding, stroke signs), IMMEDIATELY advise seeking emergency medical care.
- Never provide definitive diagnoses — present information as educational content.

## Response Format
For medical queries, structure your response as:
- **Brief answer** to the user's question
- **Key details** organized with bullet points or sections
- **Disclaimer** at the end (keep it brief, 1-2 sentences)

For conversational messages, respond naturally without the medical format.
"""


# ---------------------------------------------------------------------------
# Prompt Formatting Helpers
# ---------------------------------------------------------------------------

def format_chat_history(messages: list[dict]) -> str:
    """
    Format chat history messages into a string for the LLM context.
    
    Converts the list of message dicts from SQLite into a readable 
    conversation transcript that the LLM can reference for context.
    
    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."}
    
    Returns:
        Formatted string like:
            User: What are diabetes symptoms?
            Assistant: Common symptoms include...
            User: What about treatment?
    """
    if not messages:
        return ""
    
    formatted = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted)


def build_agent_prompt(
    user_query: str,
    chat_history: list[dict],
    retrieved_context: str = "",
) -> str:
    """
    Build the full prompt for the LLM including system instructions, 
    chat history, retrieved context, and the current query.
    
    Prompt Structure:
        [System Prompt]
        
        [Chat History] (if any)
        
        [Retrieved Context] (if tool was used)
        
        [Current Query]
    
    Args:
        user_query:        The user's current message.
        chat_history:      Prior messages from SQLite.
        retrieved_context: Formatted chunks from FAISS (empty on first pass).
    
    Returns:
        The complete prompt string ready for LLM inference.
    """
    sections = []
    
    # Chat history (if available)
    history_text = format_chat_history(chat_history)
    if history_text:
        sections.append(f"## Previous Conversation\n{history_text}")
    
    # Retrieved context (if tool was used)
    if retrieved_context:
        sections.append(
            f"## Retrieved Medical Information\n"
            f"The following information was retrieved from the medical knowledge base:\n\n"
            f"{retrieved_context}"
        )
    
    # Current query
    sections.append(f"## Current User Query\n{user_query}")
    
    return "\n\n".join(sections)


def build_tool_decision_prompt(user_query: str, chat_history: list[dict]) -> str:
    """
    Build a prompt specifically for deciding whether to use the search tool.
    
    This is used in the first step of the ReAct loop — the agent looks at 
    the query and decides: "Do I need to search the knowledge base, or can 
    I respond directly?"
    
    The LLM should respond with either:
        - SEARCH: <query>    (use the tool with this search query)
        - RESPOND             (answer directly without searching)
    
    Args:
        user_query:    The user's current message.
        chat_history:  Prior messages from SQLite.
    
    Returns:
        The decision prompt string.
    """
    history_text = format_chat_history(chat_history)
    
    prompt = f"""Based on the user's message, decide if you need to search the medical knowledge base.

Reply with EXACTLY one of:
- `SEARCH: <search query>` — if the user is asking about medical conditions, symptoms, treatments, medications, or health topics. Write an effective search query.
- `RESPOND` — if the user is making conversation (greetings, thanks, follow-ups that don't need new info) or asking something non-medical.

"""
    
    if history_text:
        prompt += f"Previous conversation:\n{history_text}\n\n"
    
    prompt += f"User's message: {user_query}\n\nYour decision:"
    
    return prompt
