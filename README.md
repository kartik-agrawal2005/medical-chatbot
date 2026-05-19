# 🏥 Agentic Medical Chatbot with RAG

An intelligent medical assistant powered by a **ReAct** (Reasoning + Acting) agent that autonomously queries a vector database of medical documents to provide grounded, hallucination-resistant responses.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![LLaMA](https://img.shields.io/badge/LLM-LLaMA--3.1--8B-orange)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS%20HNSW-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)
![SQLite](https://img.shields.io/badge/Memory-SQLite-lightblue)
![Groq](https://img.shields.io/badge/Inference-Groq%20API-yellow)

---

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────┐     ┌──────────────────────┐
│  Streamlit   │────▶│   SQLite    │────▶│   ReAct Agent Loop   │
│  Chat UI     │     │  (History)  │     │   (LangGraph)        │
└──────────────┘     └─────────────┘     └──────────┬───────────┘
                                                     │
                                              ┌──────▼──────┐
                                              │ LLaMA-3.1-8B│
                                              │  (via Groq) │
                                              └──────┬──────┘
                                                     │
                                           Needs facts? ──Yes──▶ FAISS HNSW
                                                     │                │
                                              Final Response ◀── Top-3 Chunks
```

### Key Design Principles

- **Plain Text Boundary** — The LLM never sees embedding vectors. Embeddings exist only inside the FAISS retrieval tool.
- **ReAct Loop** — The agent explicitly reasons (`Thought`), decides to act (`Action: search_kb`), observes results (`Observation`), then synthesizes a grounded answer.
- **O(log N) Retrieval** — FAISS HNSW index enables approximate nearest-neighbor search in logarithmic time.
- **Session Persistence** — Chat history stored in SQLite with session management for multi-conversation support.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Chat UI with session sidebar |
| Orchestration | Custom ReAct Graph | Cyclic reasoning loop with tool routing |
| LLM | LLaMA-3.1-8B via Groq | Fast inference (~500 tok/s) |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | Semantic text encoding |
| Vector DB | FAISS (IndexHNSWFlat) | O(log N) approximate nearest-neighbor search |
| Memory | SQLite (WAL mode) | Session-based chat history persistence |
| Language | Python 3.11+ | ML/AI ecosystem |

---

## 📁 Project Structure

```
medical-chatbot/
├── knowledge_base/         # Phase 1: Document processing & vector index
│   ├── chunker.py          #   Text loading & chunking (PDF/TXT)
│   ├── embedder.py         #   MiniLM-L6-v2 embedding wrapper
│   └── index_builder.py    #   FAISS HNSW index construction
│
├── memory/                 # Phase 2: Chat history persistence
│   ├── schema.py           #   SQLite schema, WAL mode, indexed lookups
│   └── crud.py             #   CRUD operations + session management
│
├── agent/                  # Phase 3: ReAct agent orchestration
│   ├── state.py            #   AgentState TypedDict definition
│   ├── tools.py            #   FAISS retrieval tool (embed → search → format)
│   ├── prompts.py          #   System prompt + ReAct templates
│   └── graph.py            #   ReAct loop + Groq API + chat() entry point
│
├── app/                    # Phase 4: Streamlit frontend
│   └── streamlit_app.py    #   Chat UI, session sidebar, medical disclaimers
│
├── data/raw/               # Raw medical documents (PDF/TXT)
├── scripts/
│   └── build_index.py      # CLI: build the FAISS index
├── vectorstore/            # Generated: FAISS index + metadata (gitignored)
└── database/               # Generated: SQLite DB (gitignored)
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/kartik-agrawal2005/medical-chatbot.git
cd medical-chatbot
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Build the Knowledge Base

Place your medical documents (`.pdf` or `.txt`) in `data/raw/`, then:

```bash
python scripts/build_index.py
```

This chunks the documents, embeds them with MiniLM, and saves the FAISS HNSW index to `vectorstore/`.

### 3. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at: https://console.groq.com
```

### 4. Run the Chatbot

```bash
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501` with the chat interface.

### 5. CLI Mode (Alternative)

You can also test the agent directly in the terminal:

```bash
python -m agent.graph
```

---

## 🔧 Configuration

All configuration is managed via the `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 📋 Development Roadmap

- [x] **Phase 1** — Knowledge Engine (chunking, embeddings, FAISS HNSW index)
- [x] **Phase 2** — Memory Layer (SQLite chat history with CRUD)
- [x] **Phase 3** — Agent Orchestration (ReAct loop + Groq + FAISS tool)
- [x] **Phase 4** — Frontend (Streamlit chat UI with session management)

---

## ⚠️ Disclaimer

This chatbot is a **learning project** and is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Kartik Agrawal**  
GitHub: [@kartik-agrawal2005](https://github.com/kartik-agrawal2005)
