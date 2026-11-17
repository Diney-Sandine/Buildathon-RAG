# Buildathon-RAG

# 🌾 Uzhavan Sahayak (உழவன் சகாயக்)
### AI-Powered Agricultural Schemes Assistant for Farmers (India & Tamil Nadu)

**Uzhavan Sahayak** is an intelligent, conversational Retrieval-Augmented Generation (RAG) application built specifically to help farmers easily discover, understand, and navigate central Indian and Tamil Nadu state government agricultural schemes, subsidies, loans, eligibility rules, and application procedures.

---

## 🚀 Key Features
- **PDF Scheme Ingestion:** Upload government scheme booklets and documents directly through the user interface.
- **Local FAISS Vector Store:** Embeds document chunks using OpenAI embeddings (`text-embedding-3-small`) and persists them locally.
- **Conversational Memory & Streaming:** Supports history-aware retrieval for contextual follow-up questions and streams answers in real-time.
- **Decoupled Architecture:** Built with a **FastAPI** backend and a **Streamlit** frontend.

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** FastAPI, Uvicorn
- **Orchestration & RAG:** LangChain, LangChain-Classic
- **LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
- **Vector Database:** FAISS-CPU (Local Storage)

---

## 📁 Project Structure

```text
Buildathon-RAG/
├── backend/
│   ├── __init__.py
│   ├── config.py         # Centralized paths and configuration
│   ├── main.py           # FastAPI endpoints (/upload, /chat, /documents)
│   └── rag_engine.py     # LangChain RAG pipeline, FAISS, memory & streaming
├── data/
│   ├── uploaded_pdfs/    # Source scheme PDFs
│   └── faiss_index/      # Local FAISS vector database
├── frontend/
│   └── app.py            # Streamlit user interface (Sidebar + Chat)
├── venv/                 # Python virtual environment
├── .env                  # Environment variables (OpenAI API Key)
└── requirements.txt      # Project dependencies

