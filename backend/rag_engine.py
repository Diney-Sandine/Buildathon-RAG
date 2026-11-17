import os
from typing import List, AsyncGenerator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever

from backend.config import (
    OPENAI_API_KEY,
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL
)

# Initialize Embeddings and LLM (temperature set to 0.7)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7, streaming=True, openai_api_key=OPENAI_API_KEY)

# Global in-memory chat history session store per session ID
chat_histories = {}


def get_vector_store():
    """Load local FAISS index if it exists, otherwise return None."""
    index_file = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    if os.path.exists(index_file):
        try:
            return FAISS.load_local(
                FAISS_INDEX_DIR, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None
    return None


def process_and_add_pdf(pdf_path: str):
    """Load a new PDF, split into chunks, embed, and update local FAISS index."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    vector_store = get_vector_store()
    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)

    # Save locally
    vector_store.save_local(FAISS_INDEX_DIR)
    return len(chunks)


def get_conversational_rag_chain():
    """Builds history-aware retriever and question-answering chain."""
    vector_store = get_vector_store()
    if not vector_store:
        return None

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 1. Contextualize question prompt (rewrites user question based on chat history)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Answer question prompt customized for Indian & Tamil Nadu Agricultural Schemes
    system_prompt = (
        "You are **Uzhavan Sahayak (உழவன் சகாயக்)**, an expert, compassionate, and helpful AI assistant "
        "for farmers in India and Tamil Nadu. Your goal is to explain agricultural government schemes, "
        "eligibility criteria, subsidies, required documents, and application steps clearly and accurately.\n\n"
        "Use the retrieved context to answer the farmer's question. If you do not know the answer "
        "based on the documents, politely state that the uploaded scheme documents do not contain that information "
        "and suggest checking with the local Assistant Director of Agriculture (ADA) or Uzhavan app.\n\n"
        "Keep your tone respectful, encouraging, and easy to understand. Use bullet points for steps, "
        "eligibility, and documents.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


async def ask_rag_stream(question: str, session_id: str = "default_farmer") -> AsyncGenerator[str, None]:
    """Streams response tokens from the RAG chain while managing chat history."""
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chain = get_conversational_rag_chain()
    if not chain:
        yield "⚠️ No agricultural scheme documents have been uploaded yet! Please upload scheme PDFs using the sidebar on the left."
        return

    history = chat_histories[session_id]
    full_response = ""

    try:
        # Stream response from chain
        async for chunk in chain.astream({
            "input": question,
            "chat_history": history
        }):
            if "answer" in chunk:
                full_response += chunk["answer"]
                yield chunk["answer"]

        # Save both user question and AI answer to session memory
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=full_response))
    except Exception as e:
        yield f"An error occurred while processing your request: {str(e)}"
