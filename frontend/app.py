import uuid
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

# Page Configuration
st.set_page_config(
    page_title="Uzhavan Sahayak | Agricultural Schemes AI",
    page_icon="🌾",
    layout="wide"
)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Vanakkam! 🙏 I am **Uzhavan Sahayak (உழவன் சகாயக்)**. Upload your Indian or Tamil Nadu government agricultural scheme PDFs on the left, and ask me about subsidies, eligibility, loans, or application steps!"
        }
    ]


def fetch_indexed_documents():
    """Fetch list of uploaded PDFs from the FastAPI backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json().get("uploaded_files", [])
    except requests.exceptions.RequestException:
        return None
    return []


def stream_rag_response(question: str, session_id: str):
    """Generator that streams text chunks from FastAPI /chat endpoint."""
    try:
        with requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": question, "session_id": session_id},
            stream=True,
            timeout=60
        ) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk
            else:
                yield f"⚠️ Backend Error ({response.status_code}): Unable to fetch response."
    except requests.exceptions.ConnectionError:
        yield "⚠️ Cannot connect to the backend server. Please make sure FastAPI is running on port 8000."
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"


# ==========================================
# LEFT SIDEBAR: PDF UPLOAD & KNOWLEDGE BASE
# ==========================================
with st.sidebar:
    st.title("🌾 Uzhavan Sahayak")
    st.caption("உழவன் சகாயக் • Farmer Scheme Assistant")
    st.divider()

    st.subheader("📄 Upload Scheme PDFs")
    uploaded_files = st.file_uploader(
        "Select India / Tamil Nadu Agri Scheme PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Upload & Index PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please choose at least one PDF file first.")
        else:
            for pdf_file in uploaded_files:
                with st.spinner(f"Indexing {pdf_file.name}..."):
                    try:
                        files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                        res = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
                        if res.status_code == 200:
                            data = res.json()
                            st.success(f"✅ {pdf_file.name} ({data['chunks_added']} chunks indexed)")
                        else:
                            st.error(f"❌ Failed to upload {pdf_file.name}: {res.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Backend server is offline! Start FastAPI first.")
                        break

    st.divider()
    st.subheader("📚 Indexed Scheme Documents")
    docs = fetch_indexed_documents()
    if docs is None:
        st.error("🔴 FastAPI Backend Offline")
    elif len(docs) == 0:
        st.info("No PDFs indexed yet.")
    else:
        for doc_name in docs:
            st.markdown(f"- 📑 `{doc_name}`")

    st.divider()
    if st.button("🔄 Clear Chat History", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared! How can I assist you with agricultural schemes today?"
            }
        ]
        st.rerun()


# ==========================================
# CENTER / RIGHT MAIN AREA: CHAT INTERFACE
# ==========================================
st.header("🚜 Agricultural Schemes Search & Assistant")
st.markdown("Ask about crop subsidies, drip irrigation, farm machinery, Kisan Credit Card, or TN state schemes.")

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input Box
if prompt := st.chat_input("Ask about any farming scheme (e.g., 'What subsidy is available for drip irrigation in TN?')..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        response_text = st.write_stream(
            stream_rag_response(prompt, st.session_state.session_id)
        )

    # Save full assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    