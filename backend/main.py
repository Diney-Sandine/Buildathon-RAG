import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import UPLOAD_DIR, FAISS_INDEX_DIR
from backend.rag_engine import process_and_add_pdf, ask_rag_stream, get_vector_store

app = FastAPI(
    title="Uzhavan Sahayak API",
    description="Backend API for Agricultural Schemes RAG Application",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default_farmer"


@app.get("/")
def read_root():
    return {"message": "Welcome to Uzhavan Sahayak (உழவன் சகாயக்) API!"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a scheme PDF, save it locally, and add it to the FAISS vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process and index PDF
        chunks_added = process_and_add_pdf(file_path)
        return {
            "filename": file.filename,
            "message": "Successfully uploaded and indexed PDF!",
            "chunks_added": chunks_added
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint supporting streaming responses and conversational memory."""
    async def event_generator():
        async for chunk in ask_rag_stream(request.question, request.session_id):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.get("/documents")
def list_documents():
    """List all uploaded scheme PDFs and check if vector store exists."""
    if not os.path.exists(UPLOAD_DIR):
        files = []
    else:
        files = os.listdir(UPLOAD_DIR)
    
    vector_store = get_vector_store()
    index_exists = vector_store is not None

    return {
        "uploaded_files": files,
        "vector_index_ready": index_exists
    }
