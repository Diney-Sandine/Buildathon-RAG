import os
from dotenv import load_dotenv

# Load environment variables from root .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded_pdfs")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
