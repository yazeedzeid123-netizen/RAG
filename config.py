import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION = "class12_rag"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # runs locally, no key needed
LLM_REPO = os.getenv("HF_LLM_REPO", "Qwen/Qwen2.5-1.5B-Instruct")
