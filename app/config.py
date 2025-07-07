import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "Priyank@8414#9898038051")

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "blogs")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")