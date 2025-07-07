/ 1. Qdrant Setup (Docker)
// docker-compose.yml
version: '3.9'
services:
qdrant:
image: qdrant/qdrant
ports:
- "6333:6333"
volumes:
- qdrant_data:/qdrant/storage
volumes:
qdrant_data:


# Run FastAPI: uvicorn app:app --reload --port 8001

uvicorn app.main:app --reload  --port 8001


pip install fastapi uvicorn qdrant-client sentence-transformers python-dotenv numpy


