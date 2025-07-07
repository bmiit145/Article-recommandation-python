import os
from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv
from uuid import uuid4
from typing import List
from app.embedder import get_embedding

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "blog_collection")

qdrant_client = QdrantClient(host=QDRANT_HOST.replace("http://", ""), port=QDRANT_PORT)