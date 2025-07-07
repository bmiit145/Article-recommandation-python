# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uuid

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "blogs")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

app = FastAPI()

class BlogInput(BaseModel):
    blog_id: str
    title: str
    short_description: Optional[str] = ""
    description: str
    tags: Optional[List[str]] = []
    category: Optional[str] = None

class RecommendInput(BaseModel):
    user_id: str
    viewed_blogs: List[str] = []

@app.on_event("startup")
def startup():
    # Ensure collection exists
    if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": 384, "distance": "Cosine"},
        )

@app.post("/embed")
def embed_blog(blog: BlogInput):
    content = f"{blog.title}. {blog.short_description} {blog.description}. Tags: {' '.join(blog.tags)} Category: {blog.category}"
    vector = model.encode(content).tolist()

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=blog.blog_id,
                vector=vector,
                payload={
                    "title": blog.title,
                    "tags": blog.tags,
                    "category": blog.category,
                }
            )
        ]
    )
    return {"status": "inserted", "id": blog.blog_id}

@app.delete("/delete/{blog_id}")
def delete_blog(blog_id: str):
    client.delete(collection_name=COLLECTION_NAME, points_selector={"points": [blog_id]})
    return {"status": "deleted", "id": blog_id}

@app.get("/search")
def search_similar(title: str, description: str):
    vector = model.encode(f"{title} {description}").tolist()
    hits = client.search(collection_name=COLLECTION_NAME, query_vector=vector, limit=5)
    return [{"id": hit.id, "score": hit.score, "payload": hit.payload} for hit in hits]

@app.post("/recommend")
def recommend(input: RecommendInput):
    if not input.viewed_blogs:
        raise HTTPException(status_code=400, detail="No viewed blogs provided")

    vectors = []
    for blog_id in input.viewed_blogs:
        try:
            vec = client.retrieve(collection_name=COLLECTION_NAME, ids=[blog_id])[0].vector
            vectors.append(vec)
        except:
            continue

    if not vectors:
        raise HTTPException(status_code=404, detail="No vectors found for given blog IDs")

    # Average of vectors
    import numpy as np
    mean_vector = np.mean(vectors, axis=0).tolist()
    hits = client.search(collection_name=COLLECTION_NAME, query_vector=mean_vector, limit=5, with_payload=True)
    return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]