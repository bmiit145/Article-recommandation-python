from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from app.config import COLLECTION_NAME, API_KEY
from fastapi.security.api_key import APIKeyHeader, APIKey
from app.embedding import get_embedding
from qdrant_client.models import Filter as QFilter
from app.qdrant import upsert_blog_vector, search_similar, create_collection_if_not_exists, point_exists, client, \
    truncate_collection, delete_by_article_id
from fastapi import Body
import numpy as np
from collections import Counter

app = FastAPI()

# API
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized access")
    return api_key


@app.on_event("startup")
def setup():
    create_collection_if_not_exists()


class BlogInput(BaseModel):
    id: str
    content: str
    metadata: Dict = {}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Blog Embedding API"}


@app.post("/embed")
def embed_blog(blog: BlogInput):
    try:
        embedding = get_embedding(blog.content)
        vector_id = upsert_blog_vector(id=blog.id, vector=embedding, payload=blog.metadata)
        return {
            "success": True,
            "message": "Blog embedded and stored",
            "id": blog.id,
            "vector_id": vector_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BulkBlogInput(BaseModel):
    id: str
    content: str
    metadata: Dict = {}


@app.post("/embed/bulk")
def embed_bulk_blogs(blogs: List[BulkBlogInput] = Body(...)):
    embedded = []
    skipped = []

    for blog in blogs:
        if point_exists(blog.id):
            skipped.append(blog.id)
            continue

        try:
            embedding = get_embedding(blog.content)
            vector_id = upsert_blog_vector(id=blog.id, vector=embedding, payload=blog.metadata)
            embedded.append({"article_id": blog.id, "vector_id": vector_id})
        except Exception as e:
            skipped.append({"article_id": blog.id, "error": str(e)})

    return {
        "success": True,
        "message": "Bulk embedding completed",
        "data": {
            "count": len(embedded),
            "embedded": embedded,
            "skipped": skipped
        }
    }


@app.get("/search")
def recommend(q: str = Query(...), top_k: int = 5, threshold: float = 0.30):
    try:
        embedding = get_embedding(q)
        results = search_similar(embedding, top_k=top_k, score_threshold=threshold)
        return {
            "recommendations": [r.payload for r in results]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inspect")
def inspect_all(limit: int = 10):
    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            with_payload=True
        )
        return [point.payload for point in points]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class HybridRecommendInput(BaseModel):
    articleIds: List[str]
    top_k: Optional[int] = Field(default=10, description="Number of top results to return")
    threshold: Optional[float] = Field(default=0, description="Minimum similarity score threshold")


@app.post("/recommend")
def hybrid_recommendation(input: HybridRecommendInput):
    try:
        seen_ids = set(input.articleIds)
        embeddings = []
        all_tags = []
        all_categories = []

        # Step 1: Extract embeddings + metadata from recent blogs
        for article_id in input.articleIds:
            points, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=QFilter(
                    must=[{"key": "article_id", "match": {"value": article_id}}]
                ),
                limit=1,
                with_payload=True,
                with_vectors=True
            )
            if points:
                point = points[0]
                embeddings.append(point.vector)
                payload = point.payload
                all_tags.extend(payload.get("tags", []))
                all_categories.append(payload.get("category", ""))

        if not embeddings:
            raise HTTPException(status_code=404, detail="No valid blog data found")

        composite_vector = np.mean(embeddings, axis=0).tolist()

        # Step 2: Get top N candidates by embedding
        threshold = input.threshold if input.threshold is not None else 0.3;
        results = search_similar(composite_vector, top_k=input.top_k + len(seen_ids) + 10, score_threshold=threshold);

        # Step 3: Metadata frequency
        tag_freq = Counter(all_tags)
        category_freq = Counter(all_categories)
        top_tags = set(tag_freq.keys())
        top_categories = set(category_freq.keys())

        # Step 4: Score and filter
        ranked = []
        for r in results:
            payload = r.payload
            if not payload or payload.get("article_id") in seen_ids:
                continue

            score = r.score

            # Boost score if category/tag matches
            if payload.get("category") in top_categories:
                score += 0.5
            if isinstance(payload.get("tags"), list):
                tag_matches = top_tags.intersection(payload.get("tags"))
                score += 0.2 * len(tag_matches)

            ranked.append({
                "article_id": payload.get("article_id"),
                "title": payload.get("title"),
                "category": payload.get("category"),
                "tags": payload.get("tags", []),
                "score": round(score, 4)
            })

        # Step 5: Return top K by boosted score
        recommendations = sorted(ranked, key=lambda x: x["score"], reverse=True)[:input.top_k]

        return {
            "strategy": "hybrid",
            "articleIds": input.articleIds,
            "top_categories": list(top_categories),
            "top_tags": list(top_tags),
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Delete articles
class ArticleDeleteRequest(BaseModel):
    article_id: str


@app.delete("/delete")
def delete_article(data: ArticleDeleteRequest, _: APIKey = Depends(validate_api_key)):
    try:
        if not point_exists(data.article_id):
            raise HTTPException(status_code=404, detail="Article not found")

        delete_by_article_id(data.article_id)
        return {"message": f"Article {data.article_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/truncate")
def delete_table(_: APIKey = Depends(validate_api_key)):
    try:
        truncate_collection()
        return {"message": "Collection truncated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
