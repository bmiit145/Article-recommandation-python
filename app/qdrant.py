import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList, PointStruct, VectorParams, Distance, Filter, MatchValue, FieldCondition
from app.config import QDRANT_HOST, COLLECTION_NAME
from uuid import UUID

client = QdrantClient(url=QDRANT_HOST)


def create_collection_if_not_exists():
    collections = client.get_collections().collections
    if not any(col.name == COLLECTION_NAME for col in collections):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,  # embedding vector size
                distance=Distance.COSINE
            )
        )


create_collection_if_not_exists()


def point_exists(article_id: str) -> bool:
    """
    Check if a blog vector already exists based on article_id in payload.
    """
    hits = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[{"key": "article_id", "match": {"value": article_id}}]
        ),
        limit=1,
        with_payload=True
    )
    return len(hits[0]) > 0


def upsert_blog_vector(id: str, vector: list[float], payload: dict):
    vector_id = str(uuid.uuid4())
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=vector_id,
                vector=vector,
                payload={
                    "article_id": id,
                    **payload
                }
            )
        ]
    )
    return vector_id


def search_similar(vector: list[float], top_k: int = 5 , score_threshold: float = 0.3):
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        score_threshold=score_threshold,
    )


def get_vector_id_by_article_id(article_id: str) -> str:
    result, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="article_id",
                    match=MatchValue(value=article_id)
                )
            ]
        ),
        with_payload=False,
        limit=1
    )

    if not result:
        raise ValueError(f"No vector found with article_id: {article_id}")

    vector_id = result[0].id
    print(f"Vector ID for article_id '{article_id}': {vector_id} ({type(vector_id)})")

    return str(vector_id)  # You can cast it to string if needed


def delete_by_vector_id(vector_id: str):
    # try:
    #     # point_id = UUID(vector_id)
    # except Exception:
    #     raise ValueError(f"Invalid vector_id: {vector_id} is not a valid UUID")

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[str(vector_id)])
    )

    return {"message": f"Vector {vector_id} deleted successfully"}


def delete_by_article_id(article_id: str):
    print(f"Deleting vector for article_id={article_id}")
    vector_id = get_vector_id_by_article_id(article_id)
    print(f"Deleting vector with ID: {vector_id} for article_id={article_id}")
    delete_by_vector_id(vector_id)
    return {"message": f"Deleted vector for article_id={article_id}"}

def truncate_collection():
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=[])  # Empty filter matches everything
    )
