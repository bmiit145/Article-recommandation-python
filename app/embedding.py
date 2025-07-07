from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def get_embedding(text: str) -> list[float]:
    return model.encode(text).tolist()