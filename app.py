from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
def embed_text(req: TextRequest):
    vector = model.encode([req.text])[0].tolist()
    return {"vector": vector}

@app.get("/")
def test():
    return {"message": "success"}