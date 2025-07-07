from pydantic import BaseModel
from typing import List, Optional

class BlogMetadata(BaseModel):
    id: Optional[str]
    title: str
    short_description: str
    description: str
    tags: List[str]
    category: Optional[str]

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5