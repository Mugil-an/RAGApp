from pydantic import BaseModel
from typing import Optional, List

class RAGChunkAndSrc(BaseModel):
    chunks: List[str]
    source_id: Optional[str] = None

class RAGUpsertResult(BaseModel):
    ingested: int

class RAGSearchResult(BaseModel):
    context: List[str]
    sources: List[str]

class RAGQueryResult(BaseModel):
    ans: str
    sources: List[str]
    num_contexts: int