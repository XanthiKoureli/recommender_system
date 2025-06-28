from typing import Optional
from pydantic import BaseModel


class ScientificAbstract(BaseModel):
    doi: Optional[str]
    title: Optional[str]
    # authors: Optional[list]
    authors: Optional[str] = None
    year: Optional[int]
    abstract_content: str
    pmid: Optional[int]


class UserQueryRecord(BaseModel):
    user_query_id: str
    user_query: str
    query_simplified: str
    medical_notes: Optional[str] = None
