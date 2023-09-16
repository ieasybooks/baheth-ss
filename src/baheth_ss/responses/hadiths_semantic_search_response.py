from pydantic import BaseModel, Field

from .hadith_semantic_search_response import HadithSemanticSearchResponse


class HadithsSemanticSearchResponse(BaseModel):
    limit: int = Field(ge=0, le=100, default=10)
    results: list[HadithSemanticSearchResponse]
