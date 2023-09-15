from pydantic import BaseModel, Field


class HadithSemanticSearchResponse(BaseModel):
    query: str
    matching_hadiths: list[int]


class HadithsSemanticSearchResponse(BaseModel):
    limit: int = Field(ge=0, le=100, default=10)
    results: list[HadithSemanticSearchResponse]
