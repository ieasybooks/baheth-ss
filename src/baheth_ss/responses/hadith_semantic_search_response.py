from pydantic import BaseModel


class HadithSemanticSearchResponse(BaseModel):
    query: str
    matching_hadiths: list[int]
