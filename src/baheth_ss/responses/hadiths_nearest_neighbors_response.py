from pydantic import BaseModel, Field


class HadithsNearestNeighborsResponse(BaseModel):
    hadith_index: int
    limit: int = Field(ge=0, le=100, default=10)
    nearest_neighbors: list[int]
