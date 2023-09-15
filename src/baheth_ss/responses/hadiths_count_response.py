from pydantic import BaseModel


class HadithsCountResponse(BaseModel):
    hadiths_count: int
