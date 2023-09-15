from pydantic import BaseModel, Field, validator


class HadithsSemanticSearchRequest(BaseModel):
    queries: str | list[str]
    limit: int = Field(ge=0, le=100, default=10)

    @validator('queries')
    def validate_queries(cls, queries: str | list[str]) -> list[str]:
        if isinstance(queries, str):
            queries = [queries]

        if len(queries) > 50 or max([len(query.split()) for query in queries]) > 25:
            raise ValueError('Too many queries or too long query.')

        return list(queries)
