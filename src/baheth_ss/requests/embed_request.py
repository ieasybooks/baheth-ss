from pydantic import BaseModel, validator


class EmbedRequest(BaseModel):
    queries: str | list[str]

    @validator('queries')
    def validate_queries(cls, queries: str | list[str]) -> list[str]:
        if isinstance(queries, str):
            queries = [queries]

        if len(queries) > 50 or max([len(query.split()) for query in queries]) > 25:
            raise ValueError('Too many queries or too long query.')

        return list(queries)
