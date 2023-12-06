from pydantic import BaseModel


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
