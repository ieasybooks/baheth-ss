from pydantic import BaseModel, validator


class EmbedRequest(BaseModel):
    texts: str | list[str]
    type: str

    @validator('texts')
    def validate_texts(cls, texts: str | list[str]) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]

        return list(texts)
