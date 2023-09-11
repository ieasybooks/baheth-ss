import os

import huggingface_hub

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from optimum.pipelines import pipeline
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hf_access_token: str
    hf_model_id: str


settings = Settings()

huggingface_hub.login(token=settings.hf_access_token)

app = FastAPI()
embedder = pipeline(task='feature-extraction', model=settings.hf_model_id, accelerator='ort')


@app.post('/hadiths/semantic_search')
def hadiths_semantic_search(query: str) -> JSONResponse:
    if embedder is None:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=[])

    return JSONResponse(status_code=status.HTTP_200_OK, content=[])


@app.post('/hadiths/count')
def hadiths_count() -> int:
    return 0


@app.get('/up')
def up() -> str:
    return 'أنا بخير، شكرا لسؤالك :)'
