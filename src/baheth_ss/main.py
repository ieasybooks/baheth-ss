from contextlib import asynccontextmanager
from typing import AsyncGenerator

import huggingface_hub

from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoModel, AutoTokenizer

from .pipelines.sentence_embedding_pipeline import SentenceEmbeddingPipeline
from .routers import embed
from .settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    huggingface_hub.login(token=app.settings.hf_access_token)

    model_class = AutoModel
    if app.settings.use_onnx_runtime:
        model_class = ORTModelForFeatureExtraction

    app.embedder = SentenceEmbeddingPipeline(
        model=model_class.from_pretrained(app.settings.hf_model_id),
        tokenizer=AutoTokenizer.from_pretrained(app.settings.hf_model_id),
    )

    yield

    del app.embedder


app = FastAPI(lifespan=lifespan)
app.settings = Settings()
app.include_router(embed.router)


@app.get('/')
def root() -> str:
    return 'خدمة البحث بالمعنى على منصة باحث'


@app.get('/are_you_healthy')
def are_you_healthy() -> str:
    return 'أنا بخير، شكرا لسؤالك :)'
