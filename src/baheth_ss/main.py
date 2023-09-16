from contextlib import asynccontextmanager
from typing import AsyncGenerator

import huggingface_hub

from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoModel, AutoTokenizer

import src.baheth_ss.utils.data as data_utils

from .pipelines.sentence_embedding_pipeline import SentenceEmbeddingPipeline
from .routers import hadiths
from .settings import Settings


settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    huggingface_hub.login(token=settings.hf_access_token)

    data_utils.download_hadiths_data_file_if_not_exists(settings.hadiths_data_file_path, settings.hadiths_data_file_url)

    model_class = AutoModel
    if settings.use_onnx_runtime:
        model_class = ORTModelForFeatureExtraction

    app.hadiths.data = data_utils.load_hadiths_data(settings.hadiths_data_file_path)
    app.hadiths.embedder = SentenceEmbeddingPipeline(
        model=model_class.from_pretrained(settings.hf_model_id),
        tokenizer=AutoTokenizer.from_pretrained(settings.hf_model_id),
    )

    yield

    del app.hadiths.data
    del app.hadiths.embedder


app = FastAPI(lifespan=lifespan)
app.include_router(hadiths.router)


@app.get('/')
def root() -> str:
    return 'خدمة البحث بالمعنى على منصة باحث'


@app.get('/are_you_healthy')
def are_you_healthy() -> str:
    return 'أنا بخير، شكرا لسؤالك :)'
