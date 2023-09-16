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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    huggingface_hub.login(token=app.settings.hf_access_token)

    data_utils.download_hadiths_data_file_if_not_exists(
        app.settings.hadiths_data_file_path,
        app.settings.hadiths_data_file_url,
    )

    model_class = AutoModel
    if app.settings.use_onnx_runtime:
        model_class = ORTModelForFeatureExtraction

    app.hadiths = data_utils.load_hadiths_data(app.settings.hadiths_data_file_path)
    app.hadiths.embedder = SentenceEmbeddingPipeline(
        model=model_class.from_pretrained(app.settings.hf_model_id),
        tokenizer=AutoTokenizer.from_pretrained(app.settings.hf_model_id),
    )

    yield

    del app.hadiths.data
    del app.hadiths.embedder


app = FastAPI(lifespan=lifespan)
app.settings = Settings()
app.include_router(hadiths.router)


@app.get('/')
def root() -> str:
    return 'خدمة البحث بالمعنى على منصة باحث'


@app.get('/are_you_healthy')
def are_you_healthy() -> str:
    return 'أنا بخير، شكرا لسؤالك :)'
