import huggingface_hub
import torch

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoModel, AutoTokenizer

import src.baheth_ss.utils.data as data_utils

from src.baheth_ss.pipelines.sentence_embedding_pipeline import SentenceEmbeddingPipeline
from src.baheth_ss.settings import Settings


settings = Settings()

model_class = AutoModel
if settings.use_onnx_runtime:
    model_class = ORTModelForFeatureExtraction

data_utils.download_hadiths_data_file_if_not_exists(settings.hadiths_data_file_path, settings.hadiths_data_file_url)

huggingface_hub.login(token=settings.hf_access_token)

app = FastAPI()
hadiths_data = data_utils.load_hadiths_data(settings.hadiths_data_file_path)
embedder = SentenceEmbeddingPipeline(
    model=model_class.from_pretrained(settings.hf_model_id),
    tokenizer=AutoTokenizer.from_pretrained(settings.hf_model_id),
)


@app.get('/')
def root() -> str:
    return 'خدمة البحث بالمعنى على منصة باحث'


@app.post('/hadiths/semantic_search')
def hadiths_semantic_search(queries: str | list[str], limit: int = 10) -> JSONResponse:
    try:
        hadiths_data
        embedder
    except NameError:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={})

    if type(queries) == str:
        queries = [queries]

    if len(queries) > 50 or max([len(query.split()) for query in queries]) > 25:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={})

    queries_embeddings = torch.stack((embedder([f'query: {query}' for query in queries]))).squeeze(1)

    topk_queries_hadiths = ((queries_embeddings @ hadiths_data['embeddings']) * 100).topk(limit).indices.tolist()

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=[
            {
                'query': query,
                'matching_hadiths': [
                    hadiths_data['indexes'][topk_query_hadith] for topk_query_hadith in topk_query_hadiths
                ],
            }
            for query, topk_query_hadiths in zip(queries, topk_queries_hadiths)
        ],
    )


@app.get('/hadiths/nearest_neighbors')
def hadiths_nearest_neighbors(hadith_index: int) -> JSONResponse:
    try:
        hadiths_data
    except NameError:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={})

    if hadith_index > len(hadiths_data['indexes']):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={})

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'hadith_index': hadith_index,
            'nearest_neighbors': hadiths_data['nearest_neighbors'][hadith_index],
        },
    )


@app.get('/hadiths/count')
def hadiths_count() -> JSONResponse:
    try:
        hadiths_data
    except NameError:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={})

    return JSONResponse(status_code=status.HTTP_200_OK, content={'hadiths_count': len(hadiths_data['indexes'])})


@app.get('/are_you_healthy')
def are_you_healthy() -> str:
    return 'أنا بخير، شكرا لسؤالك :)'
