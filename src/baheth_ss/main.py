import huggingface_hub
import torch

from fastapi import FastAPI, HTTPException, status
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoModel, AutoTokenizer

import src.baheth_ss.utils.data as data_utils

from src.baheth_ss.pipelines.sentence_embedding_pipeline import SentenceEmbeddingPipeline
from src.baheth_ss.requests.hadiths_nearest_neighbors_request import HadithsNearestNeighborsRequest
from src.baheth_ss.requests.hadiths_semantic_search_request import HadithsSemanticSearchRequest
from src.baheth_ss.responses.hadiths_count_response import HadithsCountResponse
from src.baheth_ss.responses.hadiths_nearest_neighbors_response import HadithsNearestNeighborsResponse
from src.baheth_ss.responses.hadiths_semantic_search_response import HadithsSemanticSearchResponse
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


@app.post('/hadiths/semantic_search', response_model=HadithsSemanticSearchResponse)
def hadiths_semantic_search(request: HadithsSemanticSearchRequest) -> HadithsSemanticSearchResponse:
    try:
        hadiths_data
        embedder
    except NameError:
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Service is still loading data...')

    queries_embeddings = torch.stack((embedder([f'query: {query}' for query in request.queries]))).squeeze(1)

    topk_queries_hadiths = (
        ((queries_embeddings @ hadiths_data['embeddings']) * 100).topk(request.limit).indices.tolist()
    )

    return HadithsSemanticSearchResponse(
        limit=request.limit,
        results=[
            {
                'query': query,
                'matching_hadiths': [
                    hadiths_data['indexes'][topk_query_hadith] for topk_query_hadith in topk_query_hadiths
                ],
            }
            for query, topk_query_hadiths in zip(request.queries, topk_queries_hadiths)
        ],
    )


@app.get('/hadiths/nearest_neighbors', response_model=HadithsNearestNeighborsResponse)
def hadiths_nearest_neighbors(request: HadithsNearestNeighborsRequest) -> HadithsNearestNeighborsResponse:
    try:
        hadiths_data
    except NameError:
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Service is still loading data...')

    if request.hadith_index > len(hadiths_data['indexes']):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Hadith does not exist.')

    return HadithsNearestNeighborsResponse(
        hadith_index=request.hadith_index,
        limit=request.limit,
        nearest_neighbors=hadiths_data['nearest_neighbors'][request.hadith_index][: request.limit],
    )


@app.get('/hadiths/count', response_model=HadithsCountResponse)
def hadiths_count() -> HadithsCountResponse:
    try:
        hadiths_data
    except NameError:
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Service is still loading data...')

    return HadithsCountResponse(hadiths_count=len(hadiths_data['indexes']))


@app.get('/are_you_healthy')
def are_you_healthy() -> str:
    return 'أنا بخير، شكرا لسؤالك :)'
