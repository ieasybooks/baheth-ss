import pickle as pkl

from pathlib import Path

import gdown
import huggingface_hub
import torch.nn.functional as F

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings
from torch import Tensor


class Settings(BaseSettings):
    use_onnx_runtime: bool
    hf_access_token: str
    hf_model_id: str
    hadiths_data_file_path: Path
    hadiths_data_file_url: str


settings = Settings()


if settings.use_onnx_runtime:
    from optimum.pipelines import pipeline

    embedder = pipeline(task='feature-extraction', model=settings.hf_model_id, accelerator='ort')
else:
    from transformers import pipeline

    embedder = pipeline(task='feature-extraction', model=settings.hf_model_id)


def download_hadiths_data_file_if_not_exists(hadiths_data_file_path: Path, hadiths_data_file_url: str) -> None:
    if hadiths_data_file_path.exists():
        return

    gdown.download(hadiths_data_file_url, output=str(hadiths_data_file_path))


def load_hadiths_data(hadiths_data_file_path: Path) -> tuple[list[int], Tensor]:
    with open(hadiths_data_file_path, 'rb') as fp:
        hadiths_data = pkl.load(fp)

    return hadiths_data['indexes'], hadiths_data['embeddings'].T


huggingface_hub.login(token=settings.hf_access_token)

download_hadiths_data_file_if_not_exists(settings.hadiths_data_file_path, settings.hadiths_data_file_url)

app = FastAPI()
indexes, embeddings = load_hadiths_data(settings.hadiths_data_file_path)


@app.get('/hadiths/semantic_search')
def hadiths_semantic_search(query: str, limit: int = 10) -> JSONResponse:
    try:
        embedder
    except NameError:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={})

    if len(query.split()) > 25:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={})

    query_embeddings = F.normalize(Tensor(embedder(f'query: {query}')[0]).mean(dim=0), p=2, dim=0)

    topk_embeddings = ((query_embeddings @ embeddings) * 100).topk(limit).indices.tolist()

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={'matching_results': [indexes[element] for element in topk_embeddings]},
    )


@app.get('/hadiths/count')
def hadiths_count() -> JSONResponse:
    try:
        indexes
    except NameError:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={})

    return JSONResponse(status_code=status.HTTP_200_OK, content={'hadiths_count': len(indexes)})


@app.get('/up')
def up() -> str:
    return 'أنا بخير، شكرا لسؤالك :)'
