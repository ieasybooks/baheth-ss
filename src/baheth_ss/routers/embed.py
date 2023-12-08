import torch

from fastapi import APIRouter, Request

from src.baheth_ss.requests.embed_request import EmbedRequest
from src.baheth_ss.responses.embed_response import EmbedResponse


router = APIRouter(prefix='/embed', tags=['embed'])


@router.post('', response_model=EmbedResponse)
def embed(request: Request, embed_request: EmbedRequest) -> EmbedResponse:
    texts = [f'{embed_request.type}: {text}' for text in embed_request.texts]

    return EmbedResponse(embeddings=torch.stack((request.app.embedder(texts))).squeeze(1))
