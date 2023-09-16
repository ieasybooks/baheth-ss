from typing import Annotated

import torch

from fastapi import APIRouter, HTTPException, Query, Request, status

from ..requests.hadiths_semantic_search_request import HadithsSemanticSearchRequest
from ..responses.hadiths_count_response import HadithsCountResponse
from ..responses.hadiths_nearest_neighbors_response import HadithsNearestNeighborsResponse
from ..responses.hadiths_semantic_search_response import HadithsSemanticSearchResponse


router = APIRouter(prefix='/hadiths', tags=['hadiths'])


@router.get('/count', response_model=HadithsCountResponse)
def count(request: Request) -> HadithsCountResponse:
    return HadithsCountResponse(hadiths_count=len(request.app.hadiths.data['indexes']))


@router.get('/{hadith_index}/nearest_neighbors', response_model=HadithsNearestNeighborsResponse)
def nearest_neighbors(
    request: Request,
    hadith_index: int,
    limit: Annotated[int, Query(ge=0, le=100)] = 50,
) -> HadithsNearestNeighborsResponse:
    if hadith_index > len(request.app.hadiths.data['indexes']):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Hadith does not exist.')

    return HadithsNearestNeighborsResponse(
        hadith_index=hadith_index,
        limit=limit,
        nearest_neighbors=request.app.hadiths.data['nearest_neighbors'][hadith_index][:limit],
    )


@router.post('/semantic_search', response_model=HadithsSemanticSearchResponse)
def semantic_search(
    request: Request,
    hadiths_semantic_search_request: HadithsSemanticSearchRequest,
) -> HadithsSemanticSearchResponse:
    queries = [f'query: {query}' for query in hadiths_semantic_search_request.queries]

    queries_embeddings = torch.stack((request.app.hadiths.embedder(queries))).squeeze(1)

    topk_queries_hadiths = (
        ((queries_embeddings @ request.app.hadiths.data['embeddings']) * 100)
        .topk(hadiths_semantic_search_request.limit)
        .indices.tolist()
    )

    return HadithsSemanticSearchResponse(
        limit=hadiths_semantic_search_request.limit,
        results=[
            {
                'query': query,
                'matching_hadiths': [
                    request.app.hadiths.data['indexes'][topk_query_hadith] for topk_query_hadith in topk_query_hadiths
                ],
            }
            for query, topk_query_hadiths in zip(hadiths_semantic_search_request.queries, topk_queries_hadiths)
        ],
    )
