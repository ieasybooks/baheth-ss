from typing import TypedDict

from torch import Tensor


class HadithsData(TypedDict):
    indexes: list[int]
    embeddings: Tensor
    nearest_neighbors: list[list[int]]
