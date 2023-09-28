import pickle as pkl

from pathlib import Path

from datasets import load_dataset
from dotmap import DotMap

from ..types.hadiths_data import HadithsData


def load_hadiths_data(hf_embeddings_dataset_id: str) -> DotMap:
    hadiths_data = load_dataset(hf_embeddings_dataset_id, split='train')
    hadiths_data.set_format('pt', columns=['embeddings'])

    return DotMap(indexes=hadiths_data['indexes'], embeddings=hadiths_data['embeddings'].T)
