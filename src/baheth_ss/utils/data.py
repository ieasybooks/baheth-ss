import pickle as pkl

from pathlib import Path

import gdown

from ..types.hadiths_data import HadithsData


def download_hadiths_data_file_if_not_exists(hadiths_data_file_path: Path, hadiths_data_file_url: str) -> None:
    if hadiths_data_file_path.exists():
        return

    gdown.download(hadiths_data_file_url, output=str(hadiths_data_file_path), fuzzy=True)


def load_hadiths_data(hadiths_data_file_path: Path) -> HadithsData:
    with open(hadiths_data_file_path, 'rb') as fp:
        hadiths_data: HadithsData = pkl.load(fp)

    hadiths_data['embeddings'] = hadiths_data['embeddings'].T

    return hadiths_data
