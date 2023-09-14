import pickle as pkl

from pathlib import Path

import gdown

from torch import Tensor


def download_hadiths_data_file_if_not_exists(hadiths_data_file_path: Path, hadiths_data_file_url: str) -> None:
    if hadiths_data_file_path.exists():
        return

    gdown.download(hadiths_data_file_url, output=str(hadiths_data_file_path))


def load_hadiths_data(hadiths_data_file_path: Path) -> tuple[list[int], Tensor]:
    with open(hadiths_data_file_path, 'rb') as fp:
        hadiths_data = pkl.load(fp)

    return hadiths_data['indexes'], hadiths_data['embeddings'].T
