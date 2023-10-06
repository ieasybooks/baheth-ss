import argparse
import pickle as pkl
import shutil
import unicodedata

from pathlib import Path
from typing import Any

import datasets
import huggingface_hub
import pandas as pd
import torch
import torch.nn.functional as F

from optimum.onnxruntime import ORTModelForFeatureExtraction
from pyarabic.araby import strip_tashkeel
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel, XLMRobertaTokenizerFast


BATCH_SIZE = 2

LB_BOOK_ID_TO_NAME = {
    'AbuDaud': {
        'ar': 'سنن أبي داود',
        'en': 'Sunan Abi Dawud',
    },
    'Bukhari': {
        'ar': 'صحيح البخاري',
        'en': 'Sahih al-Bukhari',
    },
    'IbnMaja': {
        'ar': 'سنن ابن ماجه',
        'en': 'Sunan Ibn Majah',
    },
    'Muslim': {
        'ar': 'صحيح مسلم',
        'en': 'Sahih Muslim',
    },
    'Nesai': {
        'ar': 'سنن النسائي',
        'en': "Sunan an-Nasa'i",
    },
    'Tirmizi': {
        'ar': 'جامع الترمذي',
        'en': 'Jami` at-Tirmidhi',
    },
}


def main() -> None:
    """
    poetry run python tools/lk_data_builder.py \
        --lk_data_path data/LK-Hadith-Corpus/ \
        --hf_access_token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
        --hf_model_id ieasybooks/multilingual-e5-large-onnx \
        --hf_processed_dataset_id ieasybooks/lk-hadith-corpus-processed \
        --hf_embeddings_dataset_id ieasybooks/lk-hadith-corpus-e5-embeddings-onnx \
        --use_onnx_runtime
    """

    args = parse_arguments()

    huggingface_hub.login(token=args.hf_access_token)

    lk_data = read_lk_data(args.lk_data_path)

    tokenizer, model = load_tokenizer_and_model(args.hf_model_id, args.use_onnx_runtime)
    lk_embeddings = embed_lk_data(lk_data, tokenizer, model, args.output_dir)
    lk_nearest_neighbors = get_lk_nearest_neighbors(lk_embeddings)

    lk_data['nearest_neighbors'] = lk_nearest_neighbors

    datasets.Dataset.from_pandas(lk_data, preserve_index=False).push_to_hub(args.hf_processed_dataset_id, private=True)

    datasets.Dataset.from_dict(
        {
            'indexes': lk_data['index'].tolist(),
            'embeddings': lk_embeddings,
        },
    ).push_to_hub(args.hf_embeddings_dataset_id, private=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--lk_data_path', type=Path, required=True)
    parser.add_argument('--hf_access_token')
    parser.add_argument('--hf_model_id')
    parser.add_argument('--hf_processed_dataset_id')
    parser.add_argument('--hf_embeddings_dataset_id')
    parser.add_argument('--use_onnx_runtime', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--output_dir', type=Path, default='./data')
    parser.add_argument('--output_file_name', default='lk_hadiths_data')

    return parser.parse_args()


def read_lk_data(lk_data_path: Path) -> pd.DataFrame:
    lk_books_data = []

    for lk_book_id in LB_BOOK_ID_TO_NAME.keys():
        lk_book_data = read_lk_book_data(lk_data_path, lk_book_id)
        lk_book_data = process_lk_book_data(lk_book_data, lk_book_id)

        lk_books_data.append(lk_book_data)

    lk_data = pd.concat(lk_books_data)

    lk_data.insert(loc=0, column='index', value=list(range(len(lk_data))))
    lk_data['index'] = lk_data['index'] + 1

    return lk_data


def read_lk_book_data(lk_data_path: Path, lk_book_id: str) -> pd.DataFrame:
    chapter_paths = sorted(
        lk_data_path.joinpath(lk_book_id).glob('*.csv'),
        key=lambda path: int(Path(path).stem.replace('Chapter', '')),
    )

    return pd.concat([pd.read_csv(chapter_path) for chapter_path in chapter_paths])


def process_lk_book_data(lk_book_data: pd.DataFrame, lk_book_id: str) -> pd.DataFrame:
    lk_book_data.rename(
        columns={
            'Chapter_Number': 'chapter_number',
            'Chapter_English': 'english_chapter',
            'Chapter_Arabic': 'arabic_chapter',
            'Section_Number': 'section_number',
            'Section_English': 'english_section',
            'Section_Arabic': 'arabic_section',
            'Hadith_number': 'hadith_number',
            'English_Hadith': 'english_hadith',
            'English_Isnad': 'english_isnad',
            'English_Matn': 'english_matn',
            'Arabic_Hadith': 'arabic_hadith',
            'Arabic_Isnad': 'arabic_isnad',
            'Arabic_Matn': 'arabic_matn',
            'Arabic_Comment': 'arabic_comment',
            'English_Grade': 'english_grade',
            'Arabic_Grade': 'arabic_grade',
        },
        inplace=True,
    )

    lk_book_data.dropna(subset=['arabic_hadith'], inplace=True)

    lk_book_data['text_to_embed'] = lk_book_data.apply(build_text_to_embed, axis=1)

    lk_book_data = lk_book_data.map(remove_control_characters, na_action='ignore')
    lk_book_data = lk_book_data.map(strip_str, na_action='ignore')

    lk_book_data['arabic_book_name'] = LB_BOOK_ID_TO_NAME[lk_book_id]['ar']
    lk_book_data['english_book_name'] = LB_BOOK_ID_TO_NAME[lk_book_id]['en']

    lk_book_data['chapter_number'] = lk_book_data['chapter_number'].map(to_int_if_float).astype(str)
    lk_book_data['section_number'] = lk_book_data['section_number'].map(to_int_if_float).astype(str)
    lk_book_data['hadith_number'] = lk_book_data['hadith_number'].map(to_int_if_float).astype(str)

    lk_book_data = lk_book_data[
        [
            'arabic_book_name',
            'english_book_name',
            'chapter_number',
            'arabic_chapter',
            'english_chapter',
            'section_number',
            'arabic_section',
            'english_section',
            'hadith_number',
            'arabic_hadith',
            'arabic_isnad',
            'arabic_matn',
            'english_hadith',
            'english_isnad',
            'english_matn',
            'arabic_grade',
            'english_grade',
            'arabic_comment',
            'text_to_embed',
        ]
    ]

    return lk_book_data


def build_text_to_embed(row: pd.Series) -> str:
    fields = ['arabic_chapter', 'arabic_section', 'arabic_matn', 'arabic_grade']
    text_to_embed = []

    for field in fields:
        if field == 'arabic_matn' and row[field] != row[field]:
            text_to_embed.append(f"{strip_tashkeel(row['arabic_hadith'])} - ")
        elif row[field] == row[field]:
            text_to_embed.append(f"{strip_tashkeel(row[field])} - ")

    return ' '.join(text_to_embed)[:-3]


def remove_control_characters(value: Any) -> Any:
    if type(value) == str:
        return ''.join(ch for ch in value if unicodedata.category(ch)[0] != 'C')

    return value


def strip_str(value: Any) -> Any:
    if type(value) == str:
        return ' '.join(value.strip().split())

    return value


def to_int_if_float(value: Any) -> Any:
    if value == value and type(value) == float:
        return int(value)

    return value


def load_tokenizer_and_model(
    hf_model_id: str,
    use_onnx_runtime: bool,
) -> tuple[XLMRobertaTokenizerFast, ORTModelForFeatureExtraction | XLMRobertaModel]:
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    if use_onnx_runtime:
        model = ORTModelForFeatureExtraction.from_pretrained(hf_model_id)
    else:
        model = AutoModel.from_pretrained(hf_model_id)

    return tokenizer, model


def embed_lk_data(
    lk_data: pd.DataFrame,
    tokenizer: XLMRobertaTokenizerFast,
    model: XLMRobertaModel,
    output_dir: Path,
) -> Tensor:
    tmp_embeddings_dir = output_dir.joinpath('tmp_embeddings')
    tmp_embeddings_dir.mkdir()

    indexes = lk_data['index'].tolist()
    texts = lk_data['text_to_embed'].tolist()

    indexes_buffer = []
    embeddings_buffer = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        indexes_buffer.extend(indexes[i : i + BATCH_SIZE])

        texts_batch = list(map(lambda text: f'passage: {text}', texts[i : i + BATCH_SIZE]))

        inputs = tokenizer(texts_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**inputs)

        embeddings_buffer.extend(average_pool(outputs.last_hidden_state, inputs['attention_mask']))

        if len(embeddings_buffer) == 50 or i + BATCH_SIZE == len(texts):
            embeddings_buffer = torch.stack((embeddings_buffer))
            embeddings_buffer = F.normalize(embeddings_buffer, p=2, dim=1)

            with open(tmp_embeddings_dir.joinpath(f'{i + BATCH_SIZE}.pkl'), 'wb') as fp:
                pkl.dump({'indexes': indexes_buffer, 'embeddings': embeddings_buffer}, fp)

            indexes_buffer = []
            embeddings_buffer = []

    for tmp_embeddings_file in tmp_embeddings_dir.glob('*.pkl'):
        with open(tmp_embeddings_file, 'rb') as fp:
            embeddings_file_data = pkl.load(fp)

            indexes_buffer.extend(embeddings_file_data['indexes'])
            embeddings_buffer.extend(embeddings_file_data['embeddings'])

    shutil.rmtree(tmp_embeddings_dir)

    assert len(embeddings_buffer) == len(texts)

    return torch.stack((embeddings_buffer))


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_lk_nearest_neighbors(lk_embeddings: Tensor) -> Tensor:
    lk_nearest_neighbors = (lk_embeddings @ lk_embeddings.T * 100).topk(101).indices.tolist()

    for i in range(len(lk_nearest_neighbors)):
        lk_nearest_neighbors[i].remove(i)

    return lk_nearest_neighbors


if __name__ == '__main__':
    main()
