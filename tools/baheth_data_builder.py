import argparse

from pathlib import Path
from typing import Any

import datasets
import huggingface_hub

from datasets import DatasetDict, load_dataset


def main() -> None:
    """
    poetry run python tools/baheth_data_builder.py \
        --hf_access_token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
        --hf_lk_processed_dataset_id ieasybooks/lk-hadith-corpus-processed
    """

    args = parse_arguments()

    huggingface_hub.login(token=args.hf_access_token)

    lk_data = load_and_process_lk_data(args.hf_lk_processed_dataset_id)

    baheth_data = datasets.concatenate_datasets([lk_data])
    baheth_data.to_pandas().to_json(
        args.output_dir.joinpath(f'{args.output_file_name}.json'),
        indent=2,
        force_ascii=False,
        orient='records',
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--hf_access_token')
    parser.add_argument('--hf_lk_processed_dataset_id')
    parser.add_argument('--output_dir', type=Path, default='./data')
    parser.add_argument('--output_file_name', default='baheth_data')

    return parser.parse_args()


def load_and_process_lk_data(hf_lk_processed_dataset_id: str) -> DatasetDict:
    return load_dataset(hf_lk_processed_dataset_id, download_mode='force_redownload', split='train').map(
        transform_lk_raw,
        remove_columns=[
            'index',
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
            'nearest_neighbors',
        ],
    )


def transform_lk_raw(row: dict[str, Any]) -> dict[str, Any]:
    return {
        'source': 'lk',
        'index': row['index'],
        'searchable_text': row['text_to_embed'],
        'data': {
            'arabic_book_name': row['arabic_book_name'],
            'english_book_name': row['english_book_name'],
            'chapter_number': row['chapter_number'],
            'arabic_chapter': row['arabic_chapter'],
            'english_chapter': row['english_chapter'],
            'section_number': row['section_number'],
            'arabic_section': row['arabic_section'],
            'english_section': row['english_section'],
            'hadith_number': row['hadith_number'],
            'arabic_hadith': row['arabic_hadith'],
            'arabic_isnad': row['arabic_isnad'],
            'arabic_matn': row['arabic_matn'],
            'english_hadith': row['english_hadith'],
            'english_isnad': row['english_isnad'],
            'english_matn': row['english_matn'],
            'arabic_grade': row['arabic_grade'],
            'english_grade': row['english_grade'],
            'arabic_comment': str(row['arabic_comment']),
        },
        'hadith_book_name': row['arabic_book_name'],
        'hadith_neighbors': row['nearest_neighbors'],
    }


if __name__ == '__main__':
    main()
