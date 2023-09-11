import argparse
import unicodedata

from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd

from pyarabic.araby import strip_tashkeel


LB_BOOK_ID_TO_ARABIC = {
    'AbuDaud': 'سنن أبي داود',
    'Bukhari': 'صحيح البخاري',
    'IbnMaja': 'سنن ابن ماجه',
    'Muslim': 'صحيح مسلم',
    'Nesai': 'سنن النسائي',
    'Tirmizi': 'سنن الترمذي',
}


def main() -> None:
    args = parse_arguments()

    lk_data = read_lk_data(args.lk_data_path)

    lk_data.to_json(args.output_dir.joinpath('lk_hadiths.json'), indent=2, force_ascii=False, orient='records')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--lk_data_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default='./data')

    return parser.parse_args()


def read_lk_data(lk_data_path: Path) -> pd.DataFrame:
    lk_books_data = []

    for lk_book_id in LB_BOOK_ID_TO_ARABIC.keys():
        lk_book_data = read_lk_book_data(lk_data_path, lk_book_id)
        lk_book_data = process_lk_book_data(lk_book_data, lk_book_id)

        lk_books_data.append(lk_book_data)

    lk_data = pd.concat(lk_books_data)

    lk_data.insert(loc=0, column='index', value=list(range(len(lk_data))))
    lk_data['index'] = lk_data['index'] + 1

    return lk_data


def read_lk_book_data(lk_data_path: Path, lk_book_id: str) -> pd.DataFrame:
    chapter_paths = sorted(
        glob(str(lk_data_path.joinpath(lk_book_id, '*.csv'))),
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

    lk_book_data = lk_book_data.map(remove_control_characters, na_action='ignore')
    lk_book_data = lk_book_data.map(strip_str, na_action='ignore')

    lk_book_data['arabic_book_name'] = LB_BOOK_ID_TO_ARABIC[lk_book_id]

    lk_book_data['chapter_number'] = lk_book_data['chapter_number'].map(to_int_if_float)
    lk_book_data['section_number'] = lk_book_data['section_number'].map(to_int_if_float)
    lk_book_data['hadith_number'] = lk_book_data['hadith_number'].map(to_int_if_float)

    lk_book_data = lk_book_data[
        [
            'arabic_book_name',
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


if __name__ == '__main__':
    main()
