from pathlib import Path
from typing import Any

import pandas as pd

from constants.lk import BOOK_DATA
from pyarabic.araby import strip_tashkeel
from utils.common import remove_control_characters, strip_str, to_int_if_float_or_str


def prepare_data(data_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = read_data(data_path)
    data = process_data(data)

    book_infos = [
        {'title': value['ar'], 'author': value['author'], 'description': '', 'source': 'lk', 'user_id': 1}
        for value in BOOK_DATA.values()
    ]

    return book_infos, data


def read_data(data_path: Path) -> pd.DataFrame:
    books_data = []

    for book_id in BOOK_DATA.keys():
        book_data = read_book_data(data_path, book_id)
        book_data = process_book_data(book_data, book_id)

        books_data.append(book_data)

    data = pd.concat(books_data)

    return data


def read_book_data(data_path: Path, book_id: str) -> pd.DataFrame:
    chapter_paths = sorted(
        data_path.joinpath(book_id).glob('*.csv'),
        key=lambda path: int(Path(path).stem.replace('Chapter', '')),
    )

    return pd.concat([pd.read_csv(chapter_path) for chapter_path in chapter_paths])


def process_book_data(book_data: pd.DataFrame, book_id: str) -> pd.DataFrame:
    book_data.rename(
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

    book_data.dropna(subset=['arabic_hadith'], inplace=True)

    book_data['arabic_comment'] = book_data['arabic_comment'].fillna('')
    book_data['text_to_embed'] = book_data.apply(build_text_to_embed, axis=1)

    book_data = book_data.map(remove_control_characters, na_action='ignore')
    book_data = book_data.map(strip_str, na_action='ignore')
    book_data = book_data.map(to_int_if_float_or_str, na_action='ignore')

    book_data['arabic_book_name'] = BOOK_DATA[book_id]['ar']
    book_data['english_book_name'] = BOOK_DATA[book_id]['en']

    book_data = book_data[
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

    return book_data


def build_text_to_embed(row: pd.Series) -> str:
    fields = ['arabic_chapter', 'arabic_section', 'arabic_matn', 'arabic_grade']
    text_to_embed = []

    for field in fields:
        if field == 'arabic_matn' and row[field] != row[field]:
            text_to_embed.append(f"{strip_tashkeel(row['arabic_hadith'])} - ")
        elif row[field] == row[field]:
            text_to_embed.append(f"{strip_tashkeel(row[field])} - ")

    return ' '.join(text_to_embed)[:-3]


def process_data(data: pd.DataFrame) -> list[dict[str, Any]]:
    processed_data: list[dict[str, Any]] = data.apply(
        lambda row: {
            'source': 'lk',
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
                'arabic_comment': row['arabic_comment'],
            },
            'hadith_book_name': row['arabic_book_name'],
        },
        axis=1,
    ).to_list()

    return processed_data
