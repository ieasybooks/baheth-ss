import json

from pathlib import Path
from typing import Any


def prepare_data(data_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return read_data(data_path)


def read_data(data_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    book_infos = []
    hadiths = []

    for book_data_path in (data_path / 'books').glob('*.json'):
        book_data = read_book_data(book_data_path)
        book_info, book_hadiths = process_book_data(book_data)

        book_infos.append(book_info)
        hadiths.extend(book_hadiths)

    return book_infos, hadiths


def read_book_data(book_data_path: Path) -> dict[str, Any]:
    book_data: dict[str, Any] = json.loads(book_data_path.open().read())
    return book_data


def process_book_data(book_data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    book_info = {
        'title': book_data['name'],
        'author': book_data['author_or_supervisor'],
        'description': '',
        'source': 'dorar',
        'user_id': 1,
    }

    hadiths = []

    for hadith in book_data['hadiths']:
        hadiths.append(
            {
                'source': 'dorar',
                'searchable_text': hadith['hadith_text'],
                'data': {
                    'hash': hadith['hash'],
                    'url': hadith['url'],
                    'hadith_type': hadith['hadith_type'],
                    'rawi_id': hadith['rawi_id'],
                    'rawi_name': hadith['rawi_name'],
                    'muhaddith_id': hadith['muhaddith_id'],
                    'muhaddith_name': hadith['muhaddith_name'],
                    'source_id': hadith['source_id'],
                    'source_name': hadith['source_name'],
                    'source_page_or_number': hadith['source_page_or_number'],
                    'grade_text': hadith['grade_text'],
                    'grade_type': hadith['grade_type'],
                    'grade_clarification': hadith['grade_clarification'],
                    'takhrij': hadith['takhrij'],
                    'explanation_id': hadith['explanation_id'],
                    'explanation_text': hadith['explanation_text'],
                    'explanation_type': hadith['explanation_type'],
                    'similars': hadith['similars'],
                    'alts': hadith['alts'],
                },
                'hadith_book_name': book_data['name'],
            }
        )

    return book_info, hadiths
