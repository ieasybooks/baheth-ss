import argparse
import json

from pathlib import Path
from typing import Any

from args import Args
from utils.dorar import prepare_data as prepare_dorar_data
from utils.lk import prepare_data as prepare_lk_data


def main() -> None:
    """
    poetry run python tools/hadith_data_builder/main.py \
        --lk_data_path data/LK-Hadith-Corpus/ \
        --dorar_data_path data/dorar/
    """

    args = parse_arguments()

    books_data = prepare_lk_data(args.lk_data_path) + prepare_dorar_data(args.dorar_data_path)

    show_statistics(books_data)

    write_data(books_data, args.output_dir)


def parse_arguments() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument('--lk_data_path', type=Path, required=True)
    parser.add_argument('--dorar_data_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default='./data')

    return parser.parse_args()


def show_statistics(books_data: list[dict[str, Any]]) -> None:
    dorar_hashes = set()

    for book_data in books_data:
        if book_data['source'] == 'dorar':
            for hadith in book_data['hadiths']:
                dorar_hashes.add(hadith['data']['hash'])

    missing_similar_relations = 0
    missing_alt_relations = 0

    for book_data in books_data:
        if book_data['source'] == 'dorar':
            for hadith in book_data['hadiths']:
                for hash in hadith['data']['similars']:
                    if hash not in dorar_hashes:
                        missing_similar_relations += 1

                for hash in hadith['data']['alts']:
                    if hash not in dorar_hashes:
                        missing_alt_relations += 1

    print(f'Books count: {len(books_data)}')
    print(f"Hadiths count: {sum([len(book_data['hadiths']) for book_data in books_data])}")

    print('LK:')
    print(
        f"- Hadiths count: {sum([len(book_data['hadiths']) for book_data in books_data if book_data['source'] == 'lk'])}"
    )

    print('Dorar:')
    print(
        f"- Hadiths count: {sum([len(book_data['hadiths']) for book_data in books_data if book_data['source'] == 'dorar'])}"
    )
    print(f'- Missing similar relations: {missing_similar_relations}')
    print(f'- Missing alt relations: {missing_alt_relations}')


def write_data(books_data: list[dict[str, Any]], output_dir: Path) -> None:
    (output_dir / 'hadith_books_data').mkdir(exist_ok=True)

    for book_data in books_data:
        (output_dir / 'hadith_books_data' / f"{book_data['title']}.json").open('w').write(
            json.dumps(book_data, ensure_ascii=False, indent=2)
        )


if __name__ == '__main__':
    main()
