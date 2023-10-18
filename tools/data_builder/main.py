import argparse
import json
import pickle as pkl

from pathlib import Path
from typing import Any

import datasets
import huggingface_hub

from args import Args
from torch import Tensor
from utils.dorar import prepare_data as prepare_dorar_data
from utils.lk import prepare_data as prepare_lk_data
from utils.model import embed_texts, get_nearest_neighbors, load_tokenizer_and_model


def main() -> None:
    """
    poetry run python tools/data_builder/main.py \
        --lk_data_path data/LK-Hadith-Corpus/ \
        --dorar_data_path data/dorar/ \
        --hf_access_token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
        --hf_model_id ieasybooks/multilingual-e5-large-onnx \
        --hf_embeddings_dataset_id ieasybooks/hadith-corpus-e5-embeddings-onnx \
        --embedding_batch_size 2 \
        --embedding_buffer_size 50 \
        --max_nearest_neighbors 100 \
        --use_onnx_runtime
    """

    args = parse_arguments()

    huggingface_hub.login(token=args.hf_access_token)

    lk_book_infos, lk_hadiths = prepare_lk_data(args.lk_data_path)
    dorar_book_infos, dorar_hadiths = prepare_dorar_data(args.dorar_data_path)

    book_infos = lk_book_infos + dorar_book_infos
    hadiths = lk_hadiths + dorar_hadiths

    tokenizer, model = load_tokenizer_and_model(args.hf_model_id, args.use_cuda, args.use_onnx_runtime)
    embeddings = embed_texts(
        [hadith['searchable_text'] for hadith in hadiths],
        tokenizer,
        model,
        args.embedding_batch_size,
        args.embedding_buffer_size,
        args.use_cuda,
        args.output_dir,
    )
    nearest_neighbors = get_nearest_neighbors(embeddings, args.max_nearest_neighbors)

    hadiths = post_process_hadiths(hadiths, nearest_neighbors)

    write_data(book_infos, hadiths, embeddings, args.output_dir, args.hf_embeddings_dataset_id)


def parse_arguments() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument('--lk_data_path', type=Path, required=True)
    parser.add_argument('--dorar_data_path', type=Path, required=True)
    parser.add_argument('--hf_access_token')
    parser.add_argument('--hf_model_id')
    parser.add_argument('--hf_processed_dataset_id')
    parser.add_argument('--hf_embeddings_dataset_id')
    parser.add_argument('--embedding_batch_size', type=int, default=2)
    parser.add_argument('--embedding_buffer_size', type=int, default=50)
    parser.add_argument('--max_nearest_neighbors', type=int, default=100)
    parser.add_argument('--use_cuda', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_onnx_runtime', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--output_dir', type=Path, default='./data')

    return parser.parse_args()


def post_process_hadiths(hadiths: list[dict[str, Any]], nearest_neighbors: Tensor) -> list[dict[str, Any]]:
    hash_to_index = {}

    for i in range(len(hadiths)):
        hadiths[i]['index'] = i

        if hadiths[i]['source'] == 'dorar':
            hash_to_index[hadiths[i]['data']['hash']] = i

        hadiths[i]['hadith_neighbors'] = nearest_neighbors[i]

    missing_similar_relations = 0
    missing_alt_relations = 0

    for i in range(len(hadiths)):
        if hadiths[i]['source'] == 'dorar':
            similars = []
            for h in hadiths[i]['data']['similars']:
                if h in hash_to_index:
                    similars.append(hash_to_index[h])
                else:
                    missing_similar_relations += 1
            hadiths[i]['data']['similars'] = similars

            alts = []
            for h in hadiths[i]['data']['alts']:
                if h in hash_to_index:
                    alts.append(hash_to_index[h])
                else:
                    missing_alt_relations += 1
            hadiths[i]['data']['alts'] = alts

    print(f'Missing similar relations: {missing_similar_relations}')
    print(f'Missing alt relations: {missing_alt_relations}')

    return hadiths


def write_data(
    book_infos: list[dict[str, Any]],
    hadiths: list[dict[str, Any]],
    embeddings: Tensor,
    output_dir: Path,
    hf_embeddings_dataset_id: str,
) -> None:
    (output_dir / 'hadith_books.json').open('w').write(json.dumps(book_infos, ensure_ascii=False, indent=2))
    (output_dir / 'hadiths.json').open('w').write(json.dumps(hadiths, ensure_ascii=False, indent=2))

    with open(output_dir / 'embeddings.pkl', 'wb') as fp:
        pkl.dump(embeddings, fp)

    datasets.Dataset.from_dict(
        {
            'indexes': list(range(len(hadiths))),
            'embeddings': embeddings,
        },
    ).push_to_hub(hf_embeddings_dataset_id, private=True)


if __name__ == '__main__':
    main()
