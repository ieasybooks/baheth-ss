import pickle as pkl
import shutil

from pathlib import Path

import torch
import torch.nn.functional as F

from optimum.onnxruntime import ORTModelForFeatureExtraction
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_tokenizer_and_model(
    hf_model_id: str,
    use_cuda: bool,
    use_onnx_runtime: bool,
) -> tuple[PreTrainedTokenizer, ORTModelForFeatureExtraction | PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    if use_onnx_runtime:
        model = ORTModelForFeatureExtraction.from_pretrained(hf_model_id)
    else:
        model = AutoModel.from_pretrained(hf_model_id)

    if use_cuda:
        model = model.to(torch.device('cuda'))

    return tokenizer, model


def embed_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    batch_size: int,
    embeddings_buffer_size: int,
    use_cuda: bool,
    output_dir: Path,
) -> Tensor:
    tmp_embeddings_dir = output_dir.joinpath('tmp_embeddings')

    shutil.rmtree(tmp_embeddings_dir, ignore_errors=True)

    tmp_embeddings_dir.mkdir()

    embeddings_buffer = []

    for i in tqdm(range(0, len(texts), batch_size), desc='Extract Embeddings'):
        texts_batch = list(map(lambda text: f'passage: {text}', texts[i : i + batch_size]))

        inputs = tokenizer(texts_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')

        if use_cuda:
            inputs = {k: v.to(torch.device('cuda')) for k, v in inputs.items()}

        outputs = model(**inputs)

        if use_cuda:
            embeddings_buffer.extend(average_pool(outputs.last_hidden_state, inputs['attention_mask']).cpu())
        else:
            embeddings_buffer.extend(average_pool(outputs.last_hidden_state, inputs['attention_mask']))

        if len(embeddings_buffer) == embeddings_buffer_size or i + batch_size == len(texts):
            embeddings_buffer = torch.stack((embeddings_buffer))
            embeddings_buffer = F.normalize(embeddings_buffer, p=2, dim=1)

            with open(tmp_embeddings_dir.joinpath(f'{i + batch_size}.pkl'), 'wb') as fp:
                pkl.dump(embeddings_buffer, fp)

            embeddings_buffer = []

    for tmp_embeddings_file in tmp_embeddings_dir.glob('*.pkl'):
        with open(tmp_embeddings_file, 'rb') as fp:
            embeddings_buffer.extend(pkl.load(fp))

    shutil.rmtree(tmp_embeddings_dir)

    assert len(embeddings_buffer) == len(texts)

    return torch.stack((embeddings_buffer))


def get_nearest_neighbors(embeddings: Tensor, k: int = 100, batch_size: int = 512) -> list[list[int]]:
    nearest_neighbors: list[list[int]] = [[] for _ in range(embeddings.shape[0])]

    for batch_id in tqdm(range(0, embeddings.shape[0], batch_size), desc='Compute Nearest Neighbors2'):
        embeddings_batch = embeddings[batch_id : batch_id + batch_size]
        batch_nearest_neighbors = (embeddings_batch @ embeddings.T * 100).topk(k + 1).indices.tolist()

        for i in range(len(batch_nearest_neighbors)):
            batch_nearest_neighbors[i].remove(batch_id + i)
            nearest_neighbors[batch_id + i] = batch_nearest_neighbors[i]

    return nearest_neighbors


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
