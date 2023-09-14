from collections import UserDict
from typing import Any

import torch.nn.functional as F

from torch import Tensor
from transformers import BatchEncoding, Pipeline


MODEL_MAX_LENGTH = 512


class SentenceEmbeddingPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return {}, {}, {}

    def preprocess(self, inputs: str) -> BatchEncoding:
        return self.tokenizer(inputs, max_length=MODEL_MAX_LENGTH, padding=True, truncation=True, return_tensors='pt')

    def _forward(self, model_inputs: UserDict[str, Tensor]) -> dict[str, Any]:
        print(model_inputs)
        print(type(model_inputs))
        return {
            'outputs': self.model(**model_inputs)['last_hidden_state'],
            'attention_mask': model_inputs['attention_mask'],
        }

    def postprocess(self, model_outputs: dict[str, Any]) -> Tensor:
        return F.normalize(average_pool(model_outputs['outputs'], model_outputs['attention_mask']), p=2, dim=1)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
