from dataclasses import dataclass
from pathlib import Path


@dataclass
class Args:
    lk_data_path: Path
    dorar_data_path: Path
    hf_access_token: str
    hf_model_id: str
    hf_embeddings_dataset_id: str
    embedding_batch_size: int
    embedding_buffer_size: int
    max_nearest_neighbors: int
    use_onnx_runtime: bool
    output_dir: Path
