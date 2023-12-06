from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    use_onnx_runtime: bool

    hf_access_token: str
    hf_model_id: str
