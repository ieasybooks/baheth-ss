from dataclasses import dataclass
from pathlib import Path


@dataclass
class Args:
    lk_data_path: Path
    dorar_data_path: Path
    output_dir: Path
