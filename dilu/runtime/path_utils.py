import os
from datetime import datetime


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent_dir(file_path: str) -> str:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return file_path


def timestamped_results_path(prefix: str, ext: str = ".json", results_dir: str = "results") -> str:
    ensure_dir(results_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(results_dir, f"{prefix}_{ts}{ext}")
