import copy
import os
from typing import Any, Dict, Optional, Set

import yaml


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_runtime_config(path: str, *, _seen: Optional[Set[str]] = None) -> Dict[str, Any]:
    abs_path = os.path.abspath(path)
    seen = set(_seen or set())
    if abs_path in seen:
        raise ValueError(f"Config inheritance cycle detected at: {abs_path}")
    seen.add(abs_path)

    with open(abs_path, "r", encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=yaml.FullLoader)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {abs_path}")

    raw = dict(raw)
    base_config = raw.pop("base_config", None)
    if not base_config:
        return raw
    if not isinstance(base_config, str):
        raise ValueError(f"`base_config` must be a string path in: {abs_path}")

    if os.path.isabs(base_config):
        base_path = base_config
    else:
        base_path = os.path.join(os.path.dirname(abs_path), base_config)

    base_loaded = load_runtime_config(base_path, _seen=seen)
    return _deep_merge_dicts(base_loaded, raw)
