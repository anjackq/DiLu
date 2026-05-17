from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_ID = "lampilot_highway_port_v1"
BENCHMARK_VARIANT = "port_policy_exec"
EXECUTION_MODE = "programmatic_policy_exec"
EXCLUDED_CONFIG_FILENAMES = ("go_straight.json", "turn_left.json", "turn_right.json")
DEMO_ITEM_IDS = (
    "dec_abs_dis25:s1:c0",
    "dec_abs_speed10:s0:c0",
    "dec_rel_dis15:s1:c0",
    "dec_rel_speed6:s0:c0",
    "left_lc:s0:c0",
    "right_lc:s0:c0",
    "left_overtake:s0:c0",
    "right_overtake:s0:c0",
    "pull_over:s0:c5",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _benchmark_root(identifier: str = BENCHMARK_ID) -> Path:
    root = _repo_root() / "benchmarks" / str(identifier).strip()
    if not root.exists():
        raise FileNotFoundError(f"Benchmark root not found: {root}")
    return root


def load_source_manifest(identifier: str = BENCHMARK_ID) -> dict[str, Any]:
    manifest_path = _benchmark_root(identifier) / "source_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("expected_config_count", 0)) <= 0:
        raise ValueError(f"{manifest_path} must declare expected_config_count")
    if int(payload.get("expected_item_count", 0)) <= 0:
        raise ValueError(f"{manifest_path} must declare expected_item_count")
    return payload


def _load_config_names(identifier: str = BENCHMARK_ID) -> list[str]:
    config_list_path = _benchmark_root(identifier) / "config_list.txt"
    names = [line.strip() for line in config_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"{config_list_path} is empty")
    return names


def _load_config_payload(identifier: str, config_name: str) -> dict[str, Any]:
    payload_path = _benchmark_root(identifier) / "configs" / config_name
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    commands = payload.get("commands")
    samples = payload.get("samples")
    if not isinstance(commands, list) or not commands:
        raise ValueError(f"{payload_path} must define non-empty commands")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{payload_path} must define non-empty samples")
    return payload


def _task_family_from_eval_type(eval_type: str) -> str:
    normalized = str(eval_type or "").strip().lower()
    mapping = {
        "accevalbyspeed": "acc_speed",
        "accevalbydistance": "acc_distance",
        "lanechangeeval": "lane_change",
        "overtakeeval": "overtake",
        "pullovereval": "pull_over",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported evaluator type: {eval_type!r}")
    return mapping[normalized]


def _scenario_family_from_env_type(env_type: str) -> str:
    normalized = str(env_type or "").strip().lower()
    if normalized in {"ramp-merge-v0", "dt-highway-v0"}:
        return "highway"
    raise ValueError(f"Unsupported env type in highway port: {env_type!r}")


def build_dataset(
    identifier: str = BENCHMARK_ID,
    *,
    use_demo: bool = False,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    manifest = load_source_manifest(identifier)
    items: list[dict[str, Any]] = []
    for config_name in _load_config_names(identifier):
        payload = _load_config_payload(identifier, config_name)
        config_stem = Path(config_name).stem
        commands = payload["commands"]
        samples = payload["samples"]
        for sample_index, sample in enumerate(samples):
            env_type = str(sample.get("env", {}).get("type") or "").strip()
            eval_type = str(sample.get("eval", {}).get("type") or "").strip()
            task_family = _task_family_from_eval_type(eval_type)
            scenario_family = _scenario_family_from_env_type(env_type)
            for command_index, command in enumerate(commands):
                item_id = f"{config_stem}:s{sample_index}:c{command_index}"
                items.append(
                    {
                        "benchmark_name": identifier,
                        "benchmark_variant": BENCHMARK_VARIANT,
                        "execution_mode": EXECUTION_MODE,
                        "upstream_commit": manifest["upstream_commit"],
                        "config_name": config_name,
                        "config_stem": config_stem,
                        "item_id": item_id,
                        "command": str(command),
                        "sample": sample,
                        "env_type": env_type,
                        "evaluator_type": eval_type,
                        "task_family": task_family,
                        "scenario_family": scenario_family,
                    }
                )
    if use_demo:
        demo_ids = set(DEMO_ITEM_IDS)
        items = [item for item in items if item["item_id"] in demo_ids]
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(items)

    expected = int(manifest["expected_item_count"])
    if not use_demo and len(items) != expected:
        raise ValueError(f"{identifier} item count mismatch: expected {expected}, got {len(items)}")
    return items


def load_dataset(
    identifier: str = BENCHMARK_ID,
    *,
    use_demo: bool = False,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    return build_dataset(identifier, use_demo=use_demo, shuffle=shuffle, seed=seed)


def build_benchmark_fingerprint(identifier: str = BENCHMARK_ID) -> str:
    manifest = load_source_manifest(identifier)
    payload = {
        "benchmark_name": manifest["benchmark_name"],
        "upstream_commit": manifest["upstream_commit"],
        "included_config_filenames": list(manifest["included_config_filenames"]),
        "excluded_config_filenames": list(manifest["excluded_config_filenames"]),
        "expected_config_count": int(manifest["expected_config_count"]),
        "expected_item_count": int(manifest["expected_item_count"]),
        "benchmark_variant": manifest["benchmark_variant"],
        "execution_mode": manifest["execution_mode"],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{identifier}:{digest}"


def summarize_dataset(identifier: str = BENCHMARK_ID) -> dict[str, Any]:
    items = load_dataset(identifier)
    return {
        "benchmark_name": identifier,
        "item_count": len(items),
        "config_names": sorted({item["config_name"] for item in items}),
        "task_families": sorted({item["task_family"] for item in items}),
        "env_types": sorted({item["env_type"] for item in items}),
    }


def iter_demo_item_ids() -> Iterable[str]:
    return tuple(DEMO_ITEM_IDS)
