from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from statistics import mean
from typing import Any

from rich import print
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from dilu.runtime import (
    apply_model_policy_to_env,
    build_experiment_root,
    build_model_root,
    current_timestamp,
    ensure_dir,
    ensure_experiment_layout,
    load_runtime_config,
    read_json,
    resolve_model_policy,
    slugify_model_name,
    write_json_atomic,
)

from .benchmark import (
    BENCHMARK_ID,
    BENCHMARK_VARIANT,
    EXECUTION_MODE,
    build_benchmark_fingerprint,
    load_dataset,
    load_source_manifest,
)
from .digital_twin import ControlDigitalTwin
from .evaluators import get_evaluator_class
from .llm_codegen import HighwayCodegenAgent
from .policy import PolicyCompilationError, compile_policy_response


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_progress_enabled(
    *,
    config: dict[str, Any],
    force_progress: bool,
    disable_progress: bool,
    interactive_output: bool,
) -> bool:
    if disable_progress:
        return False
    if force_progress:
        return True
    return interactive_output and _as_bool(config.get("progress_bar", True), default=True)


def _build_cache_dir(model_root: str, run_id: str) -> str:
    return ensure_dir(os.path.join(model_root, "runs", run_id, "cache"))


def _episode_cache_path(cache_dir: str, item_id: str) -> str:
    safe_item_id = item_id.replace(":", "_")
    return os.path.join(cache_dir, f"{safe_item_id}.json")


def _build_failure_record(item: dict[str, Any], model_name: str, *, failure_reason: str, policy_status: str, policy_response: str = "", policy_source: str = "", error: str | None = None) -> dict[str, Any]:
    return {
        "model": model_name,
        "benchmark_name": item["benchmark_name"],
        "benchmark_variant": BENCHMARK_VARIANT,
        "execution_mode": EXECUTION_MODE,
        "upstream_commit": item["upstream_commit"],
        "config_name": item["config_name"],
        "item_id": item["item_id"],
        "task_family": item["task_family"],
        "env_type": item["env_type"],
        "command": item["command"],
        "task_success": False,
        "failure_reason": failure_reason,
        "policy_status": policy_status,
        "policy_error": error,
        "policy_response": policy_response,
        "policy_source": policy_source,
        "episode_execution_complete": False,
        "benchmark_result_valid": False,
        "collision": False,
        "score": 0.0,
        "score_ttc": None,
        "score_speed_variance": None,
        "score_time_efficiency": None,
        "overall_time": None,
        "speed_std": None,
    }


def _evaluate_item(
    *,
    agent: HighwayCodegenAgent,
    item: dict[str, Any],
    model_name: str,
    record_video: bool,
) -> dict[str, Any]:
    evaluator_cls = get_evaluator_class(item["evaluator_type"])
    evaluator = evaluator_cls(
        config=item["sample"],
        show_window=False,
        wait_time=0.0,
        record_video=record_video,
        video_dir="",
    )
    twin = ControlDigitalTwin()
    twin.reset(evaluator.env.unwrapped.vehicle)
    try:
        response = agent.generate(command=item["command"], context_info=evaluator.get_context_info())
        compiled = compile_policy_response(response.content)
        twin.execute(compiled)
    except PolicyCompilationError as exc:
        evaluator.close()
        return _build_failure_record(
            item,
            model_name,
            failure_reason="policy_compile_error",
            policy_status="compile_error",
            policy_response=response.content if "response" in locals() else "",
            policy_source=compiled.source_code if "compiled" in locals() else "",
            error=str(exc),
        )
    except Exception as exc:
        evaluator.close()
        return _build_failure_record(
            item,
            model_name,
            failure_reason="policy_generation_error",
            policy_status="generation_error",
            error=str(exc),
        )

    while not evaluator.ended:
        evaluator.step(twin)
        if twin.policy_failed:
            break

    result = {
        "model": model_name,
        "benchmark_name": item["benchmark_name"],
        "benchmark_variant": BENCHMARK_VARIANT,
        "execution_mode": EXECUTION_MODE,
        "upstream_commit": item["upstream_commit"],
        "config_name": item["config_name"],
        "item_id": item["item_id"],
        "task_family": item["task_family"],
        "env_type": item["env_type"],
        "command": item["command"],
        "task_success": bool(evaluator.success and not twin.policy_failed),
        "failure_reason": twin.policy_error if twin.policy_failed else ("task_not_completed" if not evaluator.success else None),
        "policy_status": twin.policy_status,
        "policy_error": twin.policy_error,
        "policy_response": response.content,
        "policy_source": compiled.source_code,
        "episode_execution_complete": bool(not twin.policy_failed),
        "benchmark_result_valid": bool(not twin.policy_failed),
        "collision": bool(evaluator.collision),
        "score": round(float(evaluator.score), 6),
        "score_ttc": round(float(evaluator.score_ttc), 6),
        "score_speed_variance": round(float(evaluator.score_speed_variance), 6),
        "score_time_efficiency": round(float(evaluator.score_time_efficiency), 6),
        "overall_time": round(float(evaluator.overall_time), 6),
        "speed_std": round(float(evaluator.speed_std), 6),
    }
    evaluator.close()
    return result


def _mean_or_none(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return round(float(mean(values)), 6)


def _build_aggregate(model_name: str, episodes: list[dict[str, Any]], *, benchmark_id: str, benchmark_fingerprint: str, upstream_commit: str) -> dict[str, Any]:
    total = len(episodes)
    task_successes = sum(1 for item in episodes if item.get("task_success"))
    collisions = sum(1 for item in episodes if item.get("collision"))
    policy_failures = [item for item in episodes if item.get("policy_status") not in {"ok", "not_started"}]
    failure_counter = Counter(item.get("failure_reason") or "unknown" for item in episodes if not item.get("task_success"))
    return {
        "model": model_name,
        "benchmark_mode": True,
        "benchmark_case_set": benchmark_id,
        "benchmark_variant": BENCHMARK_VARIANT,
        "execution_mode": EXECUTION_MODE,
        "benchmark_fingerprint": benchmark_fingerprint,
        "upstream_commit": upstream_commit,
        "headline_task_metric": "task_success_rate",
        "task_success_rate": round(task_successes / max(total, 1), 6),
        "collision_rate": round(collisions / max(total, 1), 6),
        "score_mean": _mean_or_none(episodes, "score"),
        "score_ttc_mean": _mean_or_none(episodes, "score_ttc"),
        "score_speed_variance_mean": _mean_or_none(episodes, "score_speed_variance"),
        "score_time_efficiency_mean": _mean_or_none(episodes, "score_time_efficiency"),
        "overall_time_mean": _mean_or_none(episodes, "overall_time"),
        "speed_std_mean": _mean_or_none(episodes, "speed_std"),
        "benchmark_result_valid": len(policy_failures) == 0,
        "benchmark_result_invalid_reason": None if not policy_failures else "policy_runtime_failures_present",
        "policy_failure_count": len(policy_failures),
        "episode_execution_complete": all(item.get("episode_execution_complete", False) for item in episodes),
        "failure_reasons": dict(sorted(failure_counter.items())),
    }


def _build_model_extract(model_name: str, experiment_id: str, aggregate: dict[str, Any], episodes: list[dict[str, Any]], metrics_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_schema_version": "2.0",
        "source": "evaluate_lampilot_highway:model_extract",
        "experiment_id": experiment_id,
        "model": model_name,
        "aggregate": aggregate,
        "episodes": episodes,
        "metrics_config": metrics_config,
        "benchmark_mode": True,
        "benchmark_case_set": metrics_config["benchmark_case_set"],
        "benchmark_variant": metrics_config["benchmark_variant"],
        "execution_mode": metrics_config["execution_mode"],
        "headline_task_metric": "task_success_rate",
        "efficiency_metrics_reported": False,
    }


def _update_manifest(
    *,
    experiment_root: str,
    experiment_id: str,
    config_path: str,
    benchmark_id: str,
    benchmark_fingerprint: str,
    few_shot: bool,
    model_outputs: dict[str, dict[str, str]],
    compare_report_path: str,
) -> None:
    manifest_path = os.path.join(experiment_root, "manifest.json")
    manifest = read_json(manifest_path, default={})
    now = datetime.now().isoformat(timespec="seconds")
    manifest.setdefault("experiment_id", experiment_id)
    manifest.setdefault("created_at", now)
    manifest["updated_at"] = now
    manifest["config_path"] = config_path
    manifest["benchmark_case_set"] = benchmark_id
    manifest["benchmark_variant"] = BENCHMARK_VARIANT
    manifest["execution_mode"] = EXECUTION_MODE
    manifest["benchmark_fingerprint"] = benchmark_fingerprint
    manifest["few_shot_num"] = 1 if few_shot else 0
    compare = manifest.setdefault("compare", {})
    compare["latest_report"] = compare_report_path
    compare.setdefault("history", [])
    if compare_report_path not in compare["history"]:
        compare["history"].append(compare_report_path)
    models = manifest.setdefault("models", {})
    for model_name, output_paths in model_outputs.items():
        item = models.setdefault(model_name, {})
        item["slug"] = slugify_model_name(model_name)
        item["root"] = build_model_root(experiment_root, model_name)
        item["latest_eval_summary"] = output_paths["summary_path"]
        item["latest_eval_episodes"] = output_paths["episodes_path"]
    write_json_atomic(manifest_path, manifest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_runtime_config(args.config)
    if args.progress and args.no_progress:
        raise ValueError("Use only one of --progress or --no-progress.")
    manifest = load_source_manifest(args.benchmark_id)
    dataset = load_dataset(
        args.benchmark_id,
        use_demo=bool(args.use_demo),
        shuffle=bool(args.shuffle),
        seed=int(args.random_seed),
    )
    if args.test_size is not None:
        dataset = dataset[: max(0, int(args.test_size))]
    benchmark_fingerprint = build_benchmark_fingerprint(args.benchmark_id)
    experiment_id = str(args.experiment_id or f"{args.benchmark_id}_{current_timestamp()}").strip()
    results_root = args.results_root or os.path.join("results", "lampilot_highway_port")
    experiment_root = build_experiment_root(results_root, experiment_id)
    ensure_experiment_layout(experiment_root, args.models)
    progress_enabled = _resolve_progress_enabled(
        config=config,
        force_progress=bool(args.progress),
        disable_progress=bool(args.no_progress),
        interactive_output=bool(sys.stdout.isatty()),
    )

    compare_report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_schema_version": "2.0",
        "source": "evaluate_lampilot_highway",
        "experiment_id": experiment_id,
        "benchmark_mode": True,
        "benchmark_case_set": args.benchmark_id,
        "benchmark_variant": BENCHMARK_VARIANT,
        "execution_mode": EXECUTION_MODE,
        "benchmark_fingerprint": benchmark_fingerprint,
        "upstream_commit": manifest["upstream_commit"],
        "headline_task_metric": "task_success_rate",
        "efficiency_metrics_reported": False,
        "metrics_config": {
            "benchmark_mode": True,
            "benchmark_case_set": args.benchmark_id,
            "benchmark_variant": BENCHMARK_VARIANT,
            "execution_mode": EXECUTION_MODE,
            "benchmark_fingerprint": benchmark_fingerprint,
            "upstream_commit": manifest["upstream_commit"],
            "few_shot_num": 1 if args.few_shot else 0,
        },
        "aggregates": [],
    }
    model_outputs: dict[str, dict[str, str]] = {}
    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        )
        if progress_enabled
        else nullcontext(None)
    )
    with progress_cm as progress:
        emit = progress.console.print if progress is not None else print
        model_task = progress.add_task("Models", total=len(args.models)) if progress is not None else None
        item_task = None

        for model_name in args.models:
            policy = resolve_model_policy(
                config=config,
                model_name=model_name,
                provider=str(config.get("OPENAI_API_TYPE", "")),
                mode="eval",
            )
            apply_model_policy_to_env(policy, provider=str(config.get("OPENAI_API_TYPE", "")))
            agent = HighwayCodegenAgent(
                config=config,
                model_name=model_name,
                request_timeout=float(policy["decision_timeout_sec"]),
                few_shot=bool(args.few_shot),
            )
            model_root = build_model_root(experiment_root, model_name)
            cache_dir = _build_cache_dir(model_root, f"lampilot_port_{current_timestamp()}")
            episodes: list[dict[str, Any]] = []
            if progress is not None:
                item_task = progress.add_task(f"Items | {model_name}", total=len(dataset))
            for item in dataset:
                cache_path = _episode_cache_path(cache_dir, item["item_id"])
                if args.resume and os.path.exists(cache_path):
                    episodes.append(read_json(cache_path, default={}))
                    if progress is not None and item_task is not None:
                        progress.update(item_task, advance=1)
                    continue
                episode = _evaluate_item(agent=agent, item=item, model_name=model_name, record_video=bool(args.record_video))
                write_json_atomic(cache_path, episode)
                episodes.append(episode)
                if progress is not None and item_task is not None:
                    progress.update(item_task, advance=1)
            aggregate = _build_aggregate(
                model_name,
                episodes,
                benchmark_id=args.benchmark_id,
                benchmark_fingerprint=benchmark_fingerprint,
                upstream_commit=manifest["upstream_commit"],
            )
            compare_report["aggregates"].append(aggregate)
            summary_ts = current_timestamp()
            eval_dir = ensure_dir(os.path.join(model_root, "eval"))
            summary_path = os.path.join(eval_dir, f"eval_summary_{summary_ts}.json")
            episodes_path = os.path.join(eval_dir, f"eval_episodes_{summary_ts}.json")
            metrics_config = copy.deepcopy(compare_report["metrics_config"])
            write_json_atomic(summary_path, _build_model_extract(model_name, experiment_id, aggregate, episodes, metrics_config))
            write_json_atomic(episodes_path, {"episodes": episodes})
            model_outputs[model_name] = {"summary_path": summary_path, "episodes_path": episodes_path}
            emit(
                f"[cyan]{model_name}[/cyan] "
                f"task_success_rate={aggregate['task_success_rate']:.3f} "
                f"policy_failures={aggregate['policy_failure_count']}"
            )
            if progress is not None and model_task is not None:
                progress.update(model_task, advance=1)
            if progress is not None and item_task is not None:
                progress.remove_task(item_task)
                item_task = None

    compare_dir = ensure_dir(os.path.join(experiment_root, "compare"))
    compare_path = os.path.join(compare_dir, f"eval_compare_{current_timestamp()}.json")
    write_json_atomic(compare_path, compare_report)
    _update_manifest(
        experiment_root=experiment_root,
        experiment_id=experiment_id,
        config_path=args.config,
        benchmark_id=args.benchmark_id,
        benchmark_fingerprint=benchmark_fingerprint,
        few_shot=bool(args.few_shot),
        model_outputs=model_outputs,
        compare_report_path=compare_path,
    )
    return {
        "experiment_root": experiment_root,
        "compare_report_path": compare_path,
        "model_outputs": model_outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical LaMPilot highway benchmark port.")
    parser.add_argument("--config", default="config.yaml", help="Path to runtime config.")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to evaluate.")
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID, help="Benchmark id under benchmarks/.")
    parser.add_argument("--test-size", type=int, default=None, help="Limit expanded benchmark items.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle expanded benchmark items before truncation.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for dataset shuffle.")
    parser.add_argument("--few-shot", action="store_true", help="Use few-shot code-generation prompt examples.")
    parser.add_argument("--use-demo", action="store_true", help="Evaluate the canonical upstream-style demo subset only.")
    parser.add_argument("--progress", action="store_true", help="Show CLI progress bars.")
    parser.add_argument("--no-progress", action="store_true", help="Disable CLI progress bars.")
    parser.add_argument("--resume", action="store_true", help="Reuse cached item results when present.")
    parser.add_argument("--record-video", action="store_true", help="Record evaluator videos.")
    parser.add_argument("--results-root", default=None, help="Optional structured results root override.")
    parser.add_argument("--experiment-id", default=None, help="Optional experiment id override.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
