import argparse
import copy
import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

from rich import print

from dilu.runtime import read_json, slugify_model_name, timestamped_results_path, write_json_atomic


COMPAT_METRIC_KEYS = [
    "ttc_threshold_sec",
    "headway_threshold_m",
    "decision_timeout_sec",
    "decision_max_output_tokens",
    "disable_streaming",
    "disable_checker_llm",
    "ollama_think_mode",
    "ollama_use_native_chat",
    "ollama_native_chat_timeout_sec",
]


def _resolve_existing_path(raw_path: Optional[str], experiment_root: str) -> Optional[str]:
    if not raw_path:
        return None
    raw = str(raw_path).strip()
    if not raw:
        return None
    normalized = os.path.normpath(raw)
    candidates = []
    if os.path.isabs(normalized):
        candidates.append(normalized)
    else:
        candidates.append(normalized)
        candidates.append(os.path.normpath(os.path.join(experiment_root, normalized)))
        candidates.append(os.path.normpath(os.path.join(os.getcwd(), normalized)))
    seen = set()
    for cand in candidates:
        abs_cand = os.path.abspath(cand)
        if abs_cand in seen:
            continue
        seen.add(abs_cand)
        if os.path.exists(abs_cand):
            return abs_cand
    return None


def _latest_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda p: os.path.getmtime(p))


def _find_matching_episodes(summary_path: str, eval_dir: str) -> Optional[str]:
    m = re.match(r"^eval_summary_(.+)\.json$", os.path.basename(summary_path))
    if m:
        ts = m.group(1)
        sibling = os.path.join(eval_dir, f"eval_episodes_{ts}.json")
        if os.path.exists(sibling):
            return sibling
    return _latest_file(os.path.join(eval_dir, "eval_episodes_*.json"))


def _lookup_model_entry(models: Dict, model_name: str) -> Optional[Dict]:
    if not isinstance(models, dict):
        return None
    if model_name in models and isinstance(models[model_name], dict):
        return models[model_name]
    target = str(model_name).strip().lower()
    for key, value in models.items():
        if str(key).strip().lower() == target and isinstance(value, dict):
            return value
    return None


def _discover_model_artifacts(experiment_root: str, model_name: str) -> Optional[Dict]:
    manifest_path = os.path.join(experiment_root, "manifest.json")
    manifest = read_json(manifest_path, default={})
    model_entry = _lookup_model_entry(manifest.get("models", {}), model_name)
    model_slug = model_entry.get("slug") if isinstance(model_entry, dict) else None
    model_slug = model_slug or slugify_model_name(model_name)
    eval_dir = os.path.join(experiment_root, "models", model_slug, "eval")

    summary_path = None
    episodes_path = None
    source_kind = None

    # 1) Manifest pointers first.
    if isinstance(model_entry, dict):
        summary_path = _resolve_existing_path(model_entry.get("latest_eval_summary"), experiment_root)
        if summary_path:
            episodes_path = _resolve_existing_path(model_entry.get("latest_eval_episodes"), experiment_root)
            if not episodes_path and os.path.isdir(eval_dir):
                episodes_path = _find_matching_episodes(summary_path, eval_dir)
            source_kind = "manifest"

    # 2) Filesystem fallback.
    if not summary_path and os.path.isdir(eval_dir):
        newest_summary = _latest_file(os.path.join(eval_dir, "eval_summary_*.json"))
        if newest_summary:
            summary_path = os.path.abspath(newest_summary)
            episodes_path = _find_matching_episodes(summary_path, eval_dir)
            source_kind = "filesystem_latest"

    if not summary_path:
        return None

    summary_payload = read_json(summary_path, default={})
    source_compare_report = _resolve_existing_path(summary_payload.get("source_compare_report"), experiment_root)
    return {
        "experiment_root": experiment_root,
        "experiment_id": manifest.get("experiment_id") or os.path.basename(experiment_root),
        "manifest_path": manifest_path if os.path.exists(manifest_path) else None,
        "manifest": manifest,
        "model_name": model_name,
        "model_slug": model_slug,
        "summary_path": summary_path,
        "episodes_path": os.path.abspath(episodes_path) if episodes_path and os.path.exists(episodes_path) else None,
        "source_compare_report": source_compare_report,
        "source_kind": source_kind,
        "summary_mtime": os.path.getmtime(summary_path),
    }


def _discover_available_models(experiment_root: str) -> List[Dict]:
    manifest_path = os.path.join(experiment_root, "manifest.json")
    manifest = read_json(manifest_path, default={})
    rows_by_key: Dict[str, Dict] = {}

    models = manifest.get("models", {})
    if isinstance(models, dict):
        for model_name, entry in models.items():
            if not isinstance(entry, dict):
                continue
            model_str = str(model_name)
            key = model_str.strip().lower()
            slug = entry.get("slug") or slugify_model_name(str(model_name))
            summary_path = _resolve_existing_path(entry.get("latest_eval_summary"), experiment_root)
            episodes_path = _resolve_existing_path(entry.get("latest_eval_episodes"), experiment_root)
            rows_by_key[key] = {
                "model": model_str,
                "slug": slug,
                "source_kind": "manifest",
                "summary_path": summary_path,
                "episodes_path": episodes_path,
            }

    model_root = os.path.join(experiment_root, "models")
    if os.path.isdir(model_root):
        for slug in sorted(os.listdir(model_root)):
            eval_dir = os.path.join(model_root, slug, "eval")
            if not os.path.isdir(eval_dir):
                continue
            newest_summary = _latest_file(os.path.join(eval_dir, "eval_summary_*.json"))
            if not newest_summary:
                continue
            newest_episodes = _find_matching_episodes(newest_summary, eval_dir)
            inferred_name = str(slug)
            # Try to recover canonical model name from summary payload.
            try:
                payload = read_json(newest_summary, default={})
                inferred_name = str(payload.get("model") or inferred_name)
            except Exception:
                pass
            key = inferred_name.strip().lower()
            candidate = {
                "model": inferred_name,
                "slug": slug,
                "source_kind": "filesystem_latest",
                "summary_path": os.path.abspath(newest_summary),
                "episodes_path": os.path.abspath(newest_episodes) if newest_episodes else None,
            }
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = candidate
                continue
            # Prefer filesystem fallback if manifest entry exists but has no valid summary.
            if not existing.get("summary_path"):
                rows_by_key[key] = candidate

    rows = list(rows_by_key.values())
    rows.sort(key=lambda r: str(r.get("model", "")))
    return rows


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in items:
        name = str(raw).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _discover_available_model_names(experiment_root: str) -> List[str]:
    rows = _discover_available_models(experiment_root)
    discovered: List[str] = []
    for row in rows:
        model_name = str(row.get("model", "")).strip()
        if not model_name:
            continue
        # Ensure the model still resolves through the main artifact discovery path.
        if _discover_model_artifacts(experiment_root, model_name):
            discovered.append(model_name)
    return _dedupe_preserve_order(discovered)


def _read_episodes(summary_payload: Dict, episodes_path: Optional[str]) -> List[Dict]:
    if episodes_path and os.path.exists(episodes_path):
        payload = read_json(episodes_path, default={})
        episodes = payload.get("episodes")
        if isinstance(episodes, list):
            return episodes
    episodes = summary_payload.get("episodes")
    if isinstance(episodes, list):
        return episodes
    raise ValueError("Missing episodes list in both eval_episodes file and eval_summary payload.")


def _compat_profile(model_name: str, source: Dict, summary_payload: Dict, episodes: List[Dict]) -> Dict:
    manifest = source.get("manifest", {}) or {}
    few_shot_num = manifest.get("few_shot_num")
    simulation_duration = manifest.get("simulation_duration")
    if simulation_duration is None and episodes:
        simulation_duration = episodes[0].get("max_steps")
    if few_shot_num is None or simulation_duration is None:
        raise ValueError(
            f"Model '{model_name}' is missing few_shot_num/simulation_duration in manifest; cannot enforce strict compatibility."
        )
    seeds = []
    for ep in episodes:
        if "seed" not in ep:
            raise ValueError(f"Model '{model_name}' has episode without 'seed'.")
        seeds.append(int(ep["seed"]))
    metrics_config = summary_payload.get("metrics_config", {}) or {}
    metrics_subset = {k: metrics_config.get(k) for k in COMPAT_METRIC_KEYS}
    return {
        "few_shot_num": int(few_shot_num),
        "simulation_duration": int(simulation_duration),
        "seeds": seeds,
        "metrics_subset": metrics_subset,
    }


def _compare_profiles(base_model: str, base: Dict, model_name: str, candidate: Dict) -> List[str]:
    diffs = []
    if candidate["few_shot_num"] != base["few_shot_num"]:
        diffs.append(
            f"few_shot_num mismatch: {model_name}={candidate['few_shot_num']} vs {base_model}={base['few_shot_num']}"
        )
    if candidate["simulation_duration"] != base["simulation_duration"]:
        diffs.append(
            "simulation_duration mismatch: "
            f"{model_name}={candidate['simulation_duration']} vs {base_model}={base['simulation_duration']}"
        )
    if candidate["seeds"] != base["seeds"]:
        diffs.append(f"seed list mismatch: {model_name}={candidate['seeds']} vs {base_model}={base['seeds']}")
    for key in COMPAT_METRIC_KEYS:
        left = candidate["metrics_subset"].get(key)
        right = base["metrics_subset"].get(key)
        if left != right:
            diffs.append(f"metrics_config.{key} mismatch: {model_name}={left} vs {base_model}={right}")
    return diffs


def _update_manifest_for_merged_report(experiment_root: str, experiment_id: str, report_path: str) -> None:
    manifest_path = os.path.join(experiment_root, "manifest.json")
    manifest = read_json(manifest_path, default={})
    now = datetime.now().isoformat(timespec="seconds")
    manifest.setdefault("experiment_id", experiment_id)
    manifest.setdefault("created_at", now)
    manifest["updated_at"] = now
    manifest["config_path"] = "merge_eval_reports.py"
    compare = manifest.setdefault("compare", {})
    compare["latest_report"] = report_path
    history = compare.setdefault("history", [])
    if report_path not in history:
        history.append(report_path)
    write_json_atomic(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge latest per-model eval outputs inside a single experiment id."
    )
    parser.add_argument("--experiment-id", required=True, help="Experiment id under --results-root.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to merge. If omitted, all available models with latest eval artifacts are merged.",
    )
    parser.add_argument(
        "--include-available",
        action="store_true",
        help="When used with --models, merge the union of explicit models and all available models.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models with available latest eval artifacts in the given experiment and exit.",
    )
    parser.add_argument("--results-root", default="results", help="Root folder containing experiments.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path. Default: results/<experiment-id>/compare/eval_compare_<timestamp>.json",
    )
    args = parser.parse_args()

    experiment_root = os.path.abspath(os.path.join(args.results_root, args.experiment_id))
    if not os.path.isdir(experiment_root):
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_root}. Expected at <results-root>/<experiment-id>."
        )
    if args.list_models:
        rows = _discover_available_models(experiment_root)
        if not rows:
            print(
                f"[yellow]No model eval artifacts found under experiment '{args.experiment_id}'.[/yellow]"
            )
            return
        print(f"[bold cyan]Available models in {args.experiment_id}[/bold cyan]:")
        for row in rows:
            print(
                "- {model} | slug={slug} | source={source} | summary={summary}".format(
                    model=row.get("model"),
                    slug=row.get("slug"),
                    source=row.get("source_kind"),
                    summary=row.get("summary_path"),
                )
            )
        return

    available_models = _discover_available_model_names(experiment_root)
    if args.models and args.include_available:
        selected_models = _dedupe_preserve_order(list(args.models) + available_models)
    elif args.models:
        selected_models = _dedupe_preserve_order(list(args.models))
    else:
        selected_models = available_models

    if not selected_models:
        raise ValueError(
            "No mergeable model eval summaries found in this experiment. "
            "Run evaluations first or pass explicit --models."
        )

    per_model_payloads: Dict[str, Dict] = {}
    model_sources: Dict[str, Dict] = {}
    compatibility_profiles: Dict[str, Dict] = {}
    mismatches: List[str] = []
    baseline_model = None
    baseline_profile = None

    for model_name in selected_models:
        source = _discover_model_artifacts(experiment_root, model_name)
        if not source:
            raise FileNotFoundError(
                f"No eval summary found for model '{model_name}' inside experiment '{args.experiment_id}'."
            )
        summary_payload = read_json(source["summary_path"], default={})
        aggregate = summary_payload.get("aggregate")
        if not isinstance(aggregate, dict):
            raise ValueError(f"Invalid eval summary for model '{model_name}': missing 'aggregate'.")
        episodes = _read_episodes(summary_payload, source.get("episodes_path"))
        profile = _compat_profile(model_name, source, summary_payload, episodes)

        if baseline_model is None:
            baseline_model = model_name
            baseline_profile = profile
        else:
            mismatches.extend(_compare_profiles(baseline_model, baseline_profile, model_name, profile))

        aggregate_copy = copy.deepcopy(aggregate)
        aggregate_copy["model"] = model_name
        per_model_payloads[model_name] = {
            "aggregate": aggregate_copy,
            "episodes": copy.deepcopy(episodes),
            "metrics_config": copy.deepcopy(summary_payload.get("metrics_config", {})),
        }
        model_sources[model_name] = {
            "source_kind": source["source_kind"],
            "summary_path": source["summary_path"],
            "episodes_path": source["episodes_path"],
            "source_compare_report": source["source_compare_report"],
            "summary_mtime": datetime.fromtimestamp(source["summary_mtime"]).isoformat(timespec="seconds"),
        }
        compatibility_profiles[model_name] = profile

    if mismatches:
        message = ["Strict compatibility check failed:"]
        message.extend([f"- {d}" for d in mismatches])
        raise ValueError("\n".join(message))

    compare_dir = os.path.join(experiment_root, "compare")
    os.makedirs(compare_dir, exist_ok=True)
    out_path = args.output or timestamped_results_path("eval_compare", ext=".json", results_dir=compare_dir)

    ordered_aggregates = [per_model_payloads[m]["aggregate"] for m in selected_models]
    ordered_per_model = {m: per_model_payloads[m]["episodes"] for m in selected_models}
    baseline_metrics = copy.deepcopy(per_model_payloads[selected_models[0]]["metrics_config"])
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_schema_version": "2.0",
        "source": "merge_eval_reports",
        "compatibility_mode": "strict",
        "experiment_id": args.experiment_id,
        "experiment_root": experiment_root,
        "compare_dir": compare_dir,
        "models": list(selected_models),
        "seeds": baseline_profile["seeds"],
        "few_shot_num": baseline_profile["few_shot_num"],
        "simulation_duration": baseline_profile["simulation_duration"],
        "metrics_config": baseline_metrics,
        "aggregates": ordered_aggregates,
        "per_model": ordered_per_model,
        "model_sources": model_sources,
        "strict_compatibility_profile": compatibility_profiles,
        "alignment_samples": [],
        "model_eval_outputs": {},
        "model_run_outputs": {},
    }

    write_json_atomic(out_path, report)
    _update_manifest_for_merged_report(experiment_root, args.experiment_id, out_path)

    print(f"[green]Merged compare report:[/green] {out_path}")
    for model_name in selected_models:
        src = model_sources[model_name]
        print(f"- {model_name}: {src['source_kind']} | {os.path.basename(src['summary_path'])}")


if __name__ == "__main__":
    main()
