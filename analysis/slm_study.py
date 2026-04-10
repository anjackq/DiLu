from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VALID_TIERS = ("lightweight", "midclass", "highclass")
REGISTRY_FIELDS = (
    "model_id",
    "display_name",
    "ollama_tag",
    "family",
    "variant_kind",
    "base_model_id",
    "param_count_b",
    "tier",
    "notes",
)
NORMALIZED_FIELDS = (
    "study_stage",
    "source_compare_report",
    "experiment_id",
    "benchmark_case_set",
    "headline_task_metric",
    "efficiency_metrics_reported",
    "model_id",
    "display_name",
    "ollama_tag",
    "family",
    "variant_kind",
    "base_model_id",
    "param_count_b",
    "tier",
    "notes",
    "registry_match",
    "pair_group_id",
    "crash_rate",
    "collision_rate",
    "task_completion_rate",
    "overall_score_mean",
    "driving_score_v2",
    "decision_latency_ms_avg_mean",
    "p95_decision_latency_sec_mean",
    "tokens_per_second_mean",
    "net_energy_j_mean",
    "energy_per_decision_j_mean",
    "benchmark_result_valid",
    "benchmark_result_invalid_reason",
    "model_skipped_due_to_preflight",
    "model_quarantined_due_to_timeout_collapse",
    "episode_execution_complete",
    "status",
    "ranking_eligible",
)
LEADERBOARD_FIELDS = (
    "rank",
    "tier",
    "model_id",
    "display_name",
    "ollama_tag",
    "family",
    "variant_kind",
    "driving_score_v2",
    "task_completion_rate",
    "overall_score_mean",
    "crash_rate",
    "decision_latency_ms_avg_mean",
    "p95_decision_latency_sec_mean",
    "tokens_per_second_mean",
    "net_energy_j_mean",
    "energy_per_decision_j_mean",
    "status",
    "benchmark_result_invalid_reason",
)
PAIRED_DELTA_FIELDS = (
    "family",
    "tier",
    "base_model_id",
    "base_display_name",
    "base_ollama_tag",
    "fine_tuned_model_id",
    "fine_tuned_display_name",
    "fine_tuned_ollama_tag",
    "pair_eligible",
    "pair_issue",
    "delta_task_completion_rate",
    "delta_overall_score_mean",
    "delta_driving_score_v2",
    "delta_decision_latency_ms_avg_mean",
    "delta_p95_decision_latency_sec_mean",
    "delta_tokens_per_second_mean",
    "delta_net_energy_j_mean",
    "delta_energy_per_decision_j_mean",
)
FAMILY_SUMMARY_FIELDS = (
    "family",
    "base_model_count",
    "fine_tuned_model_count",
    "eligible_pair_count",
    "best_base_model_id",
    "best_base_display_name",
    "best_base_driving_score_v2",
    "best_fine_tuned_model_id",
    "best_fine_tuned_display_name",
    "best_fine_tuned_driving_score_v2",
    "avg_delta_task_completion_rate",
    "avg_delta_overall_score_mean",
    "avg_delta_driving_score_v2",
    "avg_delta_decision_latency_ms_avg_mean",
    "avg_delta_p95_decision_latency_sec_mean",
    "avg_delta_tokens_per_second_mean",
    "avg_delta_net_energy_j_mean",
    "avg_delta_energy_per_decision_j_mean",
)
SHORTLIST_FIELDS = (
    "tier",
    "shortlist_rank",
    "model_id",
    "display_name",
    "ollama_tag",
    "family",
    "variant_kind",
    "driving_score_v2",
    "task_completion_rate",
    "decision_latency_ms_avg_mean",
    "overall_score_mean",
)
INVALID_RUN_FIELDS = (
    "study_stage",
    "tier",
    "model_id",
    "display_name",
    "ollama_tag",
    "status",
    "benchmark_result_valid",
    "benchmark_result_invalid_reason",
    "model_skipped_due_to_preflight",
    "model_quarantined_due_to_timeout_collapse",
    "episode_execution_complete",
    "driving_score_v2",
    "task_completion_rate",
    "decision_latency_ms_avg_mean",
)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in writer.fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _infer_tier(param_count_b: float) -> str:
    if param_count_b <= 4.0:
        return "lightweight"
    if param_count_b <= 12.0:
        return "midclass"
    if param_count_b <= 32.0:
        return "highclass"
    raise ValueError(f"param_count_b={param_count_b} falls outside supported study tiers")


def _infer_param_count_from_tag(ollama_tag: str) -> float | None:
    match = re.search(r":(\d+(?:\.\d+)?)b\b", ollama_tag, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"[-_](\d+(?:\.\d+)?)b\b", ollama_tag, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _infer_family(ollama_tag: str) -> str:
    match = re.match(r"([a-z]+)", (ollama_tag or "").lower())
    return match.group(1) if match else "unknown"


def _status_from_record(record: dict[str, Any]) -> str:
    flags: list[str] = []
    if _to_bool(record.get("model_skipped_due_to_preflight")):
        flags.append("preflight_skip")
    if _to_bool(record.get("model_quarantined_due_to_timeout_collapse")):
        flags.append("quarantined")
    if not _to_bool(record.get("episode_execution_complete")):
        flags.append("incomplete")
    if not _to_bool(record.get("benchmark_result_valid")):
        flags.append("invalid")
    return "valid" if not flags else "+".join(flags)


def _ranking_eligible(record: dict[str, Any]) -> bool:
    return (
        _to_bool(record.get("benchmark_result_valid"))
        and not _to_bool(record.get("model_skipped_due_to_preflight"))
        and not _to_bool(record.get("model_quarantined_due_to_timeout_collapse"))
        and _to_bool(record.get("episode_execution_complete"))
        and _to_float(record.get("driving_score_v2")) is not None
    )


def _report_rank_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    latency = _to_float(record.get("decision_latency_ms_avg_mean"))
    return (
        -(_to_float(record.get("driving_score_v2")) or float("-inf")),
        -(_to_float(record.get("task_completion_rate")) or float("-inf")),
        latency if latency is not None else float("inf"),
        record.get("display_name") or record.get("model_id") or "",
    )


def _ensure_lampilot_report(payload: dict[str, Any], report_path: Path) -> None:
    if not _to_bool(payload.get("benchmark_mode")):
        raise ValueError(f"{report_path} is not a benchmark compare report")
    case_set = str(payload.get("benchmark_case_set") or "")
    if case_set != "lampilot_highway_v1":
        raise ValueError(f"{report_path} targets unsupported benchmark_case_set={case_set!r}")
    if payload.get("headline_task_metric") and payload.get("headline_task_metric") != "driving_score_v2":
        raise ValueError(f"{report_path} does not use driving_score_v2 as the headline task metric")


def _dedupe_paths(paths: Iterable[str | Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _discover_compare_reports(experiment_root_paths: Iterable[str | Path] | None) -> list[Path]:
    discovered: list[Path] = []
    for raw_root in experiment_root_paths or []:
        root = Path(raw_root).expanduser().resolve()
        if root.is_file() and root.suffix.lower() == ".json":
            discovered.append(root)
            continue
        if not root.exists():
            raise ValueError(f"experiment root does not exist: {root}")
        if root.name == "compare":
            discovered.extend(sorted(root.glob("*.json")))
            continue
        discovered.extend(sorted(root.glob("compare/*.json")))
    return _dedupe_paths(discovered)


def load_registry(registry_path: str | Path) -> dict[str, Any]:
    path = Path(registry_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} is missing registry headers")
        missing = [field for field in REGISTRY_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required registry columns: {', '.join(missing)}")
        rows = list(reader)

    registry_rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    by_tag: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows, start=2):
        row = {field: (raw_row.get(field) or "").strip() for field in REGISTRY_FIELDS}
        if not row["model_id"]:
            raise ValueError(f"{path}:{index} model_id is required")
        if row["model_id"] in by_id:
            raise ValueError(f"{path}:{index} duplicate model_id={row['model_id']}")
        if not row["ollama_tag"]:
            raise ValueError(f"{path}:{index} ollama_tag is required")
        if row["ollama_tag"] in by_tag:
            raise ValueError(f"{path}:{index} duplicate ollama_tag={row['ollama_tag']}")
        if not row["display_name"]:
            raise ValueError(f"{path}:{index} display_name is required")
        if not row["family"]:
            raise ValueError(f"{path}:{index} family is required")
        variant_kind = row["variant_kind"].lower()
        if variant_kind not in {"base", "fine_tuned"}:
            raise ValueError(f"{path}:{index} variant_kind must be 'base' or 'fine_tuned'")
        param_count = _to_float(row["param_count_b"])
        if param_count is None or param_count <= 0:
            raise ValueError(f"{path}:{index} param_count_b must be a positive number")
        tier = row["tier"].lower()
        if tier not in VALID_TIERS:
            raise ValueError(f"{path}:{index} tier must be one of {', '.join(VALID_TIERS)}")
        inferred_tier = _infer_tier(param_count)
        if tier != inferred_tier:
            raise ValueError(
                f"{path}:{index} tier mismatch for model_id={row['model_id']}: "
                f"declared {tier}, inferred {inferred_tier}"
            )
        if variant_kind == "fine_tuned" and not row["base_model_id"]:
            raise ValueError(f"{path}:{index} fine_tuned rows require base_model_id")
        if variant_kind == "base" and row["base_model_id"]:
            raise ValueError(f"{path}:{index} base rows must not declare base_model_id")

        normalized = {
            **row,
            "variant_kind": variant_kind,
            "param_count_b": param_count,
            "tier": tier,
            "pair_group_id": row["base_model_id"] or row["model_id"],
        }
        registry_rows.append(normalized)
        by_id[normalized["model_id"]] = normalized
        by_tag[normalized["ollama_tag"]] = normalized

    for row in registry_rows:
        if row["variant_kind"] != "fine_tuned":
            continue
        base = by_id.get(row["base_model_id"])
        if base is None:
            raise ValueError(
                f"{path} fine_tuned model_id={row['model_id']} references missing base_model_id={row['base_model_id']}"
            )
        if base["variant_kind"] != "base":
            raise ValueError(
                f"{path} fine_tuned model_id={row['model_id']} references non-base base_model_id={row['base_model_id']}"
            )

    return {"path": str(path), "rows": registry_rows, "by_id": by_id, "by_tag": by_tag}


def _fallback_registry_row(ollama_tag: str) -> dict[str, Any]:
    param_count = _infer_param_count_from_tag(ollama_tag)
    tier = _infer_tier(param_count) if param_count is not None else ""
    return {
        "model_id": _slugify(ollama_tag),
        "display_name": ollama_tag,
        "ollama_tag": ollama_tag,
        "family": _infer_family(ollama_tag),
        "variant_kind": "unknown",
        "base_model_id": "",
        "param_count_b": param_count,
        "tier": tier,
        "notes": "registry_missing",
        "pair_group_id": "",
    }


def _normalize_records(
    report_paths: Iterable[Path],
    registry: dict[str, Any],
    *,
    study_stage: str,
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for report_path in report_paths:
        with report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _ensure_lampilot_report(payload, report_path)
        aggregates = payload.get("aggregates") or []
        if not isinstance(aggregates, list):
            raise ValueError(f"{report_path} has non-list aggregates")
        for aggregate in aggregates:
            ollama_tag = str(aggregate.get("model") or "").strip()
            if not ollama_tag:
                raise ValueError(f"{report_path} contains an aggregate row without model")

            registry_row = registry["by_tag"].get(ollama_tag)
            benchmark_valid = _to_bool(aggregate.get("benchmark_result_valid", True))
            skipped = _to_bool(aggregate.get("model_skipped_due_to_preflight"))
            quarantined = _to_bool(aggregate.get("model_quarantined_due_to_timeout_collapse"))
            execution_complete = _to_bool(aggregate.get("episode_execution_complete", True))

            if registry_row is None and benchmark_valid and not skipped and not quarantined and execution_complete:
                raise ValueError(
                    f"{report_path} contains registry-missing ranked model {ollama_tag}. "
                    "Add it to the study registry before analysis."
                )
            if registry_row is None:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "missing_registry_row",
                        "study_stage": study_stage,
                        "model": ollama_tag,
                        "report": str(report_path),
                    }
                )
                registry_row = _fallback_registry_row(ollama_tag)

            record = {
                "study_stage": study_stage,
                "source_compare_report": str(report_path),
                "experiment_id": payload.get("experiment_id") or report_path.stem,
                "benchmark_case_set": payload.get("benchmark_case_set") or "",
                "headline_task_metric": payload.get("headline_task_metric") or "driving_score_v2",
                "efficiency_metrics_reported": _to_bool(payload.get("efficiency_metrics_reported", True)),
                "model_id": registry_row["model_id"],
                "display_name": registry_row["display_name"],
                "ollama_tag": ollama_tag,
                "family": registry_row["family"],
                "variant_kind": registry_row["variant_kind"],
                "base_model_id": registry_row.get("base_model_id", ""),
                "param_count_b": registry_row.get("param_count_b"),
                "tier": registry_row.get("tier", ""),
                "notes": registry_row.get("notes", ""),
                "registry_match": registry_row.get("notes") != "registry_missing",
                "pair_group_id": registry_row.get("pair_group_id") or registry_row.get("model_id") or "",
                "crash_rate": _to_float(aggregate.get("crash_rate")),
                "collision_rate": _to_float(aggregate.get("crash_rate")),
                "task_completion_rate": _to_float(aggregate.get("task_completion_rate")),
                "overall_score_mean": _to_float(aggregate.get("overall_score_mean")),
                "driving_score_v2": _to_float(aggregate.get("driving_score_v2")),
                "decision_latency_ms_avg_mean": _to_float(aggregate.get("decision_latency_ms_avg_mean")),
                "p95_decision_latency_sec_mean": _to_float(aggregate.get("p95_decision_latency_sec_mean")),
                "tokens_per_second_mean": _to_float(aggregate.get("tokens_per_second_mean")),
                "net_energy_j_mean": _to_float(aggregate.get("net_energy_j_mean")),
                "energy_per_decision_j_mean": _to_float(aggregate.get("energy_per_decision_j_mean")),
                "benchmark_result_valid": benchmark_valid,
                "benchmark_result_invalid_reason": aggregate.get("benchmark_result_invalid_reason") or "",
                "model_skipped_due_to_preflight": skipped,
                "model_quarantined_due_to_timeout_collapse": quarantined,
                "episode_execution_complete": execution_complete,
            }
            record["status"] = _status_from_record(record)
            record["ranking_eligible"] = _ranking_eligible(record) and bool(record.get("tier"))
            normalized.append(record)
    return normalized


def _validate_unique_model_stage(records: Iterable[dict[str, Any]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for record in records:
        key = (record["study_stage"], record["model_id"])
        existing_report = seen.get(key)
        if existing_report is not None:
            raise ValueError(
                f"Duplicate study entry for stage={record['study_stage']} model_id={record['model_id']}: "
                f"{existing_report} and {record['source_compare_report']}"
            )
        seen[key] = record["source_compare_report"]


def _build_leaderboard_rows(records: list[dict[str, Any]], tier: str) -> list[dict[str, Any]]:
    tier_records = [record for record in records if record.get("tier") == tier]
    valid_rows = sorted(
        [record for record in tier_records if _to_bool(record.get("ranking_eligible"))],
        key=_report_rank_sort_key,
    )
    invalid_rows = sorted(
        [record for record in tier_records if not _to_bool(record.get("ranking_eligible"))],
        key=lambda record: (
            record.get("status") or "",
            -(_to_float(record.get("driving_score_v2")) or float("-inf")),
            record.get("display_name") or "",
        ),
    )

    leaderboard_rows: list[dict[str, Any]] = []
    for rank, record in enumerate(valid_rows, start=1):
        leaderboard_rows.append(
            {
                "rank": rank,
                "tier": tier,
                "model_id": record["model_id"],
                "display_name": record["display_name"],
                "ollama_tag": record["ollama_tag"],
                "family": record["family"],
                "variant_kind": record["variant_kind"],
                "driving_score_v2": record["driving_score_v2"],
                "task_completion_rate": record["task_completion_rate"],
                "overall_score_mean": record["overall_score_mean"],
                "crash_rate": record["crash_rate"],
                "decision_latency_ms_avg_mean": record["decision_latency_ms_avg_mean"],
                "p95_decision_latency_sec_mean": record["p95_decision_latency_sec_mean"],
                "tokens_per_second_mean": record["tokens_per_second_mean"],
                "net_energy_j_mean": record["net_energy_j_mean"],
                "energy_per_decision_j_mean": record["energy_per_decision_j_mean"],
                "status": record["status"],
                "benchmark_result_invalid_reason": record["benchmark_result_invalid_reason"],
            }
        )
    for record in invalid_rows:
        leaderboard_rows.append(
            {
                "rank": "",
                "tier": tier,
                "model_id": record["model_id"],
                "display_name": record["display_name"],
                "ollama_tag": record["ollama_tag"],
                "family": record["family"],
                "variant_kind": record["variant_kind"],
                "driving_score_v2": record["driving_score_v2"],
                "task_completion_rate": record["task_completion_rate"],
                "overall_score_mean": record["overall_score_mean"],
                "crash_rate": record["crash_rate"],
                "decision_latency_ms_avg_mean": record["decision_latency_ms_avg_mean"],
                "p95_decision_latency_sec_mean": record["p95_decision_latency_sec_mean"],
                "tokens_per_second_mean": record["tokens_per_second_mean"],
                "net_energy_j_mean": record["net_energy_j_mean"],
                "energy_per_decision_j_mean": record["energy_per_decision_j_mean"],
                "status": record["status"],
                "benchmark_result_invalid_reason": record["benchmark_result_invalid_reason"],
            }
        )
    return leaderboard_rows


def _build_paired_deltas(records: list[dict[str, Any]], issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model_id = {record["model_id"]: record for record in records}
    paired_rows: list[dict[str, Any]] = []
    metric_names = (
        "task_completion_rate",
        "overall_score_mean",
        "driving_score_v2",
        "decision_latency_ms_avg_mean",
        "p95_decision_latency_sec_mean",
        "tokens_per_second_mean",
        "net_energy_j_mean",
        "energy_per_decision_j_mean",
    )
    for record in sorted(records, key=lambda item: item["model_id"]):
        if record.get("variant_kind") != "fine_tuned":
            continue
        base = by_model_id.get(record.get("base_model_id"))
        pair_issue = ""
        pair_eligible = False
        if base is None:
            pair_issue = f"missing_base_model_id:{record.get('base_model_id')}"
            issues.append(
                {
                    "severity": "warning",
                    "code": "unpaired_fine_tuned_model",
                    "fine_tuned_model_id": record["model_id"],
                    "base_model_id": record.get("base_model_id"),
                }
            )
        else:
            pair_eligible = _to_bool(record.get("ranking_eligible")) and _to_bool(base.get("ranking_eligible"))
            if not pair_eligible:
                pair_issue = "base_or_fine_tuned_not_rankable"

        paired_row = {
            "family": record["family"],
            "tier": record["tier"],
            "base_model_id": record.get("base_model_id", ""),
            "base_display_name": base.get("display_name", "") if base else "",
            "base_ollama_tag": base.get("ollama_tag", "") if base else "",
            "fine_tuned_model_id": record["model_id"],
            "fine_tuned_display_name": record["display_name"],
            "fine_tuned_ollama_tag": record["ollama_tag"],
            "pair_eligible": pair_eligible,
            "pair_issue": pair_issue,
        }
        for metric_name in metric_names:
            delta_key = f"delta_{metric_name}"
            if pair_eligible and base is not None:
                ft_value = _to_float(record.get(metric_name))
                base_value = _to_float(base.get(metric_name))
                paired_row[delta_key] = None if ft_value is None or base_value is None else ft_value - base_value
            else:
                paired_row[delta_key] = None
        paired_rows.append(paired_row)
    return paired_rows


def _build_family_summary(
    records: list[dict[str, Any]], paired_deltas: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)

    summary_rows: list[dict[str, Any]] = []
    delta_fields = (
        "delta_task_completion_rate",
        "delta_overall_score_mean",
        "delta_driving_score_v2",
        "delta_decision_latency_ms_avg_mean",
        "delta_p95_decision_latency_sec_mean",
        "delta_tokens_per_second_mean",
        "delta_net_energy_j_mean",
        "delta_energy_per_decision_j_mean",
    )
    for family in sorted(by_family):
        family_records = by_family[family]
        valid_bases = sorted(
            [record for record in family_records if record["variant_kind"] == "base" and _to_bool(record["ranking_eligible"])],
            key=_report_rank_sort_key,
        )
        valid_fine_tuned = sorted(
            [
                record
                for record in family_records
                if record["variant_kind"] == "fine_tuned" and _to_bool(record["ranking_eligible"])
            ],
            key=_report_rank_sort_key,
        )
        eligible_pairs = [
            row for row in paired_deltas if row["family"] == family and _to_bool(row.get("pair_eligible"))
        ]

        summary = {
            "family": family,
            "base_model_count": sum(1 for record in family_records if record["variant_kind"] == "base"),
            "fine_tuned_model_count": sum(1 for record in family_records if record["variant_kind"] == "fine_tuned"),
            "eligible_pair_count": len(eligible_pairs),
            "best_base_model_id": valid_bases[0]["model_id"] if valid_bases else "",
            "best_base_display_name": valid_bases[0]["display_name"] if valid_bases else "",
            "best_base_driving_score_v2": valid_bases[0]["driving_score_v2"] if valid_bases else None,
            "best_fine_tuned_model_id": valid_fine_tuned[0]["model_id"] if valid_fine_tuned else "",
            "best_fine_tuned_display_name": valid_fine_tuned[0]["display_name"] if valid_fine_tuned else "",
            "best_fine_tuned_driving_score_v2": valid_fine_tuned[0]["driving_score_v2"] if valid_fine_tuned else None,
        }
        for field in delta_fields:
            values = [_to_float(row.get(field)) for row in eligible_pairs]
            values = [value for value in values if value is not None]
            summary[f"avg_{field[6:]}"] = (sum(values) / len(values)) if values else None
        summary_rows.append(summary)
    return summary_rows


def _build_stage1_shortlist(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shortlist_rows: list[dict[str, Any]] = []
    for tier in VALID_TIERS:
        ranked = sorted(
            [
                record
                for record in records
                if record.get("tier") == tier and _to_bool(record.get("ranking_eligible"))
            ],
            key=_report_rank_sort_key,
        )
        for rank, record in enumerate(ranked[:2], start=1):
            shortlist_rows.append(
                {
                    "tier": tier,
                    "shortlist_rank": rank,
                    "model_id": record["model_id"],
                    "display_name": record["display_name"],
                    "ollama_tag": record["ollama_tag"],
                    "family": record["family"],
                    "variant_kind": record["variant_kind"],
                    "driving_score_v2": record["driving_score_v2"],
                    "task_completion_rate": record["task_completion_rate"],
                    "decision_latency_ms_avg_mean": record["decision_latency_ms_avg_mean"],
                    "overall_score_mean": record["overall_score_mean"],
                }
            )
    return shortlist_rows


def _build_invalid_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invalid = [record for record in records if not _to_bool(record.get("ranking_eligible"))]
    return sorted(
        invalid,
        key=lambda record: (
            record.get("study_stage") or "",
            record.get("tier") or "",
            record.get("display_name") or record.get("ollama_tag") or "",
        ),
    )


def _merge_refresh_records(
    screening_records: list[dict[str, Any]],
    refresh_records: list[dict[str, Any]],
    *,
    refresh_tier: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    def _canonical_screening_record(record: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        normalized["study_stage"] = "screening"
        return normalized

    if not refresh_records:
        return screening_records, {
            "base_records": len(screening_records),
            "refresh_records": 0,
            "refresh_candidates_in_tier": 0,
            "replaced_invalid_rows": 0,
            "preserved_valid_rows": 0,
            "selected_refresh_failures": 0,
            "added_refresh_only_rows": 0,
        }

    base_by_model = {record["model_id"]: record for record in screening_records}
    refresh_by_model = {record["model_id"]: record for record in refresh_records}
    merged_model_ids = list(dict.fromkeys([*(record["model_id"] for record in screening_records), *refresh_by_model.keys()]))

    merged_records: list[dict[str, Any]] = []
    stats = {
        "base_records": len(screening_records),
        "refresh_records": len(refresh_records),
        "refresh_candidates_in_tier": 0,
        "replaced_invalid_rows": 0,
        "preserved_valid_rows": 0,
        "selected_refresh_failures": 0,
        "added_refresh_only_rows": 0,
    }

    for model_id in merged_model_ids:
        base_record = base_by_model.get(model_id)
        refresh_record = refresh_by_model.get(model_id)

        if refresh_record and refresh_record.get("tier") == refresh_tier:
            stats["refresh_candidates_in_tier"] += 1

        if refresh_record is None or refresh_record.get("tier") != refresh_tier:
            if base_record is not None:
                merged_records.append(base_record)
            continue

        if base_record is None:
            merged_records.append(_canonical_screening_record(refresh_record))
            stats["added_refresh_only_rows"] += 1
            if not _to_bool(refresh_record.get("ranking_eligible")):
                stats["selected_refresh_failures"] += 1
            continue

        if _to_bool(base_record.get("ranking_eligible")):
            merged_records.append(base_record)
            stats["preserved_valid_rows"] += 1
            continue

        merged_records.append(_canonical_screening_record(refresh_record))
        if _to_bool(refresh_record.get("ranking_eligible")):
            stats["replaced_invalid_rows"] += 1
        else:
            stats["selected_refresh_failures"] += 1

    return merged_records, stats


def _build_acceptance_gate(
    screening_records: list[dict[str, Any]],
    paired_deltas: list[dict[str, Any]],
    *,
    target_tier: str,
) -> dict[str, Any]:
    tier_records = [record for record in screening_records if record.get("tier") == target_tier]
    valid_records = [record for record in tier_records if _to_bool(record.get("ranking_eligible"))]
    invalid_records = [record for record in tier_records if not _to_bool(record.get("ranking_eligible"))]
    eligible_pairs = [row for row in paired_deltas if row.get("tier") == target_tier and _to_bool(row.get("pair_eligible"))]

    represented_families = sorted({record["family"] for record in tier_records})
    eligible_pair_families = sorted({row["family"] for row in eligible_pairs})
    missing_pair_families = [family for family in represented_families if family not in set(eligible_pair_families)]
    invalid_fraction = (len(invalid_records) / len(tier_records)) if tier_records else 1.0

    checks = [
        {
            "name": "valid_records_at_least_10",
            "passed": len(valid_records) >= 10,
            "actual": len(valid_records),
            "required": 10,
        },
        {
            "name": "eligible_pairs_at_least_4",
            "passed": len(eligible_pairs) >= 4,
            "actual": len(eligible_pairs),
            "required": 4,
        },
        {
            "name": "family_pair_coverage",
            "passed": not missing_pair_families,
            "actual": eligible_pair_families,
            "required": represented_families,
        },
        {
            "name": "invalid_fraction_at_most_0_25",
            "passed": invalid_fraction <= 0.25,
            "actual": invalid_fraction,
            "required": 0.25,
        },
    ]
    comparison_quality = all(check["passed"] for check in checks)

    return {
        "target_tier": target_tier,
        "study_quality": "comparison-quality" if comparison_quality else "screening-quality",
        "comparison_quality_passed": comparison_quality,
        "valid_record_count": len(valid_records),
        "total_record_count": len(tier_records),
        "invalid_record_count": len(invalid_records),
        "eligible_pair_count": len(eligible_pairs),
        "represented_families": represented_families,
        "eligible_pair_families": eligible_pair_families,
        "missing_pair_families": missing_pair_families,
        "invalid_fraction": invalid_fraction,
        "checks": checks,
    }


def _plot_placeholder(output_path: Path, title: str, message: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_tier_leaderboard(output_path: Path, tier: str, leaderboard_rows: list[dict[str, Any]]) -> None:
    ranked_rows = [row for row in leaderboard_rows if row.get("rank")]
    if not ranked_rows:
        _plot_placeholder(output_path, f"{tier.title()} Leaderboard", "No ranking-eligible models in this tier.")
        return
    labels = [row["display_name"] for row in ranked_rows]
    values = [_to_float(row["driving_score_v2"]) or 0.0 for row in ranked_rows]
    fig_h = max(4.5, 0.6 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(labels, values, color="#2a9d8f")
    ax.invert_yaxis()
    ax.set_xlabel("Driving Score v2")
    ax.set_title(f"{tier.title()} Tier Leaderboard")
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:.3f}", va="center", ha="left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_deltas(output_path: Path, paired_rows: list[dict[str, Any]]) -> None:
    eligible = [row for row in paired_rows if _to_bool(row.get("pair_eligible"))]
    if not eligible:
        _plot_placeholder(output_path, "Fine-Tuned vs Base Deltas", "No eligible exact base-vs-fine-tuned pairs.")
        return
    labels = [row["fine_tuned_display_name"] for row in eligible]
    ds_values = [_to_float(row.get("delta_driving_score_v2")) or 0.0 for row in eligible]
    latency_values = [_to_float(row.get("delta_decision_latency_ms_avg_mean")) or 0.0 for row in eligible]
    fig_h = max(5.0, 0.7 * len(labels) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_h))
    axes[0].barh(labels, ds_values, color="#264653")
    axes[0].invert_yaxis()
    axes[0].set_title("Driving Score Delta")
    axes[0].set_xlabel("Fine-tuned - Base")
    axes[1].barh(labels, latency_values, color="#e76f51")
    axes[1].invert_yaxis()
    axes[1].set_title("Latency Delta (ms)")
    axes[1].set_xlabel("Fine-tuned - Base")
    fig.suptitle("Exact Fine-Tuned vs Base Deltas")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_pareto(output_path: Path, tier: str, records: list[dict[str, Any]]) -> None:
    eligible = [
        record
        for record in records
        if record.get("tier") == tier and _to_bool(record.get("ranking_eligible"))
    ]
    if not eligible:
        _plot_placeholder(output_path, f"{tier.title()} Pareto", "No ranking-eligible models with latency metrics.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [_to_float(record.get("decision_latency_ms_avg_mean")) or 0.0 for record in eligible]
    ys = [_to_float(record.get("driving_score_v2")) or 0.0 for record in eligible]
    ax.scatter(xs, ys, color="#3a86ff")
    for record, x_pos, y_pos in zip(eligible, xs, ys):
        ax.annotate(record["display_name"], (x_pos, y_pos), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Average Decision Latency (ms)")
    ax.set_ylabel("Driving Score v2")
    ax.set_title(f"{tier.title()} Tier Pareto: Driving Score vs Latency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_finalist_energy_pareto(
    output_path: Path, finalist_records: list[dict[str, Any]], shortlisted_model_ids: set[str]
) -> None:
    eligible = [
        record
        for record in finalist_records
        if record["model_id"] in shortlisted_model_ids
        and _to_bool(record.get("ranking_eligible"))
        and _to_float(record.get("net_energy_j_mean")) is not None
    ]
    if not eligible:
        _plot_placeholder(
            output_path,
            "Finalist Energy Pareto",
            "No finalist Joulescope-style energy records were provided for shortlisted models.",
        )
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [_to_float(record.get("net_energy_j_mean")) or 0.0 for record in eligible]
    ys = [_to_float(record.get("driving_score_v2")) or 0.0 for record in eligible]
    ax.scatter(xs, ys, color="#ff006e")
    for record, x_pos, y_pos in zip(eligible, xs, ys):
        ax.annotate(record["display_name"], (x_pos, y_pos), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Net Energy (J)")
    ax.set_ylabel("Driving Score v2")
    ax.set_title("Finalist Pareto: Driving Score vs Net Energy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_study_report(
    output_path: Path,
    *,
    study_id: str,
    screening_records: list[dict[str, Any]],
    finalist_records: list[dict[str, Any]],
    shortlist_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    acceptance_gate: dict[str, Any],
    refresh_summary: dict[str, Any] | None,
) -> None:
    valid_by_tier = {
        tier: sum(
            1
            for record in screening_records
            if record.get("tier") == tier and _to_bool(record.get("ranking_eligible"))
        )
        for tier in VALID_TIERS
    }
    shortlist_by_tier: dict[str, list[str]] = defaultdict(list)
    for row in shortlist_rows:
        shortlist_by_tier[row["tier"]].append(row["display_name"])
    target_tier = acceptance_gate["target_tier"]
    target_valid = sorted(
        [
            record
            for record in screening_records
            if record.get("tier") == target_tier and _to_bool(record.get("ranking_eligible"))
        ],
        key=_report_rank_sort_key,
    )
    incomplete_families = acceptance_gate.get("missing_pair_families") or []
    remaining_invalid = [record for record in invalid_rows if record.get("tier") == target_tier]

    lines = [
        f"# SLM Study Report: {study_id}",
        "",
        "## Scope",
        "- Benchmark family: LaMPilot (`lampilot_highway_v1`)",
        "- Headline task metric: `driving_score_v2`",
        "- Efficiency metrics are co-reported and not folded into the task score.",
        "",
        "## Record Counts",
        f"- Screening records: {len(screening_records)}",
        f"- Finalist energy records: {len(finalist_records)}",
        f"- Invalid or non-rankable runs: {len(invalid_rows)}",
        f"- Issues captured: {len(issues)}",
        "",
        "## Study Quality",
        f"- Target tier: {target_tier}",
        f"- Classification: `{acceptance_gate['study_quality']}`",
        f"- Comparison-quality gate passed: `{str(acceptance_gate['comparison_quality_passed']).lower()}`",
        "",
        "## Acceptance Gate",
    ]
    for check in acceptance_gate["checks"]:
        lines.append(
            f"- {check['name']}: {'pass' if check['passed'] else 'fail'} "
            f"(actual={check['actual']}, required={check['required']})"
        )
    lines.extend(
        [
            "",
        "## Valid Screening Models by Tier",
        ]
    )
    for tier in VALID_TIERS:
        lines.append(f"- {tier}: {valid_by_tier[tier]}")
    if refresh_summary:
        lines.extend(
            [
                "",
                "## Refresh Merge",
                f"- Refresh tier: {refresh_summary['refresh_tier']}",
                f"- Base screening records: {refresh_summary['base_records']}",
                f"- Refresh records: {refresh_summary['refresh_records']}",
                f"- Refresh candidates in tier: {refresh_summary['refresh_candidates_in_tier']}",
                f"- Replaced invalid rows with valid reruns: {refresh_summary['replaced_invalid_rows']}",
                f"- Preserved existing valid rows: {refresh_summary['preserved_valid_rows']}",
                f"- Selected rerun rows that still failed: {refresh_summary['selected_refresh_failures']}",
                f"- Added refresh-only rows: {refresh_summary['added_refresh_only_rows']}",
            ]
        )
    lines.extend(["", "## Valid Ranking Conclusions"])
    if target_valid:
        for index, record in enumerate(target_valid[:3], start=1):
            lines.append(
                f"- Top {index}: {record['display_name']} | "
                f"driving_score_v2={record['driving_score_v2']:.4f} | "
                f"task_completion_rate={record['task_completion_rate']:.4f}"
            )
    else:
        lines.append(f"- No ranking-eligible models in {target_tier}.")
    lines.extend(["", "## Incomplete Family Conclusions"])
    if incomplete_families:
        for family in incomplete_families:
            lines.append(f"- {family}: no eligible exact base-vs-fine-tuned pair yet")
    else:
        lines.append("- All represented families have at least one eligible exact pair.")
    lines.extend(["", "## Stage-1 Shortlist"])
    for tier in VALID_TIERS:
        winners = shortlist_by_tier.get(tier) or ["None"]
        lines.append(f"- {tier}: {', '.join(winners)}")
    lines.extend(["", "## Remaining Invalid Models"])
    if remaining_invalid:
        for record in remaining_invalid:
            reason = record.get("benchmark_result_invalid_reason") or "unknown"
            lines.append(f"- {record['display_name']}: {reason}")
    else:
        lines.append(f"- No remaining invalid models in {target_tier}.")
    if issues:
        lines.extend(["", "## Issue Codes"])
        codes = defaultdict(int)
        for issue in issues:
            codes[issue.get("code", "unknown")] += 1
        for code, count in sorted(codes.items()):
            lines.append(f"- {code}: {count}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(
    *,
    registry_path: str | Path,
    compare_report_paths: list[str | Path] | None = None,
    experiment_root_paths: list[str | Path] | None = None,
    refresh_compare_report_paths: list[str | Path] | None = None,
    finalist_energy_report_paths: list[str | Path] | None = None,
    output_root: str | Path = "analysis/out",
    study_id: str,
    refresh_tier: str = "lightweight",
    acceptance_tier: str = "lightweight",
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    screening_paths = _dedupe_paths(compare_report_paths or []) + _discover_compare_reports(experiment_root_paths)
    screening_paths = _dedupe_paths(screening_paths)
    if not screening_paths:
        raise ValueError("No compare reports were provided. Use --compare-report or --experiment-root.")
    refresh_paths = _dedupe_paths(refresh_compare_report_paths or [])
    finalist_paths = _dedupe_paths(finalist_energy_report_paths or [])

    issues: list[dict[str, Any]] = []
    base_screening_records = _normalize_records(screening_paths, registry, study_stage="screening", issues=issues)
    refresh_records = _normalize_records(refresh_paths, registry, study_stage="screening_rerun", issues=issues)
    screening_records, refresh_summary = _merge_refresh_records(
        base_screening_records,
        refresh_records,
        refresh_tier=refresh_tier,
    )
    refresh_summary["refresh_tier"] = refresh_tier
    finalist_records = _normalize_records(finalist_paths, registry, study_stage="finalist_energy", issues=issues)
    all_records = screening_records + finalist_records
    _validate_unique_model_stage(all_records)

    study_dir = Path(output_root).expanduser().resolve() / study_id
    plots_dir = study_dir / "plots"
    study_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for tier in VALID_TIERS:
        leaderboard_rows = _build_leaderboard_rows(screening_records, tier)
        _write_csv(study_dir / f"tier_leaderboard_{tier}.csv", LEADERBOARD_FIELDS, leaderboard_rows)
        _plot_tier_leaderboard(plots_dir / f"tier_leaderboard_{tier}.png", tier, leaderboard_rows)
        _plot_latency_pareto(plots_dir / f"pareto_{tier}_latency.png", tier, screening_records)

    paired_deltas = _build_paired_deltas(screening_records, issues)
    family_summary = _build_family_summary(screening_records, paired_deltas)
    shortlist_rows = _build_stage1_shortlist(screening_records)
    invalid_rows = _build_invalid_runs(screening_records)
    acceptance_gate = _build_acceptance_gate(
        screening_records,
        paired_deltas,
        target_tier=acceptance_tier,
    )

    _write_csv(study_dir / "normalized_records.csv", NORMALIZED_FIELDS, all_records)
    _write_csv(study_dir / "paired_deltas.csv", PAIRED_DELTA_FIELDS, paired_deltas)
    _write_csv(study_dir / "family_summary.csv", FAMILY_SUMMARY_FIELDS, family_summary)
    _write_csv(study_dir / "stage1_shortlist.csv", SHORTLIST_FIELDS, shortlist_rows)
    _write_csv(study_dir / "invalid_runs.csv", INVALID_RUN_FIELDS, invalid_rows)

    shortlisted_model_ids = {row["model_id"] for row in shortlist_rows}
    _plot_paired_deltas(plots_dir / "paired_deltas.png", paired_deltas)
    _plot_finalist_energy_pareto(plots_dir / "finalist_energy_pareto.png", finalist_records, shortlisted_model_ids)

    issue_summary = {
        "study_id": study_id,
        "issue_count": len(issues),
        "issues": issues,
    }
    _write_json(study_dir / "issues.json", issue_summary)
    _write_study_report(
        study_dir / "study_report.md",
        study_id=study_id,
        screening_records=screening_records,
        finalist_records=finalist_records,
        shortlist_rows=shortlist_rows,
        invalid_rows=invalid_rows,
        issues=issues,
        acceptance_gate=acceptance_gate,
        refresh_summary=refresh_summary if refresh_paths else None,
    )

    return {
        "study_id": study_id,
        "study_dir": str(study_dir),
        "screening_record_count": len(screening_records),
        "finalist_record_count": len(finalist_records),
        "issue_count": len(issues),
        "shortlist_count": len(shortlist_rows),
        "study_quality": acceptance_gate["study_quality"],
        "comparison_quality_passed": acceptance_gate["comparison_quality_passed"],
        "refresh_records": len(refresh_records),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc tiered SLM study pipeline for LaMPilot compare reports.")
    parser.add_argument("--registry", required=True, help="CSV registry describing model tier and lineage metadata.")
    parser.add_argument(
        "--compare-report",
        action="append",
        default=[],
        help="Path to a LaMPilot compare JSON report. May be supplied multiple times.",
    )
    parser.add_argument(
        "--experiment-root",
        action="append",
        default=[],
        help="Experiment root containing compare/*.json reports. May be supplied multiple times.",
    )
    parser.add_argument(
        "--refresh-compare-report",
        action="append",
        default=[],
        help="Optional rerun compare JSON used to supersede currently invalid screening rows.",
    )
    parser.add_argument(
        "--finalist-energy-report",
        action="append",
        default=[],
        help="Optional finalist-stage compare JSON with hardware energy metrics. May be supplied multiple times.",
    )
    parser.add_argument("--output-root", default="analysis/out", help="Directory where study outputs will be written.")
    parser.add_argument("--study-id", required=True, help="Unique identifier for the study output folder.")
    parser.add_argument(
        "--refresh-tier",
        default="lightweight",
        choices=VALID_TIERS,
        help="Tier whose invalid screening rows may be superseded by refresh compare reports.",
    )
    parser.add_argument(
        "--acceptance-tier",
        default="lightweight",
        choices=VALID_TIERS,
        help="Tier used for the study quality acceptance gate.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = run_study(
        registry_path=args.registry,
        compare_report_paths=args.compare_report,
        experiment_root_paths=args.experiment_root,
        refresh_compare_report_paths=args.refresh_compare_report,
        finalist_energy_report_paths=args.finalist_energy_report,
        output_root=args.output_root,
        study_id=args.study_id,
        refresh_tier=args.refresh_tier,
        acceptance_tier=args.acceptance_tier,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
