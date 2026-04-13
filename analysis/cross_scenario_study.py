from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.slm_study import load_registry
else:
    from analysis.slm_study import load_registry


NORMALIZED_FIELDS = (
    "scenario_name",
    "source_compare_report",
    "experiment_id",
    "benchmark_case_set",
    "headline_task_metric",
    "headline_task_value",
    "model_id",
    "display_name",
    "ollama_tag",
    "family",
    "variant_kind",
    "base_model_id",
    "pair_group_id",
    "param_count_b",
    "tier",
    "crash_rate",
    "task_completion_rate",
    "ttc_score_mean",
    "time_efficiency_score_mean",
    "decision_latency_ms_avg_mean",
    "net_energy_j_mean",
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
    "scenario_name",
    "headline_task_metric",
    "headline_task_value",
    "model_id",
    "display_name",
    "ollama_tag",
    "family",
    "variant_kind",
    "tier",
    "task_completion_rate",
    "crash_rate",
    "ttc_score_mean",
    "time_efficiency_score_mean",
    "decision_latency_ms_avg_mean",
    "net_energy_j_mean",
    "status",
    "benchmark_result_invalid_reason",
)
ROBUSTNESS_FIELDS = (
    "model_id",
    "display_name",
    "ollama_tag",
    "family",
    "variant_kind",
    "tier",
    "reported_scenario_count",
    "valid_scenario_count",
    "mean_rank_across_valid_scenarios",
    "highway_status",
    "highway_rank",
    "merge_status",
    "merge_rank",
    "intersection_status",
    "intersection_rank",
)
PAIRED_DELTA_FIELDS = (
    "scenario_name",
    "family",
    "tier",
    "headline_task_metric",
    "base_model_id",
    "base_display_name",
    "base_ollama_tag",
    "fine_tuned_model_id",
    "fine_tuned_display_name",
    "fine_tuned_ollama_tag",
    "pair_eligible",
    "pair_issue",
    "delta_headline_task_value",
    "delta_task_completion_rate",
    "delta_crash_rate",
    "delta_decision_latency_ms_avg_mean",
)
INVALID_RUN_FIELDS = (
    "scenario_name",
    "model_id",
    "display_name",
    "ollama_tag",
    "status",
    "headline_task_metric",
    "headline_task_value",
    "benchmark_result_invalid_reason",
)
SUPPORTED_SCENARIOS = ("highway", "merge", "intersection")


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
            writer.writerow({field: _csv_value(row.get(field)) for field in writer.fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _fallback_registry_row(ollama_tag: str) -> dict[str, Any]:
    return {
        "model_id": ollama_tag.replace(":", "_").replace("-", "_").replace(".", "_"),
        "display_name": ollama_tag,
        "ollama_tag": ollama_tag,
        "family": ollama_tag.split(":")[0].split("-")[0],
        "variant_kind": "unknown",
        "base_model_id": "",
        "pair_group_id": "",
        "param_count_b": None,
        "tier": "",
        "notes": "registry_missing",
    }


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
        and _to_float(record.get("headline_task_value")) is not None
    )


def _sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    latency = _to_float(record.get("decision_latency_ms_avg_mean"))
    return (
        -(_to_float(record.get("headline_task_value")) or float("-inf")),
        -(_to_float(record.get("task_completion_rate")) or float("-inf")),
        latency if latency is not None else float("inf"),
        record.get("display_name") or record.get("model_id") or "",
    )


def _parse_scenario_reports(values: Iterable[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        scenario_name, sep, path_text = value.partition("=")
        if sep != "=" or not scenario_name.strip() or not path_text.strip():
            raise ValueError(f"Invalid --scenario-report value: {value!r}. Expected <scenario>=<path>.")
        scenario = scenario_name.strip().lower()
        if scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(f"Unsupported scenario {scenario!r}. Supported: {', '.join(SUPPORTED_SCENARIOS)}")
        if scenario in seen:
            raise ValueError(f"Duplicate scenario report for {scenario!r}")
        path = Path(path_text.strip()).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Scenario report does not exist: {path}")
        seen.add(scenario)
        parsed.append((scenario, path))
    if not parsed:
        raise ValueError("At least one --scenario-report is required.")
    return parsed


def _normalize_record(
    *,
    scenario_name: str,
    payload: dict[str, Any],
    aggregate: dict[str, Any],
    report_path: Path,
    registry: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    ollama_tag = str(aggregate.get("model") or "").strip()
    if not ollama_tag:
        raise ValueError(f"{report_path} contains an aggregate row without model")

    registry_row = registry["by_tag"].get(ollama_tag)
    if registry_row is None:
        registry_row = _fallback_registry_row(ollama_tag)
        issues.append(
            {
                "severity": "warning",
                "code": "missing_registry_row",
                "scenario_name": scenario_name,
                "model": ollama_tag,
                "report": str(report_path),
            }
        )

    headline_task_metric = str(payload.get("headline_task_metric") or "").strip() or (
        "driving_score_v2" if aggregate.get("driving_score_v2") is not None else "driving_score"
    )
    record = {
        "scenario_name": scenario_name,
        "source_compare_report": str(report_path),
        "experiment_id": payload.get("experiment_id") or report_path.stem,
        "benchmark_case_set": payload.get("benchmark_case_set") or "",
        "headline_task_metric": headline_task_metric,
        "headline_task_value": _to_float(aggregate.get(headline_task_metric)),
        "model_id": registry_row["model_id"],
        "display_name": registry_row["display_name"],
        "ollama_tag": ollama_tag,
        "family": registry_row["family"],
        "variant_kind": registry_row["variant_kind"],
        "base_model_id": registry_row.get("base_model_id", ""),
        "pair_group_id": registry_row.get("pair_group_id") or registry_row.get("model_id") or "",
        "param_count_b": registry_row.get("param_count_b"),
        "tier": registry_row.get("tier", ""),
        "crash_rate": _to_float(aggregate.get("crash_rate")),
        "task_completion_rate": _to_float(aggregate.get("task_completion_rate")),
        "ttc_score_mean": _to_float(aggregate.get("ttc_score_mean")),
        "time_efficiency_score_mean": _to_float(aggregate.get("time_efficiency_score_mean")),
        "decision_latency_ms_avg_mean": _to_float(aggregate.get("decision_latency_ms_avg_mean")),
        "net_energy_j_mean": _to_float(aggregate.get("net_energy_j_mean")),
        "benchmark_result_valid": _to_bool(aggregate.get("benchmark_result_valid", True)),
        "benchmark_result_invalid_reason": aggregate.get("benchmark_result_invalid_reason") or "",
        "model_skipped_due_to_preflight": _to_bool(aggregate.get("model_skipped_due_to_preflight")),
        "model_quarantined_due_to_timeout_collapse": _to_bool(
            aggregate.get("model_quarantined_due_to_timeout_collapse")
        ),
        "episode_execution_complete": _to_bool(aggregate.get("episode_execution_complete", True)),
    }
    record["status"] = _status_from_record(record)
    record["ranking_eligible"] = _ranking_eligible(record)
    return record


def _load_scenario_records(
    scenario_reports: list[tuple[str, Path]], registry: dict[str, Any], issues: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scenario_name, report_path in scenario_reports:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not _to_bool(payload.get("benchmark_mode")):
            raise ValueError(f"{report_path} is not a benchmark compare report")
        for aggregate in payload.get("aggregates") or []:
            records.append(
                _normalize_record(
                    scenario_name=scenario_name,
                    payload=payload,
                    aggregate=aggregate,
                    report_path=report_path,
                    registry=registry,
                    issues=issues,
                )
            )
    return records


def _build_leaderboard_rows(records: list[dict[str, Any]], scenario_name: str) -> list[dict[str, Any]]:
    scenario_records = [record for record in records if record["scenario_name"] == scenario_name]
    valid_rows = sorted(
        [record for record in scenario_records if _to_bool(record.get("ranking_eligible"))],
        key=_sort_key,
    )
    invalid_rows = sorted(
        [record for record in scenario_records if not _to_bool(record.get("ranking_eligible"))],
        key=lambda record: (record.get("status") or "", record.get("display_name") or ""),
    )
    rows: list[dict[str, Any]] = []
    for rank, record in enumerate(valid_rows, start=1):
        rows.append(
            {
                "rank": rank,
                "scenario_name": scenario_name,
                "headline_task_metric": record["headline_task_metric"],
                "headline_task_value": record["headline_task_value"],
                "model_id": record["model_id"],
                "display_name": record["display_name"],
                "ollama_tag": record["ollama_tag"],
                "family": record["family"],
                "variant_kind": record["variant_kind"],
                "tier": record["tier"],
                "task_completion_rate": record["task_completion_rate"],
                "crash_rate": record["crash_rate"],
                "ttc_score_mean": record["ttc_score_mean"],
                "time_efficiency_score_mean": record["time_efficiency_score_mean"],
                "decision_latency_ms_avg_mean": record["decision_latency_ms_avg_mean"],
                "net_energy_j_mean": record["net_energy_j_mean"],
                "status": record["status"],
                "benchmark_result_invalid_reason": record["benchmark_result_invalid_reason"],
            }
        )
    for record in invalid_rows:
        rows.append(
            {
                "rank": "",
                "scenario_name": scenario_name,
                "headline_task_metric": record["headline_task_metric"],
                "headline_task_value": record["headline_task_value"],
                "model_id": record["model_id"],
                "display_name": record["display_name"],
                "ollama_tag": record["ollama_tag"],
                "family": record["family"],
                "variant_kind": record["variant_kind"],
                "tier": record["tier"],
                "task_completion_rate": record["task_completion_rate"],
                "crash_rate": record["crash_rate"],
                "ttc_score_mean": record["ttc_score_mean"],
                "time_efficiency_score_mean": record["time_efficiency_score_mean"],
                "decision_latency_ms_avg_mean": record["decision_latency_ms_avg_mean"],
                "net_energy_j_mean": record["net_energy_j_mean"],
                "status": record["status"],
                "benchmark_result_invalid_reason": record["benchmark_result_invalid_reason"],
            }
        )
    return rows


def _build_robustness_matrix(records: list[dict[str, Any]], scenarios: list[str]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for record in records:
        row = by_model.setdefault(
            record["model_id"],
            {
                "model_id": record["model_id"],
                "display_name": record["display_name"],
                "ollama_tag": record["ollama_tag"],
                "family": record["family"],
                "variant_kind": record["variant_kind"],
                "tier": record["tier"],
                "reported_scenario_count": 0,
                "valid_scenario_count": 0,
                "mean_rank_across_valid_scenarios": None,
                "highway_status": "",
                "highway_rank": "",
                "merge_status": "",
                "merge_rank": "",
                "intersection_status": "",
                "intersection_rank": "",
            },
        )
        row["reported_scenario_count"] += 1

    rank_maps: dict[str, dict[str, int]] = {}
    for scenario_name in scenarios:
        leaderboard_rows = _build_leaderboard_rows(records, scenario_name)
        rank_maps[scenario_name] = {
            row["model_id"]: int(row["rank"])
            for row in leaderboard_rows
            if str(row.get("rank") or "").strip()
        }

    for model_id, row in by_model.items():
        ranks: list[int] = []
        for scenario_name in scenarios:
            record = next(
                (item for item in records if item["model_id"] == model_id and item["scenario_name"] == scenario_name),
                None,
            )
            if record is None:
                continue
            row[f"{scenario_name}_status"] = record["status"]
            rank = rank_maps[scenario_name].get(model_id)
            row[f"{scenario_name}_rank"] = rank or ""
            if _to_bool(record.get("ranking_eligible")):
                row["valid_scenario_count"] += 1
            if rank is not None:
                ranks.append(rank)
        row["mean_rank_across_valid_scenarios"] = sum(ranks) / len(ranks) if ranks else None

    return sorted(
        by_model.values(),
        key=lambda row: (
            -int(row["valid_scenario_count"]),
            _to_float(row["mean_rank_across_valid_scenarios"]) or float("inf"),
            row["display_name"],
        ),
    )


def _build_paired_deltas(records: list[dict[str, Any]], issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_scenario[record["scenario_name"]][record["model_id"]] = record

    rows: list[dict[str, Any]] = []
    for scenario_name, scenario_records in by_scenario.items():
        for record in sorted(
            scenario_records.values(),
            key=lambda item: (item["family"], item["display_name"]),
        ):
            if record["variant_kind"] != "fine_tuned":
                continue
            base_model_id = record.get("base_model_id") or ""
            base_record = scenario_records.get(base_model_id)
            issue = ""
            pair_eligible = False
            if not base_model_id:
                issue = "missing_base_model_id"
            elif base_record is None:
                issue = "missing_exact_base_record"
            elif not _to_bool(base_record.get("ranking_eligible")):
                issue = "base_not_ranking_eligible"
            elif not _to_bool(record.get("ranking_eligible")):
                issue = "fine_tuned_not_ranking_eligible"
            else:
                pair_eligible = True

            row = {
                "scenario_name": scenario_name,
                "family": record["family"],
                "tier": record["tier"],
                "headline_task_metric": record["headline_task_metric"],
                "base_model_id": base_model_id,
                "base_display_name": base_record["display_name"] if base_record else "",
                "base_ollama_tag": base_record["ollama_tag"] if base_record else "",
                "fine_tuned_model_id": record["model_id"],
                "fine_tuned_display_name": record["display_name"],
                "fine_tuned_ollama_tag": record["ollama_tag"],
                "pair_eligible": pair_eligible,
                "pair_issue": issue,
                "delta_headline_task_value": (
                    (_to_float(record.get("headline_task_value")) or 0.0)
                    - (_to_float(base_record.get("headline_task_value")) or 0.0)
                    if base_record is not None and pair_eligible
                    else None
                ),
                "delta_task_completion_rate": (
                    (_to_float(record.get("task_completion_rate")) or 0.0)
                    - (_to_float(base_record.get("task_completion_rate")) or 0.0)
                    if base_record is not None and pair_eligible
                    else None
                ),
                "delta_crash_rate": (
                    (_to_float(record.get("crash_rate")) or 0.0)
                    - (_to_float(base_record.get("crash_rate")) or 0.0)
                    if base_record is not None and pair_eligible
                    else None
                ),
                "delta_decision_latency_ms_avg_mean": (
                    (_to_float(record.get("decision_latency_ms_avg_mean")) or 0.0)
                    - (_to_float(base_record.get("decision_latency_ms_avg_mean")) or 0.0)
                    if base_record is not None and pair_eligible
                    else None
                ),
            }
            if issue:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "unpaired_fine_tuned_model",
                        "scenario_name": scenario_name,
                        "model_id": record["model_id"],
                        "pair_issue": issue,
                    }
                )
            rows.append(row)
    return rows


def _build_invalid_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scenario_name": record["scenario_name"],
            "model_id": record["model_id"],
            "display_name": record["display_name"],
            "ollama_tag": record["ollama_tag"],
            "status": record["status"],
            "headline_task_metric": record["headline_task_metric"],
            "headline_task_value": record["headline_task_value"],
            "benchmark_result_invalid_reason": record["benchmark_result_invalid_reason"],
        }
        for record in records
        if not _to_bool(record.get("ranking_eligible"))
    ]


def _plot_placeholder(output_path: Path, title: str, message: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_scenario_leaderboard(output_path: Path, scenario_name: str, rows: list[dict[str, Any]]) -> None:
    ranked = [row for row in rows if str(row.get("rank") or "").strip()]
    if not ranked:
        _plot_placeholder(output_path, f"{scenario_name.title()} Leaderboard", "No ranking-eligible models.")
        return
    labels = [row["display_name"] for row in ranked]
    values = [_to_float(row["headline_task_value"]) or 0.0 for row in ranked]
    fig_h = max(4.5, 0.6 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(labels, values, color="#2a9d8f")
    ax.invert_yaxis()
    ax.set_xlabel(ranked[0]["headline_task_metric"])
    ax.set_title(f"{scenario_name.title()} Leaderboard")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_scenario_latency(output_path: Path, scenario_name: str, rows: list[dict[str, Any]]) -> None:
    ranked = [row for row in rows if str(row.get("rank") or "").strip()]
    if not ranked:
        _plot_placeholder(output_path, f"{scenario_name.title()} Latency", "No ranking-eligible models.")
        return
    labels = [row["display_name"] for row in ranked]
    values = [_to_float(row["decision_latency_ms_avg_mean"]) or 0.0 for row in ranked]
    fig_h = max(4.5, 0.6 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(labels, values, color="#457b9d")
    ax.invert_yaxis()
    ax.set_xlabel("Average Decision Latency (ms)")
    ax.set_title(f"{scenario_name.title()} Latency Companion")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_robustness_matrix(output_path: Path, robustness_rows: list[dict[str, Any]], scenarios: list[str]) -> None:
    if not robustness_rows:
        _plot_placeholder(output_path, "Cross-Scenario Robustness", "No records to summarize.")
        return
    labels = [row["display_name"] for row in robustness_rows]
    matrix: list[list[float]] = []
    for row in robustness_rows:
        matrix.append([
            1.0 if str(row.get(f"{scenario_name}_status") or "") == "valid" else 0.0
            for scenario_name in scenarios
        ])
    fig_h = max(4.5, 0.45 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(2.2 + 1.7 * len(scenarios), fig_h))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([name.title() for name in scenarios])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Cross-Scenario Validity Matrix")
    for row_idx, label in enumerate(labels):
        for col_idx, scenario_name in enumerate(scenarios):
            ax.text(col_idx, row_idx, "valid" if matrix[row_idx][col_idx] > 0 else "invalid", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Ranking eligible")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_paired_deltas(output_path: Path, scenario_name: str, rows: list[dict[str, Any]]) -> None:
    eligible = [row for row in rows if row["scenario_name"] == scenario_name and _to_bool(row.get("pair_eligible"))]
    if not eligible:
        _plot_placeholder(output_path, f"{scenario_name.title()} Pair Deltas", "No eligible exact pairs.")
        return
    labels = [row["fine_tuned_display_name"] for row in eligible]
    values = [_to_float(row["delta_headline_task_value"]) or 0.0 for row in eligible]
    fig_h = max(4.5, 0.6 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(labels, values, color="#e76f51")
    ax.invert_yaxis()
    ax.set_xlabel("Fine-tuned - Base")
    ax.set_title(f"{scenario_name.title()} Headline Delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_study_report(
    output_path: Path,
    *,
    study_id: str,
    scenarios: list[str],
    leaderboard_by_scenario: dict[str, list[dict[str, Any]]],
    robustness_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    lines = [
        f"# Cross-Scenario Study Report: {study_id}",
        "",
        "## Scope",
        f"- Scenarios: {', '.join(scenarios)}",
        "- Ranking is scenario-local and excludes invalid or quarantined runs.",
        "- Highway keeps `driving_score_v2` as the headline metric; non-highway scenarios follow their report-local headline metric.",
        "",
        "## Scenario Rankings",
    ]
    for scenario_name in scenarios:
        ranked = [row for row in leaderboard_by_scenario[scenario_name] if str(row.get('rank') or '').strip()]
        if not ranked:
            lines.append(f"- {scenario_name}: no ranking-eligible models")
            continue
        top = ranked[0]
        lines.append(
            f"- {scenario_name}: top model {top['display_name']} ({top['headline_task_metric']}={_csv_value(top['headline_task_value'])})"
        )
    lines.extend(["", "## Robustness Summary"])
    if robustness_rows:
        for row in robustness_rows[:5]:
            lines.append(
                f"- {row['display_name']}: valid_scenarios={row['valid_scenario_count']} | mean_rank={_csv_value(row['mean_rank_across_valid_scenarios'])}"
            )
    else:
        lines.append("- No robustness rows produced.")
    lines.extend(["", "## Exact Base-vs-Fine-Tuned Pairs"])
    eligible_pairs = [row for row in paired_rows if _to_bool(row.get("pair_eligible"))]
    if eligible_pairs:
        for row in eligible_pairs:
            lines.append(
                f"- {row['scenario_name']}: {row['fine_tuned_display_name']} vs {row['base_display_name']} | delta_{row['headline_task_metric']}={_csv_value(row['delta_headline_task_value'])}"
            )
    else:
        lines.append("- No eligible exact base-vs-fine-tuned pairs.")
    lines.extend(["", "## Invalid Runs"])
    if invalid_rows:
        for row in invalid_rows:
            reason = row["benchmark_result_invalid_reason"] or row["status"]
            lines.append(f"- {row['scenario_name']}: {row['display_name']} | {reason}")
    else:
        lines.append("- No invalid runs.")
    if issues:
        lines.extend(["", "## Issues"])
        issue_counts: dict[str, int] = defaultdict(int)
        for issue in issues:
            issue_counts[issue.get("code", "unknown")] += 1
        for code, count in sorted(issue_counts.items()):
            lines.append(f"- {code}: {count}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cross_scenario_study(
    *,
    registry_path: str | Path,
    scenario_reports: list[str],
    output_root: str | Path = "analysis/out",
    study_id: str,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    parsed_reports = _parse_scenario_reports(scenario_reports)
    scenarios = [scenario for scenario, _ in parsed_reports]
    issues: list[dict[str, Any]] = []
    records = _load_scenario_records(parsed_reports, registry, issues)

    study_dir = Path(output_root).expanduser().resolve() / study_id
    plots_dir = study_dir / "plots"
    study_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(study_dir / "normalized_records.csv", NORMALIZED_FIELDS, records)

    leaderboard_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for scenario_name in scenarios:
        rows = _build_leaderboard_rows(records, scenario_name)
        leaderboard_by_scenario[scenario_name] = rows
        _write_csv(study_dir / f"scenario_leaderboard_{scenario_name}.csv", LEADERBOARD_FIELDS, rows)
        _plot_scenario_leaderboard(plots_dir / f"scenario_leaderboard_{scenario_name}.png", scenario_name, rows)
        _plot_scenario_latency(plots_dir / f"scenario_latency_{scenario_name}.png", scenario_name, rows)

    robustness_rows = _build_robustness_matrix(records, scenarios)
    paired_rows = _build_paired_deltas(records, issues)
    invalid_rows = _build_invalid_runs(records)

    _write_csv(study_dir / "robustness_matrix.csv", ROBUSTNESS_FIELDS, robustness_rows)
    _write_csv(study_dir / "paired_deltas_by_scenario.csv", PAIRED_DELTA_FIELDS, paired_rows)
    _write_csv(study_dir / "invalid_runs.csv", INVALID_RUN_FIELDS, invalid_rows)
    _write_json(study_dir / "issues.json", {"issue_count": len(issues), "issues": issues})

    _plot_robustness_matrix(plots_dir / "robustness_matrix.png", robustness_rows, scenarios)
    for scenario_name in scenarios:
        _plot_paired_deltas(plots_dir / f"paired_deltas_{scenario_name}.png", scenario_name, paired_rows)

    _write_study_report(
        study_dir / "study_report.md",
        study_id=study_id,
        scenarios=scenarios,
        leaderboard_by_scenario=leaderboard_by_scenario,
        robustness_rows=robustness_rows,
        paired_rows=paired_rows,
        invalid_rows=invalid_rows,
        issues=issues,
    )

    return {
        "study_id": study_id,
        "study_dir": str(study_dir),
        "scenario_count": len(scenarios),
        "record_count": len(records),
        "invalid_run_count": len(invalid_rows),
        "issue_count": len(issues),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a cross-scenario study bundle from highway, merge, and intersection compare reports."
    )
    parser.add_argument("--registry", required=True, help="CSV model registry with tier and lineage metadata.")
    parser.add_argument(
        "--scenario-report",
        action="append",
        default=[],
        help="Scenario compare report in the form <scenario>=<path>. Repeatable.",
    )
    parser.add_argument("--output-root", default="analysis/out", help="Generated study output directory.")
    parser.add_argument("--study-id", required=True, help="Output folder name under --output-root.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = run_cross_scenario_study(
        registry_path=args.registry,
        scenario_reports=args.scenario_report,
        output_root=args.output_root,
        study_id=args.study_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
