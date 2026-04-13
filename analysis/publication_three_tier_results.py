from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


VALID_TIERS = ("lightweight", "midclass", "highclass")
VALIDITY_SUMMARY_FIELDS = (
    "tier",
    "total_records",
    "valid_records",
    "invalid_records",
    "invalid_fraction",
    "eligible_exact_pairs",
    "shortlist_count",
    "shortlist_available",
    "publication_interpretation",
)
LIGHTWEIGHT_LEADERBOARD_FIELDS = (
    "rank",
    "display_name",
    "ollama_tag",
    "family",
    "variant_kind",
    "driving_score_v2",
    "task_completion_rate",
    "crash_rate",
    "decision_latency_ms_avg_mean",
    "tokens_per_second_mean",
)
LIGHTWEIGHT_PAIR_FIELDS = (
    "family",
    "base_display_name",
    "fine_tuned_display_name",
    "delta_driving_score_v2",
    "delta_task_completion_rate",
    "delta_overall_score_mean",
    "delta_decision_latency_ms_avg_mean",
    "delta_tokens_per_second_mean",
)
MIDCLASS_SCREENING_FIELDS = (
    "rank",
    "family",
    "best_valid_model_display_name",
    "best_valid_model_id",
    "best_valid_variant_kind",
    "driving_score_v2",
    "task_completion_rate",
    "crash_rate",
    "decision_latency_ms_avg_mean",
    "exact_pair_eligible_for_family",
    "base_model_status",
    "fine_tuned_model_status",
    "pair_issue",
)
HIGHCLASS_FAILURE_FIELDS = (
    "display_name",
    "ollama_tag",
    "family",
    "variant_kind",
    "status",
    "benchmark_result_invalid_reason",
    "decision_latency_ms_avg_mean",
    "p95_decision_latency_sec_mean",
    "tokens_per_second_mean",
    "driving_score_v2",
    "task_completion_rate",
)


def _valid_rows_from_leaderboard(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in _valid_only_ranked_rows(rows) if row.get("status") == "valid"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in writer.fieldnames})


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _study_inputs(study_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "normalized": _read_csv(study_dir / "normalized_records.csv"),
        "paired": _read_csv(study_dir / "paired_deltas.csv"),
        "shortlist": _read_csv(study_dir / "stage1_shortlist.csv"),
        "leaderboards": {
            tier: _read_csv(study_dir / f"tier_leaderboard_{tier}.csv") for tier in VALID_TIERS
        },
    }


def _tier_records(normalized_rows: list[dict[str, str]], tier: str) -> list[dict[str, str]]:
    return [row for row in normalized_rows if row.get("tier") == tier]


def _count_valid_records(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if _to_bool(row.get("ranking_eligible")))


def _eligible_pairs(rows: list[dict[str, str]], tier: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("tier") == tier and _to_bool(row.get("pair_eligible"))]


def _publication_interpretation(valid_count: int, eligible_pair_count: int) -> str:
    if valid_count == 0:
        return "failure-analysis-only"
    if eligible_pair_count > 0:
        return "results-ready"
    return "screening-only"


def _validity_summary_row(study: dict[str, list[dict[str, str]]], tier: str) -> dict[str, Any]:
    tier_rows = _tier_records(study["normalized"], tier)
    total_records = len(tier_rows)
    valid_records = _count_valid_records(tier_rows)
    invalid_records = total_records - valid_records
    eligible_pairs = len(_eligible_pairs(study["paired"], tier))
    shortlist_count = sum(1 for row in study["shortlist"] if row.get("tier") == tier)
    invalid_fraction = (invalid_records / total_records) if total_records else 0.0
    return {
        "tier": tier,
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "invalid_fraction": invalid_fraction,
        "eligible_exact_pairs": eligible_pairs,
        "shortlist_count": shortlist_count,
        "shortlist_available": shortlist_count > 0,
        "publication_interpretation": _publication_interpretation(valid_records, eligible_pairs),
    }


def _valid_only_ranked_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ranked = [row for row in rows if row.get("rank")]
    return sorted(ranked, key=lambda row: _to_int(row.get("rank")) or 10**9)


def _build_lightweight_leaderboard(study: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows = _valid_only_ranked_rows(study["leaderboards"]["lightweight"])
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append({field: row.get(field, "") for field in LIGHTWEIGHT_LEADERBOARD_FIELDS})
    return output


def _build_lightweight_pairs(study: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows = _eligible_pairs(study["paired"], "lightweight")
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append({field: row.get(field, "") for field in LIGHTWEIGHT_PAIR_FIELDS})
    return output


def _build_midclass_summary(study: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    ranked_rows = _valid_only_ranked_rows(study["leaderboards"]["midclass"])
    pair_rows = [row for row in study["paired"] if row.get("tier") == "midclass"]
    normalized = _tier_records(study["normalized"], "midclass")

    pair_by_family = {row.get("family", ""): row for row in pair_rows}
    status_by_family_variant: dict[tuple[str, str], str] = {}
    for row in normalized:
        status_by_family_variant[(row.get("family", ""), row.get("variant_kind", ""))] = row.get("status", "")

    output: list[dict[str, Any]] = []
    families = []
    for row in ranked_rows:
        family = row.get("family", "")
        if family not in families:
            families.append(family)
    for row in pair_rows:
        family = row.get("family", "")
        if family not in families:
            families.append(family)

    best_by_family = {row.get("family", ""): row for row in ranked_rows}
    for family in families:
        best = best_by_family.get(family, {})
        pair = pair_by_family.get(family, {})
        output.append(
            {
                "rank": best.get("rank", ""),
                "family": family,
                "best_valid_model_display_name": best.get("display_name", ""),
                "best_valid_model_id": best.get("model_id", ""),
                "best_valid_variant_kind": best.get("variant_kind", ""),
                "driving_score_v2": best.get("driving_score_v2", ""),
                "task_completion_rate": best.get("task_completion_rate", ""),
                "crash_rate": best.get("crash_rate", ""),
                "decision_latency_ms_avg_mean": best.get("decision_latency_ms_avg_mean", ""),
                "exact_pair_eligible_for_family": _to_bool(pair.get("pair_eligible")),
                "base_model_status": status_by_family_variant.get((family, "base"), ""),
                "fine_tuned_model_status": status_by_family_variant.get((family, "fine_tuned"), ""),
                "pair_issue": pair.get("pair_issue", ""),
            }
        )
    return output


def _build_highclass_failure_summary(study: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows = [row for row in study["leaderboards"]["highclass"] if not row.get("rank")]
    rows.sort(key=lambda row: (row.get("family", ""), row.get("display_name", "")))
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append({field: row.get(field, "") for field in HIGHCLASS_FAILURE_FIELDS})
    return output


def _select_primary_compare_report(normalized_rows: list[dict[str, str]], tier: str) -> Path | None:
    tier_rows = _tier_records(normalized_rows, tier)
    paths = [row.get("source_compare_report", "") for row in tier_rows if row.get("source_compare_report")]
    if not paths:
        return None
    dominant, _ = Counter(paths).most_common(1)[0]
    path = Path(dominant)
    return path if path.exists() else None


def _copy_if_exists(source_path: Path | None, destination_path: Path) -> bool:
    if source_path is None or not source_path.exists():
        return False
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return True


def _compare_plot_path(compare_report: Path | None, suffix: str) -> Path | None:
    if compare_report is None:
        return None
    return compare_report.with_name(f"{compare_report.stem}{suffix}")


def _study_plot_path(study_dir: Path, filename: str) -> Path | None:
    candidate = study_dir / "plots" / filename
    return candidate if candidate.exists() else None


def _build_cross_tier_validity_figure(path: Path, validity_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiers = [row["tier"] for row in validity_rows]
    valid = [int(row["valid_records"]) for row in validity_rows]
    total = [int(row["total_records"]) for row in validity_rows]
    invalid = [max(t - v, 0) for t, v in zip(total, valid)]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    positions = range(len(tiers))
    ax.bar(positions, valid, label="Valid", color="#3A7D44")
    ax.bar(positions, invalid, bottom=valid, label="Invalid / non-rankable", color="#C44536")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([tier.title() for tier in tiers])
    ax.set_ylabel("Model count")
    ax.set_title("Cross-tier ranking eligibility")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for idx, row in enumerate(validity_rows):
        ax.text(idx, total[idx] + 0.05, f"{valid[idx]}/{total[idx]}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _build_cross_tier_pareto_figure(path: Path, tier_rows: dict[str, list[dict[str, str]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {
        "lightweight": "#2E6F95",
        "midclass": "#C44536",
        "highclass": "#6C757D",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    any_points = False
    for tier in VALID_TIERS:
        rows = _valid_rows_from_leaderboard(tier_rows.get(tier, []))
        if not rows:
            continue
        x_values = [_to_float(row.get("decision_latency_ms_avg_mean")) or 0.0 for row in rows]
        y_values = [_to_float(row.get("driving_score_v2")) or 0.0 for row in rows]
        ax.scatter(
            x_values,
            y_values,
            s=72,
            alpha=0.9,
            color=palette[tier],
            edgecolors="white",
            linewidths=0.8,
            label=tier.title(),
        )
        for row, x_value, y_value in zip(rows, x_values, y_values):
            ax.annotate(
                row.get("display_name", row.get("model_id", "")),
                (x_value, y_value),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                alpha=0.9,
            )
        any_points = True
    ax.set_xlabel("Decision latency mean (ms)")
    ax.set_ylabel("Driving score v2")
    ax.set_title("Cross-tier Pareto view: task quality vs latency")
    ax.grid(alpha=0.25)
    if any_points:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No ranking-eligible models available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _format_float(value: Any, digits: int = 4) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.{digits}f}"


def _build_evidence_summary(
    validity_rows: list[dict[str, Any]],
    lightweight_leaderboard: list[dict[str, Any]],
    lightweight_pairs: list[dict[str, Any]],
    midclass_summary: list[dict[str, Any]],
    highclass_failures: list[dict[str, Any]],
) -> str:
    lightweight = validity_rows[0]
    midclass = validity_rows[1]
    highclass = validity_rows[2]
    top_lightweight = lightweight_leaderboard[0] if lightweight_leaderboard else {}
    top_midclass = midclass_summary[0] if midclass_summary else {}
    phi_pair = lightweight_pairs[0] if lightweight_pairs else {}

    lines = [
        "# Evidence Summary",
        "",
        "## Established Findings",
        (
            f"- The lightweight tier is the strongest current evidence base: "
            f"{lightweight['valid_records']} of {lightweight['total_records']} models were ranking-eligible."
        ),
        (
            f"- The best valid lightweight model was {top_lightweight.get('display_name', 'n/a')} "
            f"with `driving_score_v2={_format_float(top_lightweight.get('driving_score_v2'))}`, "
            f"`task_completion_rate={_format_float(top_lightweight.get('task_completion_rate'))}`, and "
            f"`decision_latency_ms_avg_mean={_format_float(top_lightweight.get('decision_latency_ms_avg_mean'), 1)}`."
        ),
    ]
    if phi_pair:
        lines.extend(
            [
                (
                    f"- The only exact ranking-eligible base-vs-fine-tuned comparison is the Phi pair "
                    f"({phi_pair.get('base_display_name')} vs {phi_pair.get('fine_tuned_display_name')}). "
                    f"The fine-tuned model improved `driving_score_v2` by "
                    f"{_format_float(phi_pair.get('delta_driving_score_v2'))}, improved "
                    f"`overall_score_mean` by {_format_float(phi_pair.get('delta_overall_score_mean'))}, and reduced "
                    f"mean decision latency by {_format_float(phi_pair.get('delta_decision_latency_ms_avg_mean'), 1)} ms."
                )
            ]
        )
    lines.extend(
        [
            "",
            "## Provisional Findings",
            (
                f"- The midclass tier remains screening-only: {midclass['valid_records']} of "
                f"{midclass['total_records']} models were ranking-eligible, but no exact base-vs-fine-tuned pair is eligible."
            ),
            (
                f"- The best valid midclass model was {top_midclass.get('best_valid_model_display_name', 'n/a')} "
                f"with `driving_score_v2={_format_float(top_midclass.get('driving_score_v2'))}`. "
                "This supports provisional screening conclusions about feasible models, but not family-level fine-tuning claims."
            ),
            "",
            "## Blocked Findings",
            (
                f"- The highclass tier currently supports a runtime-scalability finding rather than a competitive model comparison: "
                f"{highclass['valid_records']} of {highclass['total_records']} runs were ranking-eligible."
            ),
        ]
    )
    if highclass_failures:
        fastest_failure = min(
            highclass_failures,
            key=lambda row: _to_float(row.get("decision_latency_ms_avg_mean")) or float("inf"),
        )
        lines.append(
            (
                f"- The least latent highclass failure was {fastest_failure.get('display_name')} at "
                f"`decision_latency_ms_avg_mean={_format_float(fastest_failure.get('decision_latency_ms_avg_mean'), 1)}` ms, "
                "but it still failed ranking eligibility due to timeout-collapse or incomplete execution."
            )
        )
    lines.extend(
        [
            "",
            "## Publication Guidance",
            "- Center the main Results section on lightweight comparative evidence.",
            "- Present midclass as partial screening evidence with blocked exact-pair conclusions.",
            "- Present highclass as a timeout-collapse and local-runtime scalability result, not as a task-performance leaderboard.",
            "",
            "## Core Message",
            "- Under the current local runtime policy, practical benchmark viability decreases with model tier. Lightweight models provide the most usable comparative evidence, while fine-tuning appears promising but remains underdetermined because exact pair coverage is still sparse.",
        ]
    )
    return "\n".join(lines)


def _build_results_outline(
    validity_rows: list[dict[str, Any]],
    lightweight_leaderboard: list[dict[str, Any]],
    lightweight_pairs: list[dict[str, Any]],
    midclass_summary: list[dict[str, Any]],
    highclass_failures: list[dict[str, Any]],
) -> str:
    top_lightweight = lightweight_leaderboard[0] if lightweight_leaderboard else {}
    top_midclass = midclass_summary[0] if midclass_summary else {}
    phi_pair = lightweight_pairs[0] if lightweight_pairs else {}
    lines = [
        "# Results Outline",
        "",
        "## 1. Three-Tier Screening Overview",
        "- Open with the cross-tier validity message rather than raw performance.",
        (
            f"- Report ranking eligibility as lightweight {validity_rows[0]['valid_records']}/{validity_rows[0]['total_records']}, "
            f"midclass {validity_rows[1]['valid_records']}/{validity_rows[1]['total_records']}, "
            f"highclass {validity_rows[2]['valid_records']}/{validity_rows[2]['total_records']}."
        ),
        "- State that later comparative claims are restricted to ranking-eligible rows only.",
        "",
        "## 2. Lightweight Comparative Results",
        (
            f"- Lead with {top_lightweight.get('display_name', 'n/a')} as the top lightweight model "
            f"(`driving_score_v2={_format_float(top_lightweight.get('driving_score_v2'))}`)."
        ),
        "- Discuss task quality and latency jointly, but do not combine them into a composite score.",
        "- Use the lightweight task-first compare plot and efficiency companion plot as the main figures.",
        "- Use Pareto plots to show the latency-quality frontier across and within tiers without collapsing the metrics into one score.",
        "",
        "## 3. Lightweight Exact Pair Evidence",
        "- Limit fine-tuning claims to exact eligible pairs only.",
    ]
    if phi_pair:
        lines.append(
            (
                f"- The Phi pair is the only eligible pair in v1, with "
                f"`delta_driving_score_v2={_format_float(phi_pair.get('delta_driving_score_v2'))}` and "
                f"`delta_decision_latency_ms_avg_mean={_format_float(phi_pair.get('delta_decision_latency_ms_avg_mean'), 1)}` ms."
            )
        )
    lines.extend(
        [
            "- Explicitly state that other lightweight families remain blocked by invalid or quarantined runs.",
            "",
            "## 4. Midclass Screening Results",
            (
                f"- Report {top_midclass.get('best_valid_model_display_name', 'n/a')} as the best valid midclass model, "
                "but frame the tier as provisional because no exact pair is eligible."
            ),
            "- Use the tier table to show which side of each family pair failed ranking eligibility.",
            "",
            "## 5. Highclass Failure Analysis",
            "- Present highclass as evidence about runtime scalability under the present timeout policy.",
            "- State clearly that no 14B run was ranking-eligible and that no performance ordering is claimed.",
            "- Use the failure summary table to document timeout-collapse and latency burden.",
            "",
            "## 6. Synthesis",
            "- Conclude that practical benchmark viability decreases with model tier under the current local runtime setup.",
            "- State that lightweight is the only tier ready for substantive comparative discussion in the current manuscript.",
        ]
    )
    return "\n".join(lines)


def _build_caption_bank(include_validity_figure: bool) -> str:
    lines = [
        "# Caption Bank",
        "",
        "## Tables",
        "- Table 1. Cross-tier screening status across lightweight, midclass, and highclass models. Counts include both ranking-eligible and invalid runs; later comparative claims use valid rows only.",
        "- Table 2. Valid-only ranking for lightweight models under the LaMPilot highway benchmark, ordered by `driving_score_v2`.",
        "- Table 3. Exact eligible lightweight base-vs-fine-tuned deltas. Only ranking-eligible pairs are shown; ineligible or missing pairs are excluded from this table.",
        "- Table 4. Midclass screening summary showing valid ranked models together with blocked exact-pair status for each family.",
        "- Table 5. Highclass timeout-collapse failure summary. These rows document screening status and latency burden rather than competitive model ranking.",
        "",
        "## Figures",
        "- Figure 1. Lightweight task-first compare plot showing the main valid-only ranking context for the current publication evidence base.",
        "- Figure 2. Lightweight efficiency companion plot showing valid-only latency and throughput context for the lightweight tier.",
    ]
    if include_validity_figure:
        lines.append(
            "- Figure 3. Cross-tier ranking-eligibility summary showing how the proportion of valid runs decreases from lightweight to highclass under the current local runtime policy."
        )
    else:
        lines.append("- Figure 3. Optional cross-tier validity bar chart omitted in v1; Table 1 carries the same screening-status message in tabular form.")
    lines.extend(
        [
            "- Figure 4. Cross-tier Pareto view of `driving_score_v2` versus mean decision latency for ranking-eligible models only. Highclass contributes no valid points in the current bundle.",
            "- Figure 5. Lightweight Pareto plot copied from the lightweight study outputs, showing the within-tier latency-quality frontier for the current evidence-ready tier.",
            "- Figure 6. Midclass Pareto plot copied from the midclass study outputs, showing a provisional latency-quality frontier under partial screening coverage.",
            "- Figure 7. Highclass Pareto plot copied from the highclass study outputs, retained as a timeout-collapse failure context rather than a valid competitive frontier.",
        ]
    )
    return "\n".join(lines)


def _build_figure_plan(
    bundle_dir: Path,
    lightweight_source_task: Path | None,
    lightweight_source_efficiency: Path | None,
    lightweight_source_pareto: Path | None,
    midclass_source_pareto: Path | None,
    highclass_source_pareto: Path | None,
    include_validity_figure: bool,
) -> str:
    figure1_bundle = bundle_dir / "figures" / "figure_1_lightweight_task_summary.png"
    figure2_bundle = bundle_dir / "figures" / "figure_2_lightweight_efficiency.png"
    figure3_bundle = bundle_dir / "figures" / "figure_3_cross_tier_validity.png"
    figure4_bundle = bundle_dir / "figures" / "figure_4_cross_tier_pareto.png"
    figure5_bundle = bundle_dir / "figures" / "figure_5_lightweight_pareto.png"
    figure6_bundle = bundle_dir / "figures" / "figure_6_midclass_pareto.png"
    figure7_bundle = bundle_dir / "figures" / "figure_7_highclass_pareto.png"
    lines = [
        "# Figure and Table Plan",
        "",
        "| Order | Asset | Source artifact | Bundled path | Status label | Intended message |",
        "| --- | --- | --- | --- | --- | --- |",
        "| Table 1 | Cross-tier validity summary | `analysis/out/*/normalized_records.csv`, `paired_deltas.csv`, `stage1_shortlist.csv` | `table_1_cross_tier_validity_summary.csv` | screening status | Show which tier is results-ready versus screening-only or failure-analysis-only. |",
        "| Table 2 | Lightweight leaderboard | `analysis/out/slm_lightweight_stage1_refreshed/tier_leaderboard_lightweight.csv` | `table_2_lightweight_leaderboard.csv` | valid-only ranking | Present the main comparative ranking for the evidence-ready tier. |",
        "| Table 3 | Lightweight exact pair deltas | `analysis/out/slm_lightweight_stage1_refreshed/paired_deltas.csv` | `table_3_lightweight_exact_pair_deltas.csv` | valid-only ranking | Limit fine-tuning claims to exact eligible pairs. |",
        "| Table 4 | Midclass screening summary | `analysis/out/slm_midclass_stage1_study/tier_leaderboard_midclass.csv`, `paired_deltas.csv`, `normalized_records.csv` | `table_4_midclass_screening_summary.csv` | screening status | Show valid midclass models while documenting why pairwise conclusions remain blocked. |",
        "| Table 5 | Highclass failure summary | `analysis/out/slm_highclass_stage1_study/tier_leaderboard_highclass.csv` | `table_5_highclass_failure_summary.csv` | timeout-collapse failures | Document highclass failure modes and latency burden without implying ranking validity. |",
        (
            f"| Figure 1 | Lightweight task-first compare plot | `{lightweight_source_task}` | `{figure1_bundle}` | valid-only ranking | Main publication figure for lightweight task quality comparison. |"
            if lightweight_source_task
            else f"| Figure 1 | Lightweight task-first compare plot | unavailable | `{figure1_bundle}` | valid-only ranking | Main publication figure for lightweight task quality comparison. |"
        ),
        (
            f"| Figure 2 | Lightweight efficiency companion plot | `{lightweight_source_efficiency}` | `{figure2_bundle}` | valid-only ranking | Companion figure for lightweight latency and throughput context. |"
            if lightweight_source_efficiency
            else f"| Figure 2 | Lightweight efficiency companion plot | unavailable | `{figure2_bundle}` | valid-only ranking | Companion figure for lightweight latency and throughput context. |"
        ),
    ]
    if include_validity_figure:
        lines.append(
            f"| Figure 3 | Cross-tier validity bar chart | generated from Table 1 inputs | `{figure3_bundle}` | screening status | Visualize the decline in ranking eligibility with model tier. |"
        )
    else:
        lines.append(
            "| Figure 3 | Cross-tier validity bar chart | not included in v1 | omitted | screening status | Table 1 already carries the cross-tier validity message. |"
        )
    lines.extend(
        [
            f"| Figure 4 | Cross-tier Pareto plot | generated from valid leaderboard rows across tiers | `{figure4_bundle}` | valid-only ranking | Show the cross-tier latency-quality frontier without implying a single composite score. |",
            (
                f"| Figure 5 | Lightweight Pareto plot | `{lightweight_source_pareto}` | `{figure5_bundle}` | valid-only ranking | Preserve the within-tier Pareto frontier for the main evidence tier. |"
                if lightweight_source_pareto
                else f"| Figure 5 | Lightweight Pareto plot | unavailable | `{figure5_bundle}` | valid-only ranking | Preserve the within-tier Pareto frontier for the main evidence tier. |"
            ),
            (
                f"| Figure 6 | Midclass Pareto plot | `{midclass_source_pareto}` | `{figure6_bundle}` | screening status | Show the provisional midclass frontier while keeping the screening-only interpretation explicit. |"
                if midclass_source_pareto
                else f"| Figure 6 | Midclass Pareto plot | unavailable | `{figure6_bundle}` | screening status | Show the provisional midclass frontier while keeping the screening-only interpretation explicit. |"
            ),
            (
                f"| Figure 7 | Highclass Pareto plot | `{highclass_source_pareto}` | `{figure7_bundle}` | timeout-collapse failures | Retain the highclass latency-quality plot as failure context, not as a valid ranking figure. |"
                if highclass_source_pareto
                else f"| Figure 7 | Highclass Pareto plot | unavailable | `{figure7_bundle}` | timeout-collapse failures | Retain the highclass latency-quality plot as failure context, not as a valid ranking figure. |"
            ),
        ]
    )
    return "\n".join(lines)


def run_publication_bundle(
    *,
    lightweight_study_dir: str | Path,
    midclass_study_dir: str | Path,
    highclass_study_dir: str | Path,
    output_root: str | Path,
    bundle_id: str,
) -> dict[str, Any]:
    lightweight_study_dir = Path(lightweight_study_dir)
    midclass_study_dir = Path(midclass_study_dir)
    highclass_study_dir = Path(highclass_study_dir)
    output_root = Path(output_root)
    bundle_dir = output_root / bundle_id
    figures_dir = bundle_dir / "figures"

    lightweight_study = _study_inputs(lightweight_study_dir)
    midclass_study = _study_inputs(midclass_study_dir)
    highclass_study = _study_inputs(highclass_study_dir)

    validity_rows = [
        _validity_summary_row(lightweight_study, "lightweight"),
        _validity_summary_row(midclass_study, "midclass"),
        _validity_summary_row(highclass_study, "highclass"),
    ]
    lightweight_leaderboard = _build_lightweight_leaderboard(lightweight_study)
    lightweight_pairs = _build_lightweight_pairs(lightweight_study)
    midclass_summary = _build_midclass_summary(midclass_study)
    highclass_failures = _build_highclass_failure_summary(highclass_study)

    _write_csv(bundle_dir / "table_1_cross_tier_validity_summary.csv", VALIDITY_SUMMARY_FIELDS, validity_rows)
    _write_csv(bundle_dir / "table_2_lightweight_leaderboard.csv", LIGHTWEIGHT_LEADERBOARD_FIELDS, lightweight_leaderboard)
    _write_csv(bundle_dir / "table_3_lightweight_exact_pair_deltas.csv", LIGHTWEIGHT_PAIR_FIELDS, lightweight_pairs)
    _write_csv(bundle_dir / "table_4_midclass_screening_summary.csv", MIDCLASS_SCREENING_FIELDS, midclass_summary)
    _write_csv(bundle_dir / "table_5_highclass_failure_summary.csv", HIGHCLASS_FAILURE_FIELDS, highclass_failures)

    primary_compare = _select_primary_compare_report(lightweight_study["normalized"], "lightweight")
    task_plot_source = _compare_plot_path(primary_compare, "_plot.png")
    efficiency_plot_source = _compare_plot_path(primary_compare, "_plot_efficiency.png")
    task_plot_bundled = figures_dir / "figure_1_lightweight_task_summary.png"
    efficiency_plot_bundled = figures_dir / "figure_2_lightweight_efficiency.png"
    task_plot_available = _copy_if_exists(task_plot_source, task_plot_bundled)
    efficiency_plot_available = _copy_if_exists(efficiency_plot_source, efficiency_plot_bundled)

    include_validity_figure = True
    validity_figure_path = figures_dir / "figure_3_cross_tier_validity.png"
    _build_cross_tier_validity_figure(validity_figure_path, validity_rows)
    cross_tier_pareto_path = figures_dir / "figure_4_cross_tier_pareto.png"
    _build_cross_tier_pareto_figure(
        cross_tier_pareto_path,
        {
            "lightweight": lightweight_study["leaderboards"]["lightweight"],
            "midclass": midclass_study["leaderboards"]["midclass"],
            "highclass": highclass_study["leaderboards"]["highclass"],
        },
    )
    lightweight_pareto_source = _study_plot_path(lightweight_study_dir, "pareto_lightweight_latency.png")
    midclass_pareto_source = _study_plot_path(midclass_study_dir, "pareto_midclass_latency.png")
    highclass_pareto_source = _study_plot_path(highclass_study_dir, "pareto_highclass_latency.png")
    lightweight_pareto_available = _copy_if_exists(
        lightweight_pareto_source, figures_dir / "figure_5_lightweight_pareto.png"
    )
    midclass_pareto_available = _copy_if_exists(
        midclass_pareto_source, figures_dir / "figure_6_midclass_pareto.png"
    )
    highclass_pareto_available = _copy_if_exists(
        highclass_pareto_source, figures_dir / "figure_7_highclass_pareto.png"
    )

    _write_text(
        bundle_dir / "evidence_summary.md",
        _build_evidence_summary(
            validity_rows,
            lightweight_leaderboard,
            lightweight_pairs,
            midclass_summary,
            highclass_failures,
        ),
    )
    _write_text(
        bundle_dir / "results_outline.md",
        _build_results_outline(
            validity_rows,
            lightweight_leaderboard,
            lightweight_pairs,
            midclass_summary,
            highclass_failures,
        ),
    )
    _write_text(bundle_dir / "caption_bank.md", _build_caption_bank(include_validity_figure))
    _write_text(
        bundle_dir / "figure_plan.md",
        _build_figure_plan(
            bundle_dir,
            task_plot_source if task_plot_available else None,
            efficiency_plot_source if efficiency_plot_available else None,
            lightweight_pareto_source if lightweight_pareto_available else None,
            midclass_pareto_source if midclass_pareto_available else None,
            highclass_pareto_source if highclass_pareto_available else None,
            include_validity_figure,
        ),
    )

    _write_json(
        bundle_dir / "bundle_manifest.json",
        {
            "bundle_id": bundle_id,
            "lightweight_study_dir": str(lightweight_study_dir),
            "midclass_study_dir": str(midclass_study_dir),
            "highclass_study_dir": str(highclass_study_dir),
            "task_plot_source": str(task_plot_source) if task_plot_available else None,
            "efficiency_plot_source": str(efficiency_plot_source) if efficiency_plot_available else None,
            "lightweight_pareto_source": str(lightweight_pareto_source) if lightweight_pareto_available else None,
            "midclass_pareto_source": str(midclass_pareto_source) if midclass_pareto_available else None,
            "highclass_pareto_source": str(highclass_pareto_source) if highclass_pareto_available else None,
            "validity_summary": validity_rows,
        },
    )

    return {
        "bundle_dir": str(bundle_dir),
        "table_1_rows": len(validity_rows),
        "lightweight_ranked_rows": len(lightweight_leaderboard),
        "lightweight_pair_rows": len(lightweight_pairs),
        "midclass_summary_rows": len(midclass_summary),
        "highclass_failure_rows": len(highclass_failures),
        "task_plot_available": task_plot_available,
        "efficiency_plot_available": efficiency_plot_available,
        "validity_figure_path": str(validity_figure_path),
        "cross_tier_pareto_path": str(cross_tier_pareto_path),
        "lightweight_pareto_available": lightweight_pareto_available,
        "midclass_pareto_available": midclass_pareto_available,
        "highclass_pareto_available": highclass_pareto_available,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a publication-facing three-tier analysis bundle from existing study outputs."
    )
    parser.add_argument(
        "--lightweight-study-dir",
        default="analysis/out/slm_lightweight_stage1_refreshed",
        help="Path to the lightweight study directory.",
    )
    parser.add_argument(
        "--midclass-study-dir",
        default="analysis/out/slm_midclass_stage1_study",
        help="Path to the midclass study directory.",
    )
    parser.add_argument(
        "--highclass-study-dir",
        default="analysis/out/slm_highclass_stage1_study",
        help="Path to the highclass study directory.",
    )
    parser.add_argument(
        "--output-root",
        default="analysis/out",
        help="Directory under which the publication bundle should be created.",
    )
    parser.add_argument(
        "--bundle-id",
        default="publication_three_tier_results_v1",
        help="Name of the output publication bundle directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_publication_bundle(
        lightweight_study_dir=args.lightweight_study_dir,
        midclass_study_dir=args.midclass_study_dir,
        highclass_study_dir=args.highclass_study_dir,
        output_root=args.output_root,
        bundle_id=args.bundle_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
