import argparse
import json
import math
import os
import re

import matplotlib.pyplot as plt

from dilu.runtime import build_model_root, ensure_dir, read_json, write_json_atomic

DENSE_MODEL_THRESHOLD = 8


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_value(v, default=0.0):
    return default if v is None else v


def _normalize_aggregates(report: dict):
    """Support both eval reports ('aggregates': [...]) and run reports ('aggregate': {...})."""
    aggregates = report.get("aggregates")
    if isinstance(aggregates, list) and aggregates:
        return aggregates, "eval"

    single = report.get("aggregate")
    if isinstance(single, dict) and single:
        model_name = report.get("chat_model") or report.get("model") or "run_model"
        normalized = dict(single)
        normalized["model"] = model_name
        return [normalized], "run"

    raise ValueError("No 'aggregates' list or 'aggregate' object found in report.")


def _has_benchmark_metrics(aggregates) -> bool:
    return any(row.get("task_completion_rate") is not None for row in aggregates)


def _has_benchmark_v2_metrics(aggregates) -> bool:
    return any(row.get("driving_score_v2") is not None for row in aggregates)


def _has_energy_metrics(aggregates) -> bool:
    return any(row.get("net_energy_j_mean") is not None for row in aggregates)


def _has_latency_runtime_metrics(aggregates) -> bool:
    return any(
        row.get("decision_latency_ms_avg_mean") is not None
        or row.get("p95_decision_latency_sec_mean") is not None
        or row.get("tokens_per_second_mean") is not None
        or row.get("latency_to_first_action_sec_mean") is not None
        for row in aggregates
    )


def _has_token_metrics(aggregates) -> bool:
    return any(
        row.get("prompt_tokens_total_mean") is not None
        or row.get("completion_tokens_total_mean") is not None
        or row.get("total_tokens_mean") is not None
        for row in aggregates
    )


def _benchmark_invalid_present(aggregates) -> bool:
    return any(row.get("benchmark_result_valid") is False for row in aggregates)


def _shorten_model_name(model_name: str) -> str:
    label = str(model_name or "").strip()
    if not label:
        return "model"

    label = re.sub(r"-v\d+(?=-|$)", "", label)
    label = re.sub(r":(\d+(?:[._]\d+)?)b\b", lambda m: f":{m.group(1).replace('_', '.').replace(',', '.')}", label)
    label = re.sub(r"-(\d+(?:[._]\d+)?)b(?=-|$)", lambda m: f":{m.group(1).replace('_', '.').replace(',', '.')}", label)
    label = re.sub(r"(\d)_(\d)", r"\1.\2", label)
    label = re.sub(r"-{2,}", "-", label).strip("-")
    return label


def _display_model_label(row):
    label = _shorten_model_name(row["model"])
    if row.get("model_skipped_due_to_preflight"):
        label = f"{label}\nPREFLIGHT_SKIP"
    elif row.get("ollama_preflight_ok") is False:
        label = f"{label}\nPREFLIGHT_FAIL"
    if row.get("model_quarantined_due_to_timeout_collapse"):
        label = f"{label}\nQUARANTINED"
    if row.get("episode_execution_complete") is False:
        label = f"{label}\nINCOMPLETE"
    return label


def _display_model_labels(aggregates):
    return [_display_model_label(row) for row in aggregates]


def _benchmark_headline_score_key(aggregates) -> str:
    return "driving_score_v2" if _has_benchmark_v2_metrics(aggregates) else "driving_score"


def _benchmark_headline_score_title(aggregates) -> str:
    return "Driving Score v2" if _has_benchmark_v2_metrics(aggregates) else "Driving Score"


def _should_use_horizontal_layout(model_count: int) -> bool:
    return model_count > DENSE_MODEL_THRESHOLD


def _should_show_value_labels(model_count: int) -> bool:
    return model_count <= DENSE_MODEL_THRESHOLD


def _build_output_path(output_path: str, suffix: str | None = None) -> str:
    if not suffix:
        return output_path
    base, ext = os.path.splitext(output_path)
    return f"{base}_{suffix}{ext or '.png'}"


def _metric_value(row: dict, keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return default


def _chart(title: str, *keys: str, color: str, ylim=None) -> dict:
    return {"title": title, "keys": keys, "color": color, "ylim": ylim}


def _materialize_charts(aggregates, chart_specs):
    return [
        {
            "title": spec["title"],
            "values": [_safe_value(_metric_value(row, spec["keys"])) for row in aggregates],
            "color": spec["color"],
            "ylim": spec.get("ylim"),
        }
        for spec in chart_specs
    ]


def _format_value(value, ylim=None) -> str:
    if isinstance(ylim, tuple) and ylim == (0, 1):
        return f"{value:.3f}"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _flatten_axes(axes):
    if isinstance(axes, plt.Axes):
        return [axes]
    flat_axes = []
    for row_axes in (axes if isinstance(axes, (list, tuple)) else axes.tolist()):
        if isinstance(row_axes, (list, tuple)):
            flat_axes.extend(list(row_axes))
        else:
            flat_axes.append(row_axes)
    return flat_axes


def _sorted_aggregates_for_plot(aggregates):
    benchmark_mode = _has_benchmark_metrics(aggregates)
    if benchmark_mode:
        return sorted(
            aggregates,
            key=lambda row: (
                _safe_value(row.get("driving_score_v2", row.get("driving_score"))),
                _safe_value(row.get("task_completion_rate")),
                -_safe_value(row.get("decision_latency_ms_avg_mean"), default=1e12),
            ),
            reverse=True,
        )

    return sorted(
        aggregates,
        key=lambda row: (
            _safe_value(row.get("no_collision_rate"), default=1.0 - _safe_value(row.get("crash_rate"))),
            -_safe_value(row.get("crash_rate")),
        ),
        reverse=True,
    )


def _plot_grid(models, charts, title: str, output_path: str) -> None:
    n = len(charts)
    cols = 1 if n <= 3 else 2
    rows = math.ceil(n / cols)
    use_horizontal = _should_use_horizontal_layout(len(models))
    show_value_labels = _should_show_value_labels(len(models))

    if use_horizontal:
        fig_width = 16
        panel_height = max(3.8, 1.8 + 0.45 * len(models))
    else:
        fig_width = 14 if cols == 2 else 12
        panel_height = 4.2
    fig_height = max(4.8, panel_height * rows)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    flat_axes = _flatten_axes(axes)

    for ax, chart in zip(flat_axes[:n], charts):
        values = chart["values"]
        ax.set_title(chart["title"], fontsize=12)
        if use_horizontal:
            positions = list(range(len(models)))
            bars = ax.barh(positions, values, color=chart["color"], alpha=0.9)
            ax.set_yticks(positions)
            ax.set_yticklabels(models, fontsize=9)
            ax.invert_yaxis()
            if chart["ylim"] is not None:
                ax.set_xlim(*chart["ylim"])
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            if show_value_labels:
                max_value = max(values) if values else 0.0
                label_pad = max(0.02, max_value * 0.01 if max_value else 0.02)
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_width() + label_pad,
                        bar.get_y() + bar.get_height() / 2,
                        _format_value(value, chart["ylim"]),
                        va="center",
                        ha="left",
                        fontsize=8,
                    )
        else:
            positions = list(range(len(models)))
            bars = ax.bar(positions, values, color=chart["color"], alpha=0.9)
            ax.set_xticks(positions)
            ax.set_xticklabels(models, rotation=90, ha="right", fontsize=9)
            if chart["ylim"] is not None:
                ax.set_ylim(*chart["ylim"])
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            if show_value_labels:
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        _format_value(value, chart["ylim"]),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    for ax in flat_axes[n:]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if use_horizontal:
        longest = max((len(label) for label in models), default=0)
        fig.subplots_adjust(left=min(0.42, 0.18 + longest / 120.0), hspace=0.4, wspace=0.25)
    else:
        longest = max((len(label) for label in models), default=0)
        fig.subplots_adjust(bottom=min(0.42, 0.18 + longest / 90.0), hspace=0.35, wspace=0.25)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_energy_tradeoff_scatter(aggregates, output_path: str) -> None:
    if not _has_energy_metrics(aggregates):
        return
    ordered = _sorted_aggregates_for_plot(aggregates)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Energy / Quality Trade-offs", fontsize=14, fontweight="bold")

    scatter_specs = [
        {
            "x_key": "net_energy_j_mean",
            "y_key": _benchmark_headline_score_key(ordered) if _has_benchmark_metrics(ordered) else "no_collision_rate",
            "title": "Energy vs Quality",
            "ylabel": _benchmark_headline_score_title(ordered) if _has_benchmark_metrics(ordered) else "No-Collision Rate",
        },
        {
            "x_key": "net_energy_j_mean",
            "y_key": "decision_latency_ms_avg_mean",
            "title": "Energy vs Decision Latency",
            "ylabel": "Decision Latency Mean (ms)",
        },
    ]

    for ax, spec in zip(axes, scatter_specs):
        xs = [_safe_value(row.get(spec["x_key"])) for row in ordered]
        ys = [_safe_value(row.get(spec["y_key"])) for row in ordered]
        ax.scatter(xs, ys, color="#1f78b4", s=70, alpha=0.9)
        for row, x_val, y_val in zip(ordered, xs, ys):
            ax.annotate(_display_model_label(row), (x_val, y_val), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_title(spec["title"])
        ax.set_xlabel("Net Energy Mean (J)")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_task_summary_figures(aggregates, title_prefix: str, all_metrics: bool = False):
    charts = [
        _chart(_benchmark_headline_score_title(aggregates), _benchmark_headline_score_key(aggregates), color="#a65628", ylim=(0, 1)),
        _chart("Task Completion Rate", "task_completion_rate", color="#1f78b4", ylim=(0, 1)),
        _chart("TTC Score Mean", "ttc_score_mean", color="#e41a1c", ylim=(0, 1)),
        _chart("Time Efficiency Score Mean", "time_efficiency_score_mean", color="#ff7f00", ylim=(0, 1)),
    ]
    if all_metrics:
        charts.extend(
            [
                _chart("Overall Score Mean", "overall_score_mean", color="#984ea3", ylim=(0, 1)),
                _chart("Speed Variance Score Mean", "speed_variance_score_mean", color="#4daf4a", ylim=(0, 1)),
            ]
        )
        if _has_benchmark_v2_metrics(aggregates):
            charts.extend(
                [
                    _chart("Overall Score v2 Mean", "overall_score_v2_mean", color="#6a3d9a", ylim=(0, 1)),
                    _chart("Driving Score (legacy)", "driving_score", color="#b15928", ylim=(0, 1)),
                    _chart("Behavior Penalty Factor v2", "behavior_penalty_factor_v2_mean", color="#cab2d6", ylim=(0, 1)),
                ]
            )
    return [{"suffix": None, "title": f"{title_prefix} (Task Summary)", "charts": charts}]


def build_runtime_summary_figures(aggregates, title_prefix: str, all_metrics: bool = False):
    charts = [
        _chart("Crash Rate", "crash_rate", color="#d95f02", ylim=(0, 1)),
        _chart("No-Collision Rate", "no_collision_rate", color="#1b9e77", ylim=(0, 1)),
        _chart("Average Steps", "avg_steps", color="#7570b3"),
        _chart("Avg Episode Runtime (s)", "avg_episode_runtime_sec", color="#66a61e"),
    ]
    if all_metrics:
        charts.extend(
            [
                _chart("Average Reward Sum", "avg_reward_sum", color="#4daf4a"),
                _chart("Average Reward / Step", "avg_reward_per_step", color="#a65628"),
            ]
        )
    return [{"suffix": None, "title": title_prefix, "charts": charts}]


def build_behavior_figures(aggregates, title_prefix: str, all_metrics: bool = False):
    if not _has_benchmark_metrics(aggregates):
        return []
    charts = [
        _chart("Stop Episode Rate", "stop_episode_rate", color="#377eb8", ylim=(0, 1)),
        _chart("Near-Stop Rate Mean", "near_stop_rate_mean", color="#f781bf", ylim=(0, 1)),
    ]
    if all_metrics and any(row.get("min_ego_speed_mps_mean") is not None for row in aggregates):
        charts.append(_chart("Min Ego Speed Mean (m/s)", "min_ego_speed_mps_mean", color="#4daf4a"))
    return [{"suffix": "behavior", "title": f"{title_prefix} (Behavior)", "charts": charts}]


def build_efficiency_figures(aggregates, title_prefix: str, all_metrics: bool = False):
    charts = []
    if any(row.get("decision_latency_ms_avg_mean") is not None for row in aggregates):
        charts.append(_chart("Decision Latency Mean (ms)", "decision_latency_ms_avg_mean", color="#e6ab02"))
    if any(row.get("tokens_per_second_mean") is not None for row in aggregates):
        charts.append(_chart("Tokens / Second", "tokens_per_second_mean", color="#66a61e"))
    if any(row.get("completion_tokens_total_mean") is not None for row in aggregates):
        charts.append(_chart("Completion Tokens Mean", "completion_tokens_total_mean", color="#377eb8"))
    if all_metrics:
        if any(row.get("p95_decision_latency_sec_mean") is not None for row in aggregates):
            charts.append(_chart("P95 Decision Latency (s)", "p95_decision_latency_sec_mean", color="#a6761d"))
        if any(row.get("prompt_tokens_total_mean") is not None for row in aggregates):
            charts.append(_chart("Prompt Tokens Mean", "prompt_tokens_total_mean", color="#4daf4a"))
        if any(row.get("total_tokens_mean") is not None for row in aggregates):
            charts.append(_chart("Total Tokens Mean", "total_tokens_mean", color="#984ea3"))
    if not charts:
        return []
    return [{"suffix": "efficiency", "title": f"{title_prefix} (Efficiency)", "charts": charts}]


def build_energy_figures(aggregates, title_prefix: str, all_metrics: bool = False):
    charts = []
    if any(row.get("net_energy_j_mean") is not None for row in aggregates):
        charts.append(_chart("Net Energy Mean (J)", "net_energy_j_mean", color="#1b9e77"))
    if any(row.get("energy_per_decision_j_mean") is not None for row in aggregates):
        charts.append(_chart("Energy / Decision (J)", "energy_per_decision_j_mean", color="#d95f02"))
    if any(row.get("energy_per_token_j_mean") is not None for row in aggregates):
        charts.append(_chart("Energy / Token (J)", "energy_per_token_j_mean", color="#7570b3"))
    if not charts:
        return []
    return [{"suffix": "energy", "title": f"{title_prefix} (Energy)", "charts": charts}]


def build_extended_runtime_figures(aggregates, title_prefix: str, all_metrics: bool = False):
    charts = [
        _chart("Decision Timeout Rate", "decision_timeout_rate_mean", color="#e7298a", ylim=(0, 1)),
        _chart("Fallback Action Rate", "fallback_action_rate_mean", color="#666666", ylim=(0, 1)),
        _chart("Timeout Episode Rate", "timeout_episode_rate", color="#a6761d", ylim=(0, 1)),
        _chart("TTC Danger Rate", "ttc_danger_rate_mean", color="#e41a1c", ylim=(0, 1)),
    ]
    if all_metrics:
        if any(row.get("timeout_episode_count") is not None for row in aggregates):
            charts.append(_chart("Timeout Episodes (count)", "timeout_episode_count", color="#1f78b4"))
        if any(row.get("headway_violation_rate_mean") is not None for row in aggregates):
            charts.append(_chart("Headway Violation Rate", "headway_violation_rate_mean", color="#ff7f00", ylim=(0, 1)))
        if any(row.get("lane_change_rate_mean") is not None for row in aggregates):
            charts.append(_chart("Lane Change Rate", "lane_change_rate_mean", color="#377eb8", ylim=(0, 1)))
        if any(row.get("flap_accel_decel_rate_mean") is not None for row in aggregates):
            charts.append(_chart("Accel/Decel Flap Rate", "flap_accel_decel_rate_mean", color="#984ea3", ylim=(0, 1)))
    materialized = _materialize_charts(aggregates, charts)
    if not any(any(value != 0 for value in chart["values"]) for chart in materialized):
        return []
    return [{"suffix": "runtime", "title": f"{title_prefix} (Runtime / Safety)", "charts": charts}]


def _emit_figures(aggregates, figure_specs, output_path):
    models = _display_model_labels(aggregates)
    saved_paths = []
    for spec in figure_specs:
        path = _build_output_path(output_path, spec["suffix"])
        charts = _materialize_charts(aggregates, spec["charts"])
        _plot_grid(models, charts, spec["title"], path)
        saved_paths.append(path)
    return saved_paths


def plot_aggregates(report: dict, output_path: str, extended: bool = False, all_metrics: bool = False, emit_companions: bool = True) -> dict:
    aggregates, source_type = _normalize_aggregates(report)
    ordered = _sorted_aggregates_for_plot(aggregates)

    title_prefix = "DiLu Run Metrics" if source_type == "run" else "DiLu Model Comparison"
    benchmark_mode = _has_benchmark_metrics(ordered)
    if benchmark_mode and _benchmark_invalid_present(ordered):
        title_prefix += " [invalid benchmark runs flagged]"

    figure_specs = []
    if benchmark_mode:
        figure_specs.extend(build_task_summary_figures(ordered, title_prefix, all_metrics=all_metrics))
    else:
        figure_specs.extend(build_runtime_summary_figures(ordered, title_prefix, all_metrics=all_metrics))

    if emit_companions:
        figure_specs.extend(build_behavior_figures(ordered, title_prefix, all_metrics=all_metrics))
        figure_specs.extend(build_efficiency_figures(ordered, title_prefix, all_metrics=all_metrics))
        figure_specs.extend(build_energy_figures(ordered, title_prefix, all_metrics=all_metrics))
        if extended or all_metrics:
            figure_specs.extend(build_extended_runtime_figures(ordered, title_prefix, all_metrics=all_metrics))

    if not figure_specs:
        raise ValueError("No chart figures were generated for the provided report.")

    saved_paths = _emit_figures(ordered, figure_specs, output_path)
    return {
        "primary_path": saved_paths[0],
        "companion_paths": saved_paths[1:],
        "all_paths": saved_paths,
    }


def emit_per_model_plots(report: dict, all_metrics: bool, extended: bool) -> list:
    """Emit one plot per model under each model's plots folder (for eval reports)."""
    aggregates, source_type = _normalize_aggregates(report)
    if source_type != "eval":
        return []

    experiment_root = report.get("experiment_root")
    if not experiment_root:
        return []

    outputs = []
    for row in aggregates:
        model_name = row.get("model", "model")
        model_root = build_model_root(experiment_root, model_name)
        plots_dir = ensure_dir(os.path.join(model_root, "plots"))
        suffix = "all" if all_metrics else ("extended" if extended else "default")
        output_path = os.path.join(plots_dir, f"model_metrics_{suffix}.png")
        single_report = {
            "aggregate": row,
            "chat_model": model_name,
        }
        plot_aggregates(single_report, output_path, extended=extended, all_metrics=all_metrics, emit_companions=False)
        outputs.append(output_path)
    return outputs


def update_manifest_for_plots(report: dict, global_plot_path: str, companion_plot_paths: list, per_model_plot_paths: list) -> None:
    experiment_root = report.get("experiment_root")
    if not experiment_root:
        return

    manifest_path = os.path.join(experiment_root, "manifest.json")
    manifest = read_json(manifest_path, default={})
    compare_meta = manifest.setdefault("compare", {})
    compare_meta["latest_plot"] = global_plot_path
    compare_meta["latest_plot_companions"] = list(companion_plot_paths)

    plot_history = compare_meta.setdefault("plot_history", [])
    for path in [global_plot_path, *companion_plot_paths]:
        if path not in plot_history:
            plot_history.append(path)

    models = manifest.setdefault("models", {})
    for path in per_model_plot_paths:
        parts = path.replace("\\", "/").split("/")
        if "models" not in parts:
            continue
        idx = parts.index("models")
        if idx + 1 >= len(parts):
            continue
        model_slug = parts[idx + 1]
        for model_name, model_meta in models.items():
            if str(model_meta.get("slug")) == model_slug:
                model_meta["latest_plot"] = path
                break

    write_json_atomic(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregate metrics from evaluate_models_ollama.py JSON output.")
    parser.add_argument("-i", "--input", required=True, help="Path to comparison JSON report")
    parser.add_argument("-o", "--output", default=None, help="Output image path (PNG). Defaults next to input file.")
    parser.add_argument("--extended", action="store_true", help="Emit companion runtime/safety plots for comparison reports.")
    parser.add_argument("--all-metrics", action="store_true", help="Emit all available metrics across themed plot files.")
    parser.add_argument("--emit-per-model", action="store_true", help="Emit one plot per model under experiment model plot folders.")
    args = parser.parse_args()

    report = load_report(args.input)

    if args.output:
        output_path = args.output
    else:
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}_plot.png"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_result = plot_aggregates(report, output_path, extended=args.extended, all_metrics=args.all_metrics)
    print(f"Saved plot: {plot_result['primary_path']}")
    for companion_path in plot_result["companion_paths"]:
        print(f"Saved companion plot: {companion_path}")

    aggregates, _ = _normalize_aggregates(report)
    if _has_energy_metrics(aggregates):
        base, ext = os.path.splitext(output_path)
        pareto_path = f"{base}_pareto{ext or '.png'}"
        _plot_energy_tradeoff_scatter(aggregates, pareto_path)
        print(f"Saved energy trade-off plot: {pareto_path}")

    per_model_outputs = []
    if args.emit_per_model:
        per_model_outputs = emit_per_model_plots(report, all_metrics=args.all_metrics, extended=args.extended)
        if per_model_outputs:
            print("Saved per-model plots:")
            for path in per_model_outputs:
                print(f"- {path}")
        else:
            print("No per-model plots emitted (report is not an experiment eval report with experiment_root).")

    update_manifest_for_plots(report, plot_result["primary_path"], plot_result["companion_paths"], per_model_outputs)


if __name__ == "__main__":
    main()
