import argparse
import json
import os
import math

import matplotlib.pyplot as plt

from dilu.runtime import build_model_root, ensure_dir, read_json, write_json_atomic


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


def _display_model_label(row):
    label = str(row["model"])
    if row.get("benchmark_result_valid") is False:
        label = f"{label}\nINVALID"
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


def _plot_grid(models, charts, title: str, output_path: str) -> None:
    n = len(charts)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if isinstance(axes, plt.Axes):
        flat_axes = [axes]
    else:
        flat_axes = []
        for row_axes in (axes if isinstance(axes, (list, tuple)) else axes.tolist()):
            if isinstance(row_axes, (list, tuple)):
                flat_axes.extend(list(row_axes))
            else:
                flat_axes.append(row_axes)

    for ax, chart in zip(flat_axes[:n], charts):
        values = chart["values"]
        bars = ax.bar(models, values, color=chart["color"], alpha=0.9)
        ax.set_title(chart["title"])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right")
        if chart["ylim"] is not None:
            ax.set_ylim(*chart["ylim"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if isinstance(value, float) else str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Hide any unused axes when chart count is odd.
    for ax in flat_axes[n:]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_energy_tradeoff_scatter(aggregates, output_path: str) -> None:
    if not _has_energy_metrics(aggregates):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Energy / Quality Trade-offs", fontsize=14, fontweight="bold")

    scatter_specs = [
        {
            "x_key": "net_energy_j_mean",
            "y_key": _benchmark_headline_score_key(aggregates) if _has_benchmark_metrics(aggregates) else "no_collision_rate",
            "title": "Energy vs Quality",
            "ylabel": _benchmark_headline_score_title(aggregates) if _has_benchmark_metrics(aggregates) else "No-Collision Rate",
        },
        {
            "x_key": "net_energy_j_mean",
            "y_key": "decision_latency_ms_avg_mean",
            "title": "Energy vs Decision Latency",
            "ylabel": "Decision Latency Mean (ms)",
        },
    ]

    for ax, spec in zip(axes, scatter_specs):
        xs = [_safe_value(row.get(spec["x_key"])) for row in aggregates]
        ys = [_safe_value(row.get(spec["y_key"])) for row in aggregates]
        ax.scatter(xs, ys, color="#1f78b4", s=70, alpha=0.9)
        for row, x_val, y_val in zip(aggregates, xs, ys):
            ax.annotate(_display_model_label(row), (x_val, y_val), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_title(spec["title"])
        ax.set_xlabel("Net Energy Mean (J)")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_aggregates(report: dict, output_path: str, extended: bool = False, all_metrics: bool = False) -> None:
    aggregates, source_type = _normalize_aggregates(report)

    models = _display_model_labels(aggregates)
    title_prefix = "DiLu Run Metrics" if source_type == "run" else "DiLu Model Comparison"
    benchmark_mode = _has_benchmark_metrics(aggregates)
    energy_mode = _has_energy_metrics(aggregates)
    latency_runtime_mode = _has_latency_runtime_metrics(aggregates)
    token_metrics_mode = _has_token_metrics(aggregates)
    if benchmark_mode and _benchmark_invalid_present(aggregates):
        title_prefix += " [invalid benchmark runs flagged]"
    benchmark_headline_key = _benchmark_headline_score_key(aggregates)
    benchmark_headline_title = _benchmark_headline_score_title(aggregates)
    benchmark_has_v2 = _has_benchmark_v2_metrics(aggregates)

    if all_metrics:
        charts = [
            {"values": [_safe_value(row.get("crash_rate")) for row in aggregates], "title": "Crash Rate", "ylim": (0, 1), "color": "#d95f02"},
            {"values": [_safe_value(row.get("no_collision_rate")) for row in aggregates], "title": "No-Collision Rate", "ylim": (0, 1), "color": "#1b9e77"},
            {"values": [_safe_value(row.get("avg_steps")) for row in aggregates], "title": "Average Steps", "ylim": None, "color": "#7570b3"},
            {"values": [_safe_value(row.get("avg_episode_runtime_sec")) for row in aggregates], "title": "Avg Episode Runtime (s)", "ylim": None, "color": "#66a61e"},
            {"values": [_safe_value(row.get("ttc_danger_rate_mean")) for row in aggregates], "title": "TTC Danger Rate", "ylim": (0, 1), "color": "#e41a1c"},
            {"values": [_safe_value(row.get("headway_violation_rate_mean")) for row in aggregates], "title": "Headway Violation Rate", "ylim": (0, 1), "color": "#ff7f00"},
            {"values": [_safe_value(row.get("lane_change_rate_mean")) for row in aggregates], "title": "Lane Change Rate", "ylim": (0, 1), "color": "#377eb8"},
            {"values": [_safe_value(row.get("flap_accel_decel_rate_mean")) for row in aggregates], "title": "Accel/Decel Flap Rate", "ylim": (0, 1), "color": "#984ea3"},
            {"values": [_safe_value(row.get("avg_reward_sum")) for row in aggregates], "title": "Average Reward Sum", "ylim": None, "color": "#4daf4a"},
            {"values": [_safe_value(row.get("avg_reward_per_step")) for row in aggregates], "title": "Average Reward/Step", "ylim": None, "color": "#a65628"},
            {"values": [_safe_value(row.get("avg_ego_speed_mps")) for row in aggregates], "title": "Average Ego Speed (m/s)", "ylim": None, "color": "#f781bf"},
            {"values": [_safe_value(row.get("format_failure_rate_mean")) for row in aggregates], "title": "Format Failure Rate", "ylim": (0, 1), "color": "#999999"},
            {"values": [_safe_value(row.get("decision_latency_ms_avg_mean", row.get("decision_latency_ms_avg"))) for row in aggregates], "title": "Decision Latency (ms)", "ylim": None, "color": "#17becf"},
            {"values": [_safe_value(row.get("decision_timeout_rate_mean")) for row in aggregates], "title": "Decision Timeout Rate", "ylim": (0, 1), "color": "#e7298a"},
            {"values": [_safe_value(row.get("timeout_episode_rate")) for row in aggregates], "title": "Timeout Episode Rate", "ylim": (0, 1), "color": "#a6761d"},
            {"values": [_safe_value(row.get("fallback_action_rate_mean")) for row in aggregates], "title": "Fallback Action Rate", "ylim": (0, 1), "color": "#666666"},
            {"values": [_safe_value(row.get("timeout_episode_count")) for row in aggregates], "title": "Timeout Episodes (count)", "ylim": None, "color": "#1f78b4"},
        ]
        if energy_mode:
            charts.extend(
                [
                    {"values": [_safe_value(row.get("net_energy_j_mean")) for row in aggregates], "title": "Net Energy Mean (J)", "ylim": None, "color": "#1b9e77"},
                    {"values": [_safe_value(row.get("energy_per_decision_j_mean")) for row in aggregates], "title": "Energy / Decision (J)", "ylim": None, "color": "#d95f02"},
                    {"values": [_safe_value(row.get("energy_per_token_j_mean")) for row in aggregates], "title": "Energy / Token (J)", "ylim": None, "color": "#7570b3"},
                ]
            )
        if latency_runtime_mode:
            charts.extend(
                [
                    {"values": [_safe_value(row.get("tokens_per_second_mean")) for row in aggregates], "title": "Tokens / Second", "ylim": None, "color": "#66a61e"},
                    {"values": [_safe_value(row.get("decision_latency_ms_avg_mean")) for row in aggregates], "title": "Decision Latency Mean (ms)", "ylim": None, "color": "#e6ab02"},
                    {"values": [_safe_value(row.get("p95_decision_latency_sec_mean")) for row in aggregates], "title": "P95 Decision Latency (s)", "ylim": None, "color": "#a6761d"},
                ]
            )
        if token_metrics_mode:
            charts.extend(
                [
                    {"values": [_safe_value(row.get("prompt_tokens_total_mean")) for row in aggregates], "title": "Prompt Tokens Mean", "ylim": None, "color": "#4daf4a"},
                    {"values": [_safe_value(row.get("completion_tokens_total_mean")) for row in aggregates], "title": "Completion Tokens Mean", "ylim": None, "color": "#377eb8"},
                    {"values": [_safe_value(row.get("total_tokens_mean")) for row in aggregates], "title": "Total Tokens Mean", "ylim": None, "color": "#984ea3"},
                ]
            )
        if benchmark_mode:
            charts.extend(
                [
                    {"values": [_safe_value(row.get("task_completion_rate")) for row in aggregates], "title": "Task Completion Rate", "ylim": (0, 1), "color": "#1f78b4"},
                    {"values": [_safe_value(row.get("ttc_score_mean")) for row in aggregates], "title": "TTC Score Mean", "ylim": (0, 1), "color": "#e41a1c"},
                    {"values": [_safe_value(row.get("speed_variance_score_mean")) for row in aggregates], "title": "Speed Variance Score Mean", "ylim": (0, 1), "color": "#4daf4a"},
                    {"values": [_safe_value(row.get("time_efficiency_score_mean")) for row in aggregates], "title": "Time Efficiency Score Mean", "ylim": (0, 1), "color": "#ff7f00"},
                    {"values": [_safe_value(row.get("overall_score_mean")) for row in aggregates], "title": "Overall Score Mean", "ylim": (0, 1), "color": "#984ea3"},
                    {"values": [_safe_value(row.get(benchmark_headline_key)) for row in aggregates], "title": benchmark_headline_title, "ylim": (0, 1), "color": "#a65628"},
                    {"values": [_safe_value(row.get("stop_episode_rate")) for row in aggregates], "title": "Stop Episode Rate", "ylim": (0, 1), "color": "#377eb8"},
                    {"values": [_safe_value(row.get("near_stop_rate_mean")) for row in aggregates], "title": "Near-Stop Rate Mean", "ylim": (0, 1), "color": "#f781bf"},
                ]
            )
            if benchmark_has_v2:
                charts.extend(
                    [
                        {"values": [_safe_value(row.get("overall_score_v2_mean")) for row in aggregates], "title": "Overall Score v2 Mean", "ylim": (0, 1), "color": "#6a3d9a"},
                        {"values": [_safe_value(row.get("driving_score")) for row in aggregates], "title": "Driving Score (legacy)", "ylim": (0, 1), "color": "#b15928"},
                        {"values": [_safe_value(row.get("behavior_penalty_factor_v2_mean")) for row in aggregates], "title": "Behavior Penalty Factor v2", "ylim": (0, 1), "color": "#cab2d6"},
                    ]
                )
        _plot_grid(models, charts, f"{title_prefix} (All Metrics)", output_path)
        return

    if not extended:
        if benchmark_mode:
            charts = [
                {"values": [_safe_value(row.get("task_completion_rate")) for row in aggregates], "title": "Task Completion Rate", "ylim": (0, 1), "color": "#1f78b4"},
                {"values": [_safe_value(row.get(benchmark_headline_key)) for row in aggregates], "title": benchmark_headline_title, "ylim": (0, 1), "color": "#a65628"},
                {"values": [_safe_value(row.get("ttc_score_mean")) for row in aggregates], "title": "TTC Score Mean", "ylim": (0, 1), "color": "#e41a1c"},
                {"values": [_safe_value(row.get("time_efficiency_score_mean")) for row in aggregates], "title": "Time Efficiency Score Mean", "ylim": (0, 1), "color": "#ff7f00"},
                {"values": [_safe_value(row.get("stop_episode_rate")) for row in aggregates], "title": "Stop Episode Rate", "ylim": (0, 1), "color": "#377eb8"},
                {"values": [_safe_value(row.get("near_stop_rate_mean")) for row in aggregates], "title": "Near-Stop Rate Mean", "ylim": (0, 1), "color": "#f781bf"},
            ]
            if energy_mode:
                charts.extend(
                    [
                        {"values": [_safe_value(row.get("net_energy_j_mean")) for row in aggregates], "title": "Net Energy Mean (J)", "ylim": None, "color": "#1b9e77"},
                    ]
                )
            if latency_runtime_mode:
                charts.extend(
                    [
                        {"values": [_safe_value(row.get("tokens_per_second_mean")) for row in aggregates], "title": "Tokens / Second", "ylim": None, "color": "#66a61e"},
                        {"values": [_safe_value(row.get("decision_latency_ms_avg_mean")) for row in aggregates], "title": "Decision Latency Mean (ms)", "ylim": None, "color": "#e6ab02"},
                    ]
                )
            if token_metrics_mode:
                charts.extend(
                    [
                        {"values": [_safe_value(row.get("completion_tokens_total_mean")) for row in aggregates], "title": "Completion Tokens Mean", "ylim": None, "color": "#377eb8"},
                    ]
                )
        else:
            charts = [
                {"values": [_safe_value(row.get("crash_rate")) for row in aggregates], "title": "Crash Rate", "ylim": (0, 1), "color": "#d95f02"},
                {"values": [_safe_value(row.get("no_collision_rate")) for row in aggregates], "title": "No-Collision Rate", "ylim": (0, 1), "color": "#1b9e77"},
                {"values": [_safe_value(row.get("avg_steps")) for row in aggregates], "title": "Average Steps", "ylim": None, "color": "#7570b3"},
                {"values": [_safe_value(row.get("avg_episode_runtime_sec")) for row in aggregates], "title": "Avg Episode Runtime (s)", "ylim": None, "color": "#66a61e"},
            ]
            if energy_mode:
                charts.extend(
                    [
                        {"values": [_safe_value(row.get("net_energy_j_mean")) for row in aggregates], "title": "Net Energy Mean (J)", "ylim": None, "color": "#1b9e77"},
                    ]
                )
            if latency_runtime_mode:
                charts.extend(
                    [
                        {"values": [_safe_value(row.get("decision_latency_ms_avg_mean")) for row in aggregates], "title": "Decision Latency Mean (ms)", "ylim": None, "color": "#e6ab02"},
                        {"values": [_safe_value(row.get("tokens_per_second_mean")) for row in aggregates], "title": "Tokens / Second", "ylim": None, "color": "#66a61e"},
                    ]
                )
            if token_metrics_mode:
                charts.extend(
                    [
                        {"values": [_safe_value(row.get("completion_tokens_total_mean")) for row in aggregates], "title": "Completion Tokens Mean", "ylim": None, "color": "#377eb8"},
                    ]
                )
        _plot_grid(models, charts, title_prefix, output_path)
        return

    if benchmark_mode:
        charts = [
            {"values": [_safe_value(row.get("task_completion_rate")) for row in aggregates], "title": "Task Completion Rate", "ylim": (0, 1), "color": "#1f78b4"},
            {"values": [_safe_value(row.get(benchmark_headline_key)) for row in aggregates], "title": benchmark_headline_title, "ylim": (0, 1), "color": "#a65628"},
            {"values": [_safe_value(row.get("ttc_score_mean")) for row in aggregates], "title": "TTC Score Mean", "ylim": (0, 1), "color": "#e41a1c"},
            {"values": [_safe_value(row.get("speed_variance_score_mean")) for row in aggregates], "title": "Speed Variance Score Mean", "ylim": (0, 1), "color": "#4daf4a"},
            {"values": [_safe_value(row.get("time_efficiency_score_mean")) for row in aggregates], "title": "Time Efficiency Score Mean", "ylim": (0, 1), "color": "#ff7f00"},
            {"values": [_safe_value(row.get("overall_score_mean")) for row in aggregates], "title": "Overall Score Mean", "ylim": (0, 1), "color": "#984ea3"},
            {"values": [_safe_value(row.get("stop_episode_rate")) for row in aggregates], "title": "Stop Episode Rate", "ylim": (0, 1), "color": "#377eb8"},
            {"values": [_safe_value(row.get("near_stop_rate_mean")) for row in aggregates], "title": "Near-Stop Rate Mean", "ylim": (0, 1), "color": "#f781bf"},
        ]
        if benchmark_has_v2:
            charts.insert(
                5,
                {"values": [_safe_value(row.get("overall_score_v2_mean")) for row in aggregates], "title": "Overall Score v2 Mean", "ylim": (0, 1), "color": "#6a3d9a"},
            )
            charts.insert(
                2,
                {"values": [_safe_value(row.get("driving_score")) for row in aggregates], "title": "Driving Score (legacy)", "ylim": (0, 1), "color": "#b15928"},
            )
            charts.append(
                {"values": [_safe_value(row.get("behavior_penalty_factor_v2_mean")) for row in aggregates], "title": "Behavior Penalty Factor v2", "ylim": (0, 1), "color": "#cab2d6"},
            )
        if energy_mode:
            charts.extend(
                [
                    {"values": [_safe_value(row.get("net_energy_j_mean")) for row in aggregates], "title": "Net Energy Mean (J)", "ylim": None, "color": "#1b9e77"},
                    {"values": [_safe_value(row.get("energy_per_decision_j_mean")) for row in aggregates], "title": "Energy / Decision (J)", "ylim": None, "color": "#d95f02"},
                ]
            )
        if latency_runtime_mode:
            charts.extend(
                [
                    {"values": [_safe_value(row.get("tokens_per_second_mean")) for row in aggregates], "title": "Tokens / Second", "ylim": None, "color": "#66a61e"},
                    {"values": [_safe_value(row.get("decision_latency_ms_avg_mean")) for row in aggregates], "title": "Decision Latency Mean (ms)", "ylim": None, "color": "#e6ab02"},
                ]
            )
        if token_metrics_mode:
            charts.extend(
                [
                    {"values": [_safe_value(row.get("prompt_tokens_total_mean")) for row in aggregates], "title": "Prompt Tokens Mean", "ylim": None, "color": "#4daf4a"},
                    {"values": [_safe_value(row.get("completion_tokens_total_mean")) for row in aggregates], "title": "Completion Tokens Mean", "ylim": None, "color": "#377eb8"},
                    {"values": [_safe_value(row.get("total_tokens_mean")) for row in aggregates], "title": "Total Tokens Mean", "ylim": None, "color": "#984ea3"},
                ]
            )
        _plot_grid(models, charts, f"{title_prefix} (Benchmark Tasks)", output_path)
        return

    charts = [
        {"values": [_safe_value(row.get("decision_timeout_rate_mean")) for row in aggregates], "title": "Decision Timeout Rate", "ylim": (0, 1), "color": "#e7298a"},
        {"values": [_safe_value(row.get("fallback_action_rate_mean")) for row in aggregates], "title": "Fallback Action Rate", "ylim": (0, 1), "color": "#666666"},
        {"values": [_safe_value(row.get("timeout_episode_rate")) for row in aggregates], "title": "Timeout Episode Rate", "ylim": (0, 1), "color": "#a6761d"},
        {"values": [_safe_value(row.get("ttc_danger_rate_mean")) for row in aggregates], "title": "TTC Danger Rate", "ylim": (0, 1), "color": "#e41a1c"},
    ]
    if energy_mode:
        charts.extend(
            [
                {"values": [_safe_value(row.get("net_energy_j_mean")) for row in aggregates], "title": "Net Energy Mean (J)", "ylim": None, "color": "#1b9e77"},
                {"values": [_safe_value(row.get("energy_per_token_j_mean")) for row in aggregates], "title": "Energy / Token (J)", "ylim": None, "color": "#7570b3"},
            ]
        )
    if latency_runtime_mode:
        charts.extend(
            [
                {"values": [_safe_value(row.get("tokens_per_second_mean")) for row in aggregates], "title": "Tokens / Second", "ylim": None, "color": "#66a61e"},
                {"values": [_safe_value(row.get("p95_decision_latency_sec_mean")) for row in aggregates], "title": "P95 Decision Latency (s)", "ylim": None, "color": "#a6761d"},
            ]
        )
    if token_metrics_mode:
        charts.extend(
            [
                {"values": [_safe_value(row.get("prompt_tokens_total_mean")) for row in aggregates], "title": "Prompt Tokens Mean", "ylim": None, "color": "#4daf4a"},
                {"values": [_safe_value(row.get("completion_tokens_total_mean")) for row in aggregates], "title": "Completion Tokens Mean", "ylim": None, "color": "#377eb8"},
                {"values": [_safe_value(row.get("total_tokens_mean")) for row in aggregates], "title": "Total Tokens Mean", "ylim": None, "color": "#984ea3"},
            ]
        )
    _plot_grid(models, charts, f"{title_prefix} (Extended Runtime/Safety)", output_path)


def emit_per_model_plots(report: dict, all_metrics: bool, extended: bool) -> list:
    """Emit one chart per model under each model's plots folder (for eval reports)."""
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
        plot_aggregates(single_report, output_path, extended=extended, all_metrics=all_metrics)
        outputs.append(output_path)
    return outputs


def update_manifest_for_plots(report: dict, global_plot_path: str, per_model_plot_paths: list) -> None:
    experiment_root = report.get("experiment_root")
    if not experiment_root:
        return

    manifest_path = os.path.join(experiment_root, "manifest.json")
    manifest = read_json(manifest_path, default={})
    compare_meta = manifest.setdefault("compare", {})
    compare_meta["latest_plot"] = global_plot_path

    plot_history = compare_meta.setdefault("plot_history", [])
    if global_plot_path not in plot_history:
        plot_history.append(global_plot_path)

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
    parser.add_argument("--extended", action="store_true", help="Plot extended runtime+safety metrics (timeout/fallback/TTC).")
    parser.add_argument("--all-metrics", action="store_true", help="Plot all available aggregate metrics in one figure.")
    parser.add_argument("--emit-per-model", action="store_true", help="Emit one plot per model under experiment model plot folders.")
    args = parser.parse_args()

    report = load_report(args.input)

    if args.output:
        output_path = args.output
    else:
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}_plot.png"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_aggregates(report, output_path, extended=args.extended, all_metrics=args.all_metrics)
    print(f"Saved plot: {output_path}")
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

    update_manifest_for_plots(report, output_path, per_model_outputs)


if __name__ == "__main__":
    main()
