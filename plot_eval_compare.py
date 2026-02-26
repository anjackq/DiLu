import argparse
import json
import os

import matplotlib.pyplot as plt


def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_value(v, default=0.0):
    return default if v is None else v


def plot_aggregates(report: dict, output_path: str) -> None:
    aggregates = report.get("aggregates", [])
    if not aggregates:
        raise ValueError("No 'aggregates' section found in report.")

    models = [row["model"] for row in aggregates]
    crash_rates = [_safe_value(row.get("crash_rate")) for row in aggregates]
    no_collision_rates = [_safe_value(row.get("no_collision_rate")) for row in aggregates]
    avg_steps = [_safe_value(row.get("avg_steps")) for row in aggregates]
    avg_episode_runtime = [_safe_value(row.get("avg_episode_runtime_sec")) for row in aggregates]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DiLu Model Comparison", fontsize=14, fontweight="bold")

    charts = [
        (axes[0][0], crash_rates, "Crash Rate", (0, 1), "#d95f02"),
        (axes[0][1], no_collision_rates, "No-Collision Rate", (0, 1), "#1b9e77"),
        (axes[1][0], avg_steps, "Average Steps", None, "#7570b3"),
        (axes[1][1], avg_episode_runtime, "Avg Episode Runtime (s)", None, "#66a61e"),
    ]

    for ax, values, title, ylim, color in charts:
        bars = ax.bar(models, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xticklabels(models, rotation=20, ha="right")
        if ylim is not None:
            ax.set_ylim(*ylim)
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

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregate metrics from evaluate_models_ollama.py JSON output.")
    parser.add_argument("-i", "--input", required=True, help="Path to comparison JSON report")
    parser.add_argument("-o", "--output", default=None, help="Output image path (PNG). Defaults next to input file.")
    args = parser.parse_args()

    report = load_report(args.input)

    if args.output:
        output_path = args.output
    else:
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}_plot.png"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plot_aggregates(report, output_path)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
