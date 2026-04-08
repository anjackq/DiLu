import tempfile
import unittest
from unittest.mock import patch

from plot_eval_compare import plot_aggregates


class PlotEvalCompareTests(unittest.TestCase):
    def test_default_plot_marks_preflight_failed_models_without_ollama_panels(self):
        report = {
            "aggregates": [
                {
                    "model": "qwen3:4b",
                    "crash_rate": 0.0,
                    "no_collision_rate": 1.0,
                    "avg_steps": 30.0,
                    "avg_episode_runtime_sec": 620.0,
                    "decision_latency_ms_avg_mean": 20000.0,
                    "tokens_per_second_mean": 40.0,
                    "ollama_preflight_ok": False,
                    "ollama_openai_fallback_rate_mean": 0.7,
                    "ollama_downgrade_episode_rate": 1.0,
                }
            ]
        }
        captured = {}

        def _capture(models, charts, title, output_path):
            captured["models"] = list(models)
            captured["titles"] = [chart["title"] for chart in charts]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                plot_aggregates(report, out, extended=False, all_metrics=False)

        self.assertIn("PREFLIGHT_FAIL", captured["models"][0])
        self.assertNotIn("Ollama /v1 Fallback Rate", captured["titles"])
        self.assertNotIn("Ollama Downgrade Episode Rate", captured["titles"])

    def test_default_plot_marks_skipped_quarantined_and_incomplete_models(self):
        report = {
            "aggregates": [
                {
                    "model": "phi4-mini:3.8b",
                    "crash_rate": 0.0,
                    "no_collision_rate": 1.0,
                    "avg_steps": 5.0,
                    "avg_episode_runtime_sec": 120.0,
                    "decision_latency_ms_avg_mean": 30000.0,
                    "tokens_per_second_mean": 0.0,
                    "model_skipped_due_to_preflight": True,
                    "model_quarantined_due_to_timeout_collapse": True,
                    "episode_execution_complete": False,
                }
            ]
        }
        captured = {}

        def _capture(models, charts, title, output_path):
            captured["models"] = list(models)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                plot_aggregates(report, out, extended=False, all_metrics=False)

        self.assertIn("PREFLIGHT_SKIP", captured["models"][0])
        self.assertIn("QUARANTINED", captured["models"][0])
        self.assertIn("INCOMPLETE", captured["models"][0])

    def test_latency_only_benchmark_still_emits_latency_runtime_charts(self):
        report = {
            "aggregates": [
                {
                    "model": "qwen3:1.7b",
                    "task_completion_rate": 0.1,
                    "driving_score": 0.06,
                    "ttc_score_mean": 1.0,
                    "speed_variance_score_mean": 0.2,
                    "time_efficiency_score_mean": 0.1,
                    "overall_score_mean": 0.5,
                    "stop_episode_rate": 1.0,
                    "near_stop_rate_mean": 0.6,
                    "net_energy_j_mean": None,
                    "energy_per_decision_j_mean": None,
                    "tokens_per_second_mean": 120.0,
                    "decision_latency_ms_avg_mean": 1700.0,
                    "p95_decision_latency_sec_mean": 1.4,
                }
            ]
        }
        captured = {}

        def _capture(models, charts, title, output_path):
            captured["models"] = list(models)
            captured["titles"] = [chart["title"] for chart in charts]
            captured["title"] = title
            captured["output_path"] = output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                plot_aggregates(report, out, extended=True, all_metrics=False)

        self.assertIn("Tokens / Second", captured["titles"])
        self.assertIn("Decision Latency Mean (ms)", captured["titles"])

    def test_benchmark_v2_headlines_driving_score_v2_and_keeps_legacy_in_extended(self):
        report = {
            "aggregates": [
                {
                    "model": "qwen3:1.7b",
                    "task_completion_rate": 0.3,
                    "driving_score": 0.19,
                    "driving_score_v2": 0.08,
                    "ttc_score_mean": 0.94,
                    "speed_variance_score_mean": 0.02,
                    "time_efficiency_score_mean": 0.13,
                    "overall_score_mean": 0.5,
                    "overall_score_v2_mean": 0.22,
                    "stop_episode_rate": 0.55,
                    "near_stop_rate_mean": 0.24,
                    "behavior_penalty_factor_v2_mean": 0.42,
                }
            ]
        }
        captured = {}

        def _capture(models, charts, title, output_path):
            captured["titles"] = [chart["title"] for chart in charts]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                plot_aggregates(report, out, extended=True, all_metrics=False)

        self.assertIn("Driving Score v2", captured["titles"])
        self.assertIn("Driving Score (legacy)", captured["titles"])

    def test_all_metrics_plot_includes_token_mean_charts_when_present(self):
        report = {
            "aggregates": [
                {
                    "model": "qwen3:1.7b",
                    "task_completion_rate": 0.3,
                    "driving_score": 0.19,
                    "driving_score_v2": 0.08,
                    "tokens_per_second_mean": 12.0,
                    "decision_latency_ms_avg_mean": 1700.0,
                    "prompt_tokens_total_mean": 150.0,
                    "completion_tokens_total_mean": 220.0,
                    "total_tokens_mean": 370.0,
                }
            ]
        }
        captured = {}

        def _capture(models, charts, title, output_path):
            captured["titles"] = [chart["title"] for chart in charts]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                plot_aggregates(report, out, extended=False, all_metrics=True)

        self.assertIn("Prompt Tokens Mean", captured["titles"])
        self.assertIn("Completion Tokens Mean", captured["titles"])
        self.assertIn("Total Tokens Mean", captured["titles"])

    def test_all_metrics_plot_omits_ollama_diagnostic_charts(self):
        report = {
            "aggregates": [
                {
                    "model": "qwen3:1.7b",
                    "task_completion_rate": 0.3,
                    "driving_score": 0.19,
                    "driving_score_v2": 0.08,
                    "tokens_per_second_mean": 12.0,
                    "decision_latency_ms_avg_mean": 1700.0,
                    "ollama_native_retry_rate_mean": 0.5,
                    "ollama_openai_fallback_rate_mean": 0.2,
                    "ollama_native_decision_rate_mean": 0.4,
                    "ollama_downgrade_episode_rate": 0.3,
                }
            ]
        }
        captured = {}

        def _capture(models, charts, title, output_path):
            captured["titles"] = [chart["title"] for chart in charts]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = f"{tmpdir}\\plot.png"
            with patch("plot_eval_compare._plot_grid", side_effect=_capture):
                plot_aggregates(report, out, extended=False, all_metrics=True)

        self.assertNotIn("Ollama Native Retry Rate", captured["titles"])
        self.assertNotIn("Ollama /v1 Fallback Rate", captured["titles"])
        self.assertNotIn("Ollama Native Decision Rate", captured["titles"])
        self.assertNotIn("Ollama Downgrade Episode Rate", captured["titles"])


if __name__ == "__main__":
    unittest.main()
