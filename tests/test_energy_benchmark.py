import json
import os
import tempfile
import unittest

from dilu.runtime.energy_monitor import (
    TOKEN_COUNT_METHOD,
    build_energy_tradeoff_summary,
    create_energy_monitor,
    enrich_episode_energy_metrics,
    estimate_generated_tokens,
    load_idle_calibration,
    save_idle_calibration,
    summarize_energy_latency_episodes,
)


class EnergyBenchmarkTests(unittest.TestCase):
    def test_estimate_generated_tokens_uses_whitespace_count(self):
        self.assertEqual(estimate_generated_tokens(""), 0)
        self.assertEqual(estimate_generated_tokens("Reply with exactly 4"), 4)

    def test_idle_calibration_round_trip(self):
        artifact = {
            "created_at": "2026-03-31T10:00:00",
            "energy_mode": "joulescope_hw",
            "duration_sec": 60.0,
            "avg_idle_power_w": 2.85,
            "std_idle_power_w": 0.12,
            "sample_count": 120,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "idle.json")
            save_idle_calibration(path, artifact)
            loaded = load_idle_calibration(path)
        self.assertEqual(loaded["avg_idle_power_w"], 2.85)
        self.assertEqual(loaded["energy_mode"], "joulescope_hw")

    def test_latency_only_monitor_is_constructible(self):
        monitor = create_energy_monitor("latency_only")
        self.assertEqual(monitor.mode, "latency_only")
        self.assertIsNone(monitor.stop_episode()["raw_energy_j"])

    def test_enrich_episode_energy_metrics_computes_baseline_and_rates(self):
        episode = {
            "episode_runtime_sec": 10.0,
            "decisions_made": 5,
        }
        enriched = enrich_episode_energy_metrics(
            episode,
            energy_mode="joulescope_hw",
            raw_energy_j=50.0,
            idle_power_w=2.0,
            avg_power_w=5.0,
            peak_power_w=7.0,
            prompt_tokens_total=15,
            completion_tokens_total=25,
            total_tokens=40,
            token_count_method="ollama_native_usage",
            token_usage_source="native_api",
            latency_to_first_action_sec=1.2,
        )
        self.assertEqual(enriched["idle_baseline_energy_j"], 20.0)
        self.assertEqual(enriched["net_energy_j"], 30.0)
        self.assertEqual(enriched["energy_per_decision_j"], 6.0)
        self.assertEqual(enriched["energy_per_token_j"], 1.2)
        self.assertEqual(enriched["tokens_per_second"], 2.5)
        self.assertEqual(enriched["prompt_tokens_total"], 15)
        self.assertEqual(enriched["completion_tokens_total"], 25)
        self.assertEqual(enriched["total_tokens"], 40)
        self.assertEqual(enriched["tokens_generated_total"], 25)
        self.assertEqual(enriched["token_count_method"], "ollama_native_usage")
        self.assertEqual(enriched["token_usage_source"], "native_api")

    def test_summarize_energy_latency_episodes_returns_means(self):
        episodes = [
            {
                "energy_mode": "latency_only",
                "episode_runtime_sec": 10.0,
                "decision_latency_ms_avg": 1000.0,
                "p95_decision_latency_sec": 2.0,
                "latency_to_first_action_sec": 1.0,
                "raw_energy_j": None,
                "idle_baseline_energy_j": None,
                "net_energy_j": None,
                "avg_power_w": None,
                "peak_power_w": None,
                "energy_per_decision_j": None,
                "energy_per_token_j": None,
                "prompt_tokens_total": 10,
                "completion_tokens_total": 20,
                "total_tokens": 30,
                "tokens_generated_total": 20,
                "tokens_per_second": 2.0,
                "token_count_method": "ollama_native_usage",
                "token_usage_source": "native_api",
            },
            {
                "energy_mode": "latency_only",
                "episode_runtime_sec": 20.0,
                "decision_latency_ms_avg": 1500.0,
                "p95_decision_latency_sec": 3.0,
                "latency_to_first_action_sec": 1.5,
                "raw_energy_j": None,
                "idle_baseline_energy_j": None,
                "net_energy_j": None,
                "avg_power_w": None,
                "peak_power_w": None,
                "energy_per_decision_j": None,
                "energy_per_token_j": None,
                "prompt_tokens_total": 15,
                "completion_tokens_total": 30,
                "total_tokens": 45,
                "tokens_generated_total": 30,
                "tokens_per_second": 1.5,
                "token_count_method": "ollama_native_usage",
                "token_usage_source": "native_api",
            },
        ]
        summary = summarize_energy_latency_episodes(episodes)
        self.assertEqual(summary["energy_mode"], "latency_only")
        self.assertEqual(summary["episode_runtime_sec_mean"], 15.0)
        self.assertEqual(summary["decision_latency_ms_avg_mean"], 1250.0)
        self.assertEqual(summary["prompt_tokens_total_mean"], 12.5)
        self.assertEqual(summary["completion_tokens_total_mean"], 25.0)
        self.assertEqual(summary["total_tokens_mean"], 37.5)
        self.assertEqual(summary["tokens_generated_total_mean"], 25.0)
        self.assertEqual(summary["tokens_per_second_mean"], 1.75)

    def test_build_energy_tradeoff_summary_emits_expected_points(self):
        aggregates = [
            {
                "model": "m1",
                "net_energy_j_mean": 12.0,
                "energy_per_decision_j_mean": 3.0,
                "decision_latency_ms_avg_mean": 900.0,
                "tokens_per_second_mean": 2.5,
                "crash_rate": 0.1,
                "task_completion_rate": 0.8,
                "driving_score": 0.7,
                "driving_score_v2": 0.55,
            }
        ]
        summary = build_energy_tradeoff_summary(aggregates)
        self.assertEqual(len(summary["points"]), 1)
        self.assertEqual(summary["points"][0]["model"], "m1")
        self.assertEqual(summary["points"][0]["driving_score_v2"], 0.55)
        self.assertEqual(summary["headline_task_metric"], "driving_score_v2")
        self.assertTrue(summary["efficiency_metrics_reported"])
        self.assertEqual(summary["pareto_objectives"]["maximize"][0], "driving_score_v2")
        self.assertIn("net_energy_j_mean", summary["pareto_objectives"]["minimize"])


if __name__ == "__main__":
    unittest.main()
