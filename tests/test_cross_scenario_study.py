import csv
import json
import os
import tempfile
import unittest

from analysis.cross_scenario_study import run_cross_scenario_study


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _agg(
    model: str,
    *,
    headline_value: float,
    headline_metric: str,
    completion: float,
    crash_rate: float,
    latency: float,
    valid: bool = True,
    invalid_reason: str = "",
    quarantined: bool = False,
) -> dict:
    payload = {
        "model": model,
        "task_completion_rate": completion,
        "crash_rate": crash_rate,
        "ttc_score_mean": 0.8,
        "time_efficiency_score_mean": 0.7,
        "decision_latency_ms_avg_mean": latency,
        "benchmark_result_valid": valid,
        "benchmark_result_invalid_reason": invalid_reason,
        "model_skipped_due_to_preflight": False,
        "model_quarantined_due_to_timeout_collapse": quarantined,
        "episode_execution_complete": valid and not quarantined,
    }
    payload[headline_metric] = headline_value
    if headline_metric != "driving_score_v2":
        payload["driving_score_v2"] = None
    if headline_metric != "driving_score":
        payload["driving_score"] = headline_value
    return payload


class CrossScenarioStudyTests(unittest.TestCase):
    def test_cross_scenario_study_outputs_scenario_leaderboards_and_pair_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "registry.csv")
            with open(registry_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "model_id",
                        "display_name",
                        "ollama_tag",
                        "family",
                        "variant_kind",
                        "base_model_id",
                        "param_count_b",
                        "tier",
                        "notes",
                    ]
                )
                writer.writerow(
                    ["qwen3_8b_base", "Qwen3 8B", "qwen3:8b", "qwen", "base", "", "8.0", "midclass", ""]
                )
                writer.writerow(
                    [
                        "dilu_qwen3_8b_v1",
                        "DiLu Qwen3 8B v1",
                        "dilu-qwen3-8b-v1",
                        "qwen",
                        "fine_tuned",
                        "qwen3_8b_base",
                        "8.0",
                        "midclass",
                        "",
                    ]
                )

            highway_path = os.path.join(tmpdir, "highway.json")
            merge_path = os.path.join(tmpdir, "merge.json")
            intersection_path = os.path.join(tmpdir, "intersection.json")

            _write_json(
                highway_path,
                {
                    "benchmark_mode": True,
                    "experiment_id": "highway_exp",
                    "benchmark_case_set": "lampilot_highway_v1",
                    "headline_task_metric": "driving_score_v2",
                    "aggregates": [
                        _agg(
                            "qwen3:8b",
                            headline_value=0.08,
                            headline_metric="driving_score_v2",
                            completion=0.20,
                            crash_rate=0.4,
                            latency=2400.0,
                        ),
                        _agg(
                            "dilu-qwen3-8b-v1",
                            headline_value=0.11,
                            headline_metric="driving_score_v2",
                            completion=0.30,
                            crash_rate=0.2,
                            latency=2100.0,
                        ),
                    ],
                },
            )
            _write_json(
                merge_path,
                {
                    "benchmark_mode": True,
                    "experiment_id": "merge_exp",
                    "benchmark_case_set": "lampilot_merge_v1",
                    "headline_task_metric": "driving_score",
                    "aggregates": [
                        _agg(
                            "qwen3:8b",
                            headline_value=0.35,
                            headline_metric="driving_score",
                            completion=0.60,
                            crash_rate=0.1,
                            latency=2500.0,
                        ),
                        _agg(
                            "dilu-qwen3-8b-v1",
                            headline_value=0.48,
                            headline_metric="driving_score",
                            completion=0.75,
                            crash_rate=0.05,
                            latency=2200.0,
                        ),
                    ],
                },
            )
            _write_json(
                intersection_path,
                {
                    "benchmark_mode": True,
                    "experiment_id": "intersection_exp",
                    "benchmark_case_set": "lampilot_intersection_v1",
                    "headline_task_metric": "driving_score",
                    "aggregates": [
                        _agg(
                            "qwen3:8b",
                            headline_value=0.28,
                            headline_metric="driving_score",
                            completion=0.55,
                            crash_rate=0.15,
                            latency=2600.0,
                        ),
                        _agg(
                            "dilu-qwen3-8b-v1",
                            headline_value=0.0,
                            headline_metric="driving_score",
                            completion=0.10,
                            crash_rate=0.0,
                            latency=25000.0,
                            valid=False,
                            invalid_reason="timeout_collapse",
                            quarantined=True,
                        ),
                    ],
                },
            )

            result = run_cross_scenario_study(
                registry_path=registry_path,
                scenario_reports=[
                    f"highway={highway_path}",
                    f"merge={merge_path}",
                    f"intersection={intersection_path}",
                ],
                output_root=tmpdir,
                study_id="cross_scenario_demo",
            )

            study_dir = result["study_dir"]
            self.assertTrue(os.path.exists(os.path.join(study_dir, "normalized_records.csv")))
            self.assertTrue(os.path.exists(os.path.join(study_dir, "scenario_leaderboard_highway.csv")))
            self.assertTrue(os.path.exists(os.path.join(study_dir, "scenario_leaderboard_merge.csv")))
            self.assertTrue(os.path.exists(os.path.join(study_dir, "scenario_leaderboard_intersection.csv")))
            self.assertTrue(os.path.exists(os.path.join(study_dir, "robustness_matrix.csv")))
            self.assertTrue(os.path.exists(os.path.join(study_dir, "paired_deltas_by_scenario.csv")))
            self.assertTrue(os.path.exists(os.path.join(study_dir, "invalid_runs.csv")))
            self.assertTrue(os.path.exists(os.path.join(study_dir, "study_report.md")))

            highway_rows = _read_csv(os.path.join(study_dir, "scenario_leaderboard_highway.csv"))
            self.assertEqual(highway_rows[0]["display_name"], "DiLu Qwen3 8B v1")
            self.assertEqual(highway_rows[0]["headline_task_metric"], "driving_score_v2")

            robustness_rows = _read_csv(os.path.join(study_dir, "robustness_matrix.csv"))
            robustness_by_model = {row["model_id"]: row for row in robustness_rows}
            self.assertEqual(robustness_by_model["qwen3_8b_base"]["valid_scenario_count"], "3")
            self.assertEqual(robustness_by_model["dilu_qwen3_8b_v1"]["valid_scenario_count"], "2")

            paired_rows = _read_csv(os.path.join(study_dir, "paired_deltas_by_scenario.csv"))
            eligible_by_scenario = {
                row["scenario_name"]: row
                for row in paired_rows
                if row["pair_eligible"].lower() == "true"
            }
            self.assertAlmostEqual(float(eligible_by_scenario["highway"]["delta_headline_task_value"]), 0.03, places=6)
            self.assertAlmostEqual(float(eligible_by_scenario["merge"]["delta_headline_task_value"]), 0.13, places=6)
            self.assertNotIn("intersection", eligible_by_scenario)

            invalid_rows = _read_csv(os.path.join(study_dir, "invalid_runs.csv"))
            self.assertEqual(len(invalid_rows), 1)
            self.assertEqual(invalid_rows[0]["scenario_name"], "intersection")


if __name__ == "__main__":
    unittest.main()
