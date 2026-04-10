import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from analysis.slm_study import load_registry, run_study


def _write_csv(path: str, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _registry_fieldnames():
    return [
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


def _base_registry_rows():
    return [
        {
            "model_id": "qwen3_1_7b_base",
            "display_name": "Qwen3 1.7B Base",
            "ollama_tag": "qwen3:1.7b",
            "family": "qwen",
            "variant_kind": "base",
            "base_model_id": "",
            "param_count_b": "1.7",
            "tier": "lightweight",
            "notes": "",
        },
        {
            "model_id": "dilu_qwen3_1_7b_v1",
            "display_name": "DiLu Qwen3 1.7B v1",
            "ollama_tag": "dilu-qwen3-1_7b-v1",
            "family": "qwen",
            "variant_kind": "fine_tuned",
            "base_model_id": "qwen3_1_7b_base",
            "param_count_b": "1.7",
            "tier": "lightweight",
            "notes": "",
        },
        {
            "model_id": "llama3_2_3b_base",
            "display_name": "Llama 3.2 3B",
            "ollama_tag": "llama3.2:3b",
            "family": "llama",
            "variant_kind": "base",
            "base_model_id": "",
            "param_count_b": "3.0",
            "tier": "lightweight",
            "notes": "",
        },
        {
            "model_id": "qwen2_5_7b_base",
            "display_name": "Qwen2.5 7B Base",
            "ollama_tag": "qwen2.5:7b",
            "family": "qwen",
            "variant_kind": "base",
            "base_model_id": "",
            "param_count_b": "7.0",
            "tier": "midclass",
            "notes": "",
        },
        {
            "model_id": "dilu_qwen2_5_7b_v1",
            "display_name": "DiLu Qwen2.5 7B v1",
            "ollama_tag": "dilu-qwen2.5-7b-v1",
            "family": "qwen",
            "variant_kind": "fine_tuned",
            "base_model_id": "qwen2_5_7b_base",
            "param_count_b": "7.0",
            "tier": "midclass",
            "notes": "",
        },
        {
            "model_id": "llama3_8b_base",
            "display_name": "Llama 3 8B",
            "ollama_tag": "llama3:8b",
            "family": "llama",
            "variant_kind": "base",
            "base_model_id": "",
            "param_count_b": "8.0",
            "tier": "midclass",
            "notes": "",
        },
        {
            "model_id": "qwen2_5_14b_base",
            "display_name": "Qwen2.5 14B Base",
            "ollama_tag": "qwen2.5:14b",
            "family": "qwen",
            "variant_kind": "base",
            "base_model_id": "",
            "param_count_b": "14.0",
            "tier": "highclass",
            "notes": "",
        },
        {
            "model_id": "dilu_qwen2_5_14b_v1",
            "display_name": "DiLu Qwen2.5 14B v1",
            "ollama_tag": "dilu-qwen2.5-14b-v1",
            "family": "qwen",
            "variant_kind": "fine_tuned",
            "base_model_id": "qwen2_5_14b_base",
            "param_count_b": "14.0",
            "tier": "highclass",
            "notes": "",
        },
        {
            "model_id": "llama3_1_32b_base",
            "display_name": "Llama 3.1 32B",
            "ollama_tag": "llama3.1:32b",
            "family": "llama",
            "variant_kind": "base",
            "base_model_id": "",
            "param_count_b": "32.0",
            "tier": "highclass",
            "notes": "",
        },
    ]


def _agg(
    model,
    *,
    ds,
    tc,
    overall,
    latency,
    p95,
    tps,
    crash_rate=0.0,
    valid=True,
    skipped=False,
    quarantined=False,
    incomplete=False,
    invalid_reason=None,
    net_energy=None,
    energy_per_decision=None,
):
    return {
        "model": model,
        "crash_rate": crash_rate,
        "task_completion_rate": tc,
        "overall_score_mean": overall,
        "driving_score_v2": ds,
        "decision_latency_ms_avg_mean": latency,
        "p95_decision_latency_sec_mean": p95,
        "tokens_per_second_mean": tps,
        "benchmark_result_valid": valid,
        "benchmark_result_invalid_reason": invalid_reason,
        "model_skipped_due_to_preflight": skipped,
        "model_quarantined_due_to_timeout_collapse": quarantined,
        "episode_execution_complete": not incomplete,
        "net_energy_j_mean": net_energy,
        "energy_per_decision_j_mean": energy_per_decision,
    }


def _screening_report():
    return {
        "benchmark_mode": True,
        "benchmark_case_set": "lampilot_highway_v1",
        "headline_task_metric": "driving_score_v2",
        "efficiency_metrics_reported": True,
        "aggregates": [
            _agg("qwen3:1.7b", ds=0.40, tc=0.70, overall=0.60, latency=900.0, p95=1.2, tps=90.0),
            _agg("dilu-qwen3-1_7b-v1", ds=0.55, tc=0.80, overall=0.68, latency=1100.0, p95=1.5, tps=80.0),
            _agg(
                "llama3.2:3b",
                ds=0.50,
                tc=0.78,
                overall=0.64,
                latency=700.0,
                p95=1.0,
                tps=120.0,
            ),
            _agg(
                "qwen2.5:7b",
                ds=0.62,
                tc=0.86,
                overall=0.70,
                latency=1500.0,
                p95=1.9,
                tps=65.0,
            ),
            _agg(
                "dilu-qwen2.5-7b-v1",
                ds=0.69,
                tc=0.88,
                overall=0.75,
                latency=1700.0,
                p95=2.1,
                tps=58.0,
            ),
            _agg(
                "llama3:8b",
                ds=0.65,
                tc=0.87,
                overall=0.73,
                latency=1400.0,
                p95=1.8,
                tps=70.0,
            ),
            _agg(
                "qwen2.5:14b",
                ds=0.73,
                tc=0.90,
                overall=0.80,
                latency=2200.0,
                p95=2.5,
                tps=40.0,
            ),
            _agg(
                "dilu-qwen2.5-14b-v1",
                ds=0.80,
                tc=0.92,
                overall=0.85,
                latency=2500.0,
                p95=2.9,
                tps=35.0,
            ),
            _agg(
                "llama3.1:32b",
                ds=0.78,
                tc=0.91,
                overall=0.83,
                latency=3000.0,
                p95=3.4,
                tps=28.0,
            ),
            _agg(
                "qwen3.5:2b",
                ds=0.61,
                tc=0.82,
                overall=0.69,
                latency=850.0,
                p95=1.1,
                tps=95.0,
                valid=False,
                invalid_reason="decision_timeout_collapse",
                incomplete=True,
            ),
        ],
    }


def _finalist_energy_report():
    return {
        "benchmark_mode": True,
        "benchmark_case_set": "lampilot_highway_v1",
        "headline_task_metric": "driving_score_v2",
        "efficiency_metrics_reported": True,
        "energy_mode_effective": "joulescope_hw",
        "aggregates": [
            _agg("dilu-qwen3-1_7b-v1", ds=0.55, tc=0.80, overall=0.68, latency=1080.0, p95=1.5, tps=78.0, net_energy=12.0, energy_per_decision=0.9),
            _agg("llama3.2:3b", ds=0.50, tc=0.78, overall=0.64, latency=690.0, p95=1.0, tps=121.0, net_energy=10.0, energy_per_decision=0.7),
            _agg("dilu-qwen2.5-7b-v1", ds=0.69, tc=0.88, overall=0.75, latency=1690.0, p95=2.1, tps=57.0, net_energy=20.0, energy_per_decision=1.3),
            _agg("llama3:8b", ds=0.65, tc=0.87, overall=0.73, latency=1410.0, p95=1.8, tps=69.0, net_energy=18.0, energy_per_decision=1.1),
            _agg("dilu-qwen2.5-14b-v1", ds=0.80, tc=0.92, overall=0.85, latency=2480.0, p95=2.9, tps=34.0, net_energy=35.0, energy_per_decision=2.0),
            _agg("llama3.1:32b", ds=0.78, tc=0.91, overall=0.83, latency=2990.0, p95=3.4, tps=27.0, net_energy=40.0, energy_per_decision=2.4),
        ],
    }


class SlmStudyTests(unittest.TestCase):
    def test_load_registry_rejects_tier_param_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.csv")
            rows = _base_registry_rows()
            rows[0]["tier"] = "midclass"
            _write_csv(path, _registry_fieldnames(), rows)

            with self.assertRaisesRegex(ValueError, "tier"):
                load_registry(path)

    def test_load_registry_rejects_fine_tuned_row_without_exact_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.csv")
            rows = _base_registry_rows()
            rows[1]["base_model_id"] = ""
            _write_csv(path, _registry_fieldnames(), rows)

            with self.assertRaisesRegex(ValueError, "base_model_id"):
                load_registry(path)

    def test_run_study_rejects_compare_model_missing_from_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "registry.csv")
            report_path = os.path.join(tmpdir, "report.json")
            _write_csv(registry_path, _registry_fieldnames(), _base_registry_rows())
            payload = _screening_report()
            payload["aggregates"] = [_agg("missing:model", ds=0.2, tc=0.4, overall=0.5, latency=1000, p95=1.5, tps=50.0)]
            _write_json(report_path, payload)

            with self.assertRaisesRegex(ValueError, "missing:model"):
                run_study(
                    registry_path=registry_path,
                    compare_report_paths=[report_path],
                    output_root=os.path.join(tmpdir, "out"),
                    study_id="study",
                )

    def test_run_study_emits_outputs_and_shortlist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "registry.csv")
            report_path = os.path.join(tmpdir, "screening.json")
            output_root = os.path.join(tmpdir, "out")
            _write_csv(registry_path, _registry_fieldnames(), _base_registry_rows())
            _write_json(report_path, _screening_report())

            result = run_study(
                registry_path=registry_path,
                compare_report_paths=[report_path],
                output_root=output_root,
                study_id="study_a",
            )

            study_dir = Path(result["study_dir"])
            self.assertTrue((study_dir / "normalized_records.csv").exists())
            self.assertTrue((study_dir / "tier_leaderboard_lightweight.csv").exists())
            self.assertTrue((study_dir / "tier_leaderboard_midclass.csv").exists())
            self.assertTrue((study_dir / "tier_leaderboard_highclass.csv").exists())
            self.assertTrue((study_dir / "paired_deltas.csv").exists())
            self.assertTrue((study_dir / "family_summary.csv").exists())
            self.assertTrue((study_dir / "stage1_shortlist.csv").exists())
            self.assertTrue((study_dir / "invalid_runs.csv").exists())
            self.assertTrue((study_dir / "study_report.md").exists())
            self.assertTrue((study_dir / "issues.json").exists())
            self.assertTrue((study_dir / "plots" / "tier_leaderboard_lightweight.png").exists())
            self.assertTrue((study_dir / "plots" / "paired_deltas.png").exists())
            self.assertTrue((study_dir / "plots" / "pareto_lightweight_latency.png").exists())
            self.assertTrue((study_dir / "plots" / "finalist_energy_pareto.png").exists())

            with open(study_dir / "stage1_shortlist.csv", "r", encoding="utf-8", newline="") as handle:
                shortlist = list(csv.DictReader(handle))
            self.assertEqual(len(shortlist), 6)
            by_tier = {}
            for row in shortlist:
                by_tier.setdefault(row["tier"], []).append(row["model_id"])
            self.assertEqual(by_tier["lightweight"], ["dilu_qwen3_1_7b_v1", "llama3_2_3b_base"])
            self.assertEqual(by_tier["midclass"], ["dilu_qwen2_5_7b_v1", "llama3_8b_base"])
            self.assertEqual(by_tier["highclass"], ["dilu_qwen2_5_14b_v1", "llama3_1_32b_base"])

            with open(study_dir / "paired_deltas.csv", "r", encoding="utf-8", newline="") as handle:
                deltas = list(csv.DictReader(handle))
            qwen_light = next(row for row in deltas if row["fine_tuned_model_id"] == "dilu_qwen3_1_7b_v1")
            self.assertEqual(qwen_light["pair_eligible"], "true")
            self.assertAlmostEqual(float(qwen_light["delta_driving_score_v2"]), 0.15, places=6)

            with open(study_dir / "invalid_runs.csv", "r", encoding="utf-8", newline="") as handle:
                invalid = list(csv.DictReader(handle))
            self.assertEqual(len(invalid), 1)
            self.assertEqual(invalid[0]["ollama_tag"], "qwen3.5:2b")

    def test_run_study_augments_finalist_energy_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "registry.csv")
            screening_path = os.path.join(tmpdir, "screening.json")
            finalist_path = os.path.join(tmpdir, "finalists.json")
            output_root = os.path.join(tmpdir, "out")
            _write_csv(registry_path, _registry_fieldnames(), _base_registry_rows())
            _write_json(screening_path, _screening_report())
            _write_json(finalist_path, _finalist_energy_report())

            result = run_study(
                registry_path=registry_path,
                compare_report_paths=[screening_path],
                finalist_energy_report_paths=[finalist_path],
                output_root=output_root,
                study_id="study_b",
            )

            study_dir = Path(result["study_dir"])
            with open(study_dir / "normalized_records.csv", "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            screening_rows = [row for row in rows if row["study_stage"] == "screening"]
            finalist_rows = [row for row in rows if row["study_stage"] == "finalist_energy"]
            self.assertEqual(len(screening_rows), 10)
            self.assertEqual(len(finalist_rows), 6)
            finalist_qwen = next(row for row in finalist_rows if row["model_id"] == "dilu_qwen2_5_14b_v1")
            self.assertAlmostEqual(float(finalist_qwen["net_energy_j_mean"]), 35.0, places=6)

    def test_run_study_refresh_supersedes_invalid_rows_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "registry.csv")
            base_path = os.path.join(tmpdir, "screening_base.json")
            refresh_path = os.path.join(tmpdir, "screening_refresh.json")
            output_root = os.path.join(tmpdir, "out")
            _write_csv(registry_path, _registry_fieldnames(), _base_registry_rows())
            _write_json(
                base_path,
                {
                    "benchmark_mode": True,
                    "benchmark_case_set": "lampilot_highway_v1",
                    "headline_task_metric": "driving_score_v2",
                    "efficiency_metrics_reported": True,
                    "aggregates": [
                        _agg(
                            "qwen3:1.7b",
                            ds=0.0,
                            tc=0.0,
                            overall=0.5,
                            latency=4000.0,
                            p95=8.0,
                            tps=50.0,
                            valid=False,
                            invalid_reason="timeout_collapse",
                            quarantined=True,
                            incomplete=True,
                        ),
                        _agg(
                            "dilu-qwen3-1_7b-v1",
                            ds=0.55,
                            tc=0.80,
                            overall=0.68,
                            latency=1100.0,
                            p95=1.5,
                            tps=80.0,
                        ),
                        _agg(
                            "llama3.2:3b",
                            ds=0.50,
                            tc=0.78,
                            overall=0.64,
                            latency=700.0,
                            p95=1.0,
                            tps=120.0,
                        ),
                    ],
                },
            )
            _write_json(
                refresh_path,
                {
                    "benchmark_mode": True,
                    "benchmark_case_set": "lampilot_highway_v1",
                    "headline_task_metric": "driving_score_v2",
                    "efficiency_metrics_reported": True,
                    "aggregates": [
                        _agg(
                            "qwen3:1.7b",
                            ds=0.44,
                            tc=0.74,
                            overall=0.61,
                            latency=1350.0,
                            p95=1.7,
                            tps=82.0,
                        ),
                        _agg(
                            "llama3.2:3b",
                            ds=0.0,
                            tc=0.0,
                            overall=0.5,
                            latency=5000.0,
                            p95=10.0,
                            tps=30.0,
                            valid=False,
                            invalid_reason="timeout_collapse",
                            quarantined=True,
                            incomplete=True,
                        ),
                    ],
                },
            )

            result = run_study(
                registry_path=registry_path,
                compare_report_paths=[base_path],
                refresh_compare_report_paths=[refresh_path],
                output_root=output_root,
                study_id="study_refresh",
            )

            study_dir = Path(result["study_dir"])
            with open(study_dir / "normalized_records.csv", "r", encoding="utf-8", newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row["study_stage"] == "screening"]

            self.assertEqual(len(rows), 3)
            qwen_base = next(row for row in rows if row["model_id"] == "qwen3_1_7b_base")
            llama_base = next(row for row in rows if row["model_id"] == "llama3_2_3b_base")
            self.assertEqual(qwen_base["benchmark_result_valid"], "true")
            self.assertAlmostEqual(float(qwen_base["driving_score_v2"]), 0.44, places=6)
            self.assertEqual(qwen_base["source_compare_report"], str(Path(refresh_path).resolve()))
            self.assertEqual(llama_base["benchmark_result_valid"], "true")
            self.assertAlmostEqual(float(llama_base["driving_score_v2"]), 0.50, places=6)
            self.assertEqual(llama_base["source_compare_report"], str(Path(base_path).resolve()))
            self.assertEqual(result["refresh_records"], 2)

    def test_run_study_report_includes_quality_gate_and_refresh_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = os.path.join(tmpdir, "registry.csv")
            report_path = os.path.join(tmpdir, "screening.json")
            refresh_path = os.path.join(tmpdir, "refresh.json")
            output_root = os.path.join(tmpdir, "out")
            _write_csv(registry_path, _registry_fieldnames(), _base_registry_rows())
            _write_json(report_path, _screening_report())
            _write_json(
                refresh_path,
                {
                    "benchmark_mode": True,
                    "benchmark_case_set": "lampilot_highway_v1",
                    "headline_task_metric": "driving_score_v2",
                    "efficiency_metrics_reported": True,
                    "aggregates": [
                        _agg(
                            "qwen3:1.7b",
                            ds=0.41,
                            tc=0.72,
                            overall=0.60,
                            latency=980.0,
                            p95=1.3,
                            tps=90.0,
                        ),
                    ],
                },
            )

            result = run_study(
                registry_path=registry_path,
                compare_report_paths=[report_path],
                refresh_compare_report_paths=[refresh_path],
                output_root=output_root,
                study_id="study_quality",
            )

            study_dir = Path(result["study_dir"])
            report_text = (study_dir / "study_report.md").read_text(encoding="utf-8")
            self.assertIn("## Study Quality", report_text)
            self.assertIn("## Acceptance Gate", report_text)
            self.assertIn("## Refresh Merge", report_text)
            self.assertIn("screening-quality", report_text)
            self.assertIn("## Incomplete Family Conclusions", report_text)
            self.assertIn("## Remaining Invalid Models", report_text)
            self.assertEqual(result["study_quality"], "screening-quality")


if __name__ == "__main__":
    unittest.main()
