import csv
import os
import tempfile
import unittest
from pathlib import Path

from analysis.publication_three_tier_results import run_publication_bundle


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class PublicationThreeTierResultsTests(unittest.TestCase):
    def test_run_publication_bundle_emits_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            compare_dir = root / "results" / "compare"
            compare_dir.mkdir(parents=True, exist_ok=True)
            (compare_dir / "energy_latency_compare_mock.json").write_text("{}", encoding="utf-8")
            (compare_dir / "energy_latency_compare_mock_plot.png").write_bytes(b"task")
            (compare_dir / "energy_latency_compare_mock_plot_efficiency.png").write_bytes(b"eff")

            lightweight_dir = root / "analysis" / "out" / "slm_lightweight_stage1_refreshed"
            midclass_dir = root / "analysis" / "out" / "slm_midclass_stage1_study"
            highclass_dir = root / "analysis" / "out" / "slm_highclass_stage1_study"
            (lightweight_dir / "plots").mkdir(parents=True, exist_ok=True)
            (midclass_dir / "plots").mkdir(parents=True, exist_ok=True)
            (highclass_dir / "plots").mkdir(parents=True, exist_ok=True)
            (lightweight_dir / "plots" / "pareto_lightweight_latency.png").write_bytes(b"lightweight-pareto")
            (midclass_dir / "plots" / "pareto_midclass_latency.png").write_bytes(b"midclass-pareto")
            (highclass_dir / "plots" / "pareto_highclass_latency.png").write_bytes(b"highclass-pareto")

            normalized_fields = [
                "source_compare_report",
                "model_id",
                "display_name",
                "family",
                "variant_kind",
                "tier",
                "benchmark_result_valid",
                "ranking_eligible",
                "status",
                "benchmark_result_invalid_reason",
            ]
            leaderboard_fields = [
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
                "status",
                "benchmark_result_invalid_reason",
            ]
            paired_fields = [
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
                "delta_tokens_per_second_mean",
            ]
            shortlist_fields = [
                "tier",
                "shortlist_rank",
                "model_id",
                "display_name",
            ]

            _write_csv(
                lightweight_dir / "normalized_records.csv",
                normalized_fields,
                [
                    {
                        "source_compare_report": str(compare_dir / "energy_latency_compare_mock.json"),
                        "model_id": "llama3_2_3b_base",
                        "display_name": "Llama 3.2 3B Base",
                        "family": "llama",
                        "variant_kind": "base",
                        "tier": "lightweight",
                        "benchmark_result_valid": "true",
                        "ranking_eligible": "true",
                        "status": "valid",
                        "benchmark_result_invalid_reason": "",
                    },
                    {
                        "source_compare_report": str(compare_dir / "energy_latency_compare_mock.json"),
                        "model_id": "phi4_mini_3_8b_base",
                        "display_name": "Phi-4 Mini 3.8B",
                        "family": "phi",
                        "variant_kind": "base",
                        "tier": "lightweight",
                        "benchmark_result_valid": "true",
                        "ranking_eligible": "true",
                        "status": "valid",
                        "benchmark_result_invalid_reason": "",
                    },
                    {
                        "source_compare_report": str(compare_dir / "energy_latency_compare_mock.json"),
                        "model_id": "qwen3_1_7b_base",
                        "display_name": "Qwen3 1.7B Base",
                        "family": "qwen",
                        "variant_kind": "base",
                        "tier": "lightweight",
                        "benchmark_result_valid": "false",
                        "ranking_eligible": "false",
                        "status": "quarantined+incomplete+invalid",
                        "benchmark_result_invalid_reason": "timeout_collapse_quarantine",
                    },
                ],
            )
            _write_csv(
                lightweight_dir / "tier_leaderboard_lightweight.csv",
                leaderboard_fields,
                [
                    {
                        "rank": "1",
                        "tier": "lightweight",
                        "model_id": "llama3_2_3b_base",
                        "display_name": "Llama 3.2 3B Base",
                        "ollama_tag": "llama3.2:3b",
                        "family": "llama",
                        "variant_kind": "base",
                        "driving_score_v2": "0.1403",
                        "task_completion_rate": "0.35",
                        "overall_score_mean": "0.3998",
                        "crash_rate": "0.35",
                        "decision_latency_ms_avg_mean": "1196.2431",
                        "p95_decision_latency_sec_mean": "1.49564",
                        "tokens_per_second_mean": "216.16517",
                        "status": "valid",
                        "benchmark_result_invalid_reason": "",
                    },
                    {
                        "rank": "2",
                        "tier": "lightweight",
                        "model_id": "dilu_phi4_mini_3_8b_instruct_v1",
                        "display_name": "DiLu Phi-4 Mini 3.8B Instruct v1",
                        "ollama_tag": "dilu-phi4-mini-3_8b-instruct-v1",
                        "family": "phi",
                        "variant_kind": "fine_tuned",
                        "driving_score_v2": "0.1257",
                        "task_completion_rate": "0.35",
                        "overall_score_mean": "0.2857",
                        "crash_rate": "0.575",
                        "decision_latency_ms_avg_mean": "1215.9492",
                        "p95_decision_latency_sec_mean": "1.697507",
                        "tokens_per_second_mean": "80.472426",
                        "status": "valid",
                        "benchmark_result_invalid_reason": "",
                    },
                    {
                        "rank": "",
                        "tier": "lightweight",
                        "model_id": "qwen3_1_7b_base",
                        "display_name": "Qwen3 1.7B Base",
                        "ollama_tag": "qwen3:1.7b",
                        "family": "qwen",
                        "variant_kind": "base",
                        "driving_score_v2": "0",
                        "task_completion_rate": "0",
                        "overall_score_mean": "0.5",
                        "crash_rate": "0",
                        "decision_latency_ms_avg_mean": "4621.5",
                        "p95_decision_latency_sec_mean": "9.848625",
                        "tokens_per_second_mean": "234.606074",
                        "status": "quarantined+incomplete+invalid",
                        "benchmark_result_invalid_reason": "timeout_collapse_quarantine",
                    },
                ],
            )
            _write_csv(
                lightweight_dir / "paired_deltas.csv",
                paired_fields,
                [
                    {
                        "family": "phi",
                        "tier": "lightweight",
                        "base_model_id": "phi4_mini_3_8b_base",
                        "base_display_name": "Phi-4 Mini 3.8B",
                        "base_ollama_tag": "phi4-mini:3.8b",
                        "fine_tuned_model_id": "dilu_phi4_mini_3_8b_instruct_v1",
                        "fine_tuned_display_name": "DiLu Phi-4 Mini 3.8B Instruct v1",
                        "fine_tuned_ollama_tag": "dilu-phi4-mini-3_8b-instruct-v1",
                        "pair_eligible": "true",
                        "pair_issue": "",
                        "delta_task_completion_rate": "-0.075",
                        "delta_overall_score_mean": "0.1006",
                        "delta_driving_score_v2": "0.0419",
                        "delta_decision_latency_ms_avg_mean": "-106.264975",
                        "delta_tokens_per_second_mean": "-93.476888",
                    },
                    {
                        "family": "qwen",
                        "tier": "lightweight",
                        "base_model_id": "qwen3_1_7b_base",
                        "base_display_name": "Qwen3 1.7B Base",
                        "base_ollama_tag": "qwen3:1.7b",
                        "fine_tuned_model_id": "dilu_qwen3_1_7b_v1",
                        "fine_tuned_display_name": "DiLu Qwen3 1.7B v1",
                        "fine_tuned_ollama_tag": "dilu-qwen3-1_7b-v1",
                        "pair_eligible": "false",
                        "pair_issue": "base_or_fine_tuned_not_rankable",
                        "delta_task_completion_rate": "",
                        "delta_overall_score_mean": "",
                        "delta_driving_score_v2": "",
                        "delta_decision_latency_ms_avg_mean": "",
                        "delta_tokens_per_second_mean": "",
                    },
                ],
            )
            _write_csv(
                lightweight_dir / "stage1_shortlist.csv",
                shortlist_fields,
                [
                    {
                        "tier": "lightweight",
                        "shortlist_rank": "1",
                        "model_id": "llama3_2_3b_base",
                        "display_name": "Llama 3.2 3B Base",
                    },
                    {
                        "tier": "lightweight",
                        "shortlist_rank": "2",
                        "model_id": "dilu_qwen3_1_7b_v1",
                        "display_name": "DiLu Qwen3 1.7B v1",
                    },
                ],
            )

            _write_csv(
                midclass_dir / "normalized_records.csv",
                normalized_fields,
                [
                    {
                        "source_compare_report": str(compare_dir / "energy_latency_compare_mock.json"),
                        "model_id": "dilu_deepseek_r1_8b_v1",
                        "display_name": "DiLu DeepSeek R1 8B v1",
                        "family": "deepseek",
                        "variant_kind": "fine_tuned",
                        "tier": "midclass",
                        "benchmark_result_valid": "true",
                        "ranking_eligible": "true",
                        "status": "valid",
                        "benchmark_result_invalid_reason": "",
                    },
                    {
                        "source_compare_report": str(compare_dir / "energy_latency_compare_mock.json"),
                        "model_id": "deepseek_r1_8b_base",
                        "display_name": "DeepSeek R1 8B Base",
                        "family": "deepseek",
                        "variant_kind": "base",
                        "tier": "midclass",
                        "benchmark_result_valid": "false",
                        "ranking_eligible": "false",
                        "status": "quarantined+incomplete+invalid",
                        "benchmark_result_invalid_reason": "timeout_collapse_quarantine",
                    },
                    {
                        "source_compare_report": str(compare_dir / "energy_latency_compare_mock.json"),
                        "model_id": "llama3_1_8b_base",
                        "display_name": "Llama 3.1 8B Base",
                        "family": "llama",
                        "variant_kind": "base",
                        "tier": "midclass",
                        "benchmark_result_valid": "true",
                        "ranking_eligible": "true",
                        "status": "valid",
                        "benchmark_result_invalid_reason": "",
                    },
                ],
            )
            _write_csv(
                midclass_dir / "tier_leaderboard_midclass.csv",
                leaderboard_fields,
                [
                    {
                        "rank": "1",
                        "tier": "midclass",
                        "model_id": "dilu_deepseek_r1_8b_v1",
                        "display_name": "DiLu DeepSeek R1 8B v1",
                        "ollama_tag": "dilu-deepseek-r1-8b-v1",
                        "family": "deepseek",
                        "variant_kind": "fine_tuned",
                        "driving_score_v2": "0.113",
                        "task_completion_rate": "0.2",
                        "overall_score_mean": "0.4704",
                        "crash_rate": "0.1",
                        "decision_latency_ms_avg_mean": "5054.184975",
                        "p95_decision_latency_sec_mean": "6.543468",
                        "tokens_per_second_mean": "51.37559",
                        "status": "valid",
                        "benchmark_result_invalid_reason": "",
                    },
                    {
                        "rank": "2",
                        "tier": "midclass",
                        "model_id": "llama3_1_8b_base",
                        "display_name": "Llama 3.1 8B Base",
                        "ollama_tag": "llama3.1:8b",
                        "family": "llama",
                        "variant_kind": "base",
                        "driving_score_v2": "0.093",
                        "task_completion_rate": "0.275",
                        "overall_score_mean": "0.2137",
                        "crash_rate": "0.7",
                        "decision_latency_ms_avg_mean": "2329.529975",
                        "p95_decision_latency_sec_mean": "2.78222",
                        "tokens_per_second_mean": "132.592456",
                        "status": "valid",
                        "benchmark_result_invalid_reason": "",
                    },
                ],
            )
            _write_csv(
                midclass_dir / "paired_deltas.csv",
                paired_fields,
                [
                    {
                        "family": "deepseek",
                        "tier": "midclass",
                        "base_model_id": "deepseek_r1_8b_base",
                        "base_display_name": "DeepSeek R1 8B Base",
                        "base_ollama_tag": "deepseek-r1:8b",
                        "fine_tuned_model_id": "dilu_deepseek_r1_8b_v1",
                        "fine_tuned_display_name": "DiLu DeepSeek R1 8B v1",
                        "fine_tuned_ollama_tag": "dilu-deepseek-r1-8b-v1",
                        "pair_eligible": "false",
                        "pair_issue": "base_or_fine_tuned_not_rankable",
                        "delta_task_completion_rate": "",
                        "delta_overall_score_mean": "",
                        "delta_driving_score_v2": "",
                        "delta_decision_latency_ms_avg_mean": "",
                        "delta_tokens_per_second_mean": "",
                    },
                    {
                        "family": "llama",
                        "tier": "midclass",
                        "base_model_id": "llama3_1_8b_base",
                        "base_display_name": "Llama 3.1 8B Base",
                        "base_ollama_tag": "llama3.1:8b",
                        "fine_tuned_model_id": "dilu_llama3_1_8b_v1",
                        "fine_tuned_display_name": "DiLu Llama 3.1 8B v1",
                        "fine_tuned_ollama_tag": "dilu-llama3_1-8b-v1",
                        "pair_eligible": "false",
                        "pair_issue": "base_or_fine_tuned_not_rankable",
                        "delta_task_completion_rate": "",
                        "delta_overall_score_mean": "",
                        "delta_driving_score_v2": "",
                        "delta_decision_latency_ms_avg_mean": "",
                        "delta_tokens_per_second_mean": "",
                    },
                ],
            )
            _write_csv(
                midclass_dir / "stage1_shortlist.csv",
                shortlist_fields,
                [
                    {
                        "tier": "midclass",
                        "shortlist_rank": "1",
                        "model_id": "dilu_deepseek_r1_8b_v1",
                        "display_name": "DiLu DeepSeek R1 8B v1",
                    }
                ],
            )

            _write_csv(
                highclass_dir / "normalized_records.csv",
                normalized_fields,
                [
                    {
                        "source_compare_report": str(compare_dir / "energy_latency_compare_mock.json"),
                        "model_id": "deepseek_r1_14b_base",
                        "display_name": "DeepSeek R1 14B Base",
                        "family": "deepseek",
                        "variant_kind": "base",
                        "tier": "highclass",
                        "benchmark_result_valid": "false",
                        "ranking_eligible": "false",
                        "status": "quarantined+incomplete+invalid",
                        "benchmark_result_invalid_reason": "timeout_collapse_quarantine",
                    },
                    {
                        "source_compare_report": str(compare_dir / "energy_latency_compare_mock.json"),
                        "model_id": "qwen3_14b_base",
                        "display_name": "Qwen3 14B Base",
                        "family": "qwen",
                        "variant_kind": "base",
                        "tier": "highclass",
                        "benchmark_result_valid": "false",
                        "ranking_eligible": "false",
                        "status": "quarantined+incomplete+invalid",
                        "benchmark_result_invalid_reason": "timeout_collapse_quarantine",
                    },
                ],
            )
            _write_csv(
                highclass_dir / "tier_leaderboard_highclass.csv",
                leaderboard_fields,
                [
                    {
                        "rank": "",
                        "tier": "highclass",
                        "model_id": "deepseek_r1_14b_base",
                        "display_name": "DeepSeek R1 14B Base",
                        "ollama_tag": "deepseek-r1:14b",
                        "family": "deepseek",
                        "variant_kind": "base",
                        "driving_score_v2": "0",
                        "task_completion_rate": "0",
                        "overall_score_mean": "0.5238",
                        "crash_rate": "0",
                        "decision_latency_ms_avg_mean": "20694.8",
                        "p95_decision_latency_sec_mean": "27.9396",
                        "tokens_per_second_mean": "33.911819",
                        "status": "quarantined+incomplete+invalid",
                        "benchmark_result_invalid_reason": "timeout_collapse_quarantine",
                    },
                    {
                        "rank": "",
                        "tier": "highclass",
                        "model_id": "qwen3_14b_base",
                        "display_name": "Qwen3 14B Base",
                        "ollama_tag": "qwen3:14b",
                        "family": "qwen",
                        "variant_kind": "base",
                        "driving_score_v2": "0",
                        "task_completion_rate": "0",
                        "overall_score_mean": "0.5",
                        "crash_rate": "0",
                        "decision_latency_ms_avg_mean": "24638.8",
                        "p95_decision_latency_sec_mean": "30.0168",
                        "tokens_per_second_mean": "10.12629",
                        "status": "quarantined+incomplete+invalid",
                        "benchmark_result_invalid_reason": "timeout_collapse_quarantine",
                    },
                ],
            )
            _write_csv(highclass_dir / "paired_deltas.csv", paired_fields, [])
            _write_csv(highclass_dir / "stage1_shortlist.csv", shortlist_fields, [])

            output_root = root / "analysis" / "out"
            result = run_publication_bundle(
                lightweight_study_dir=lightweight_dir,
                midclass_study_dir=midclass_dir,
                highclass_study_dir=highclass_dir,
                output_root=output_root,
                bundle_id="publication_three_tier_results_v1",
            )

            bundle_dir = output_root / "publication_three_tier_results_v1"
            self.assertEqual(result["bundle_dir"], str(bundle_dir))
            self.assertTrue((bundle_dir / "evidence_summary.md").exists())
            self.assertTrue((bundle_dir / "table_1_cross_tier_validity_summary.csv").exists())
            self.assertTrue((bundle_dir / "table_2_lightweight_leaderboard.csv").exists())
            self.assertTrue((bundle_dir / "table_3_lightweight_exact_pair_deltas.csv").exists())
            self.assertTrue((bundle_dir / "table_4_midclass_screening_summary.csv").exists())
            self.assertTrue((bundle_dir / "table_5_highclass_failure_summary.csv").exists())
            self.assertTrue((bundle_dir / "figure_plan.md").exists())
            self.assertTrue((bundle_dir / "caption_bank.md").exists())
            self.assertTrue((bundle_dir / "results_outline.md").exists())
            self.assertTrue((bundle_dir / "figures" / "figure_1_lightweight_task_summary.png").exists())
            self.assertTrue((bundle_dir / "figures" / "figure_2_lightweight_efficiency.png").exists())
            self.assertTrue((bundle_dir / "figures" / "figure_4_cross_tier_pareto.png").exists())
            self.assertTrue((bundle_dir / "figures" / "figure_5_lightweight_pareto.png").exists())
            self.assertTrue((bundle_dir / "figures" / "figure_6_midclass_pareto.png").exists())
            self.assertTrue((bundle_dir / "figures" / "figure_7_highclass_pareto.png").exists())

            with (bundle_dir / "table_1_cross_tier_validity_summary.csv").open("r", encoding="utf-8", newline="") as handle:
                validity_rows = list(csv.DictReader(handle))
            self.assertEqual([row["tier"] for row in validity_rows], ["lightweight", "midclass", "highclass"])
            self.assertEqual(validity_rows[0]["publication_interpretation"], "results-ready")
            self.assertEqual(validity_rows[1]["publication_interpretation"], "screening-only")
            self.assertEqual(validity_rows[2]["publication_interpretation"], "failure-analysis-only")

            with (bundle_dir / "table_3_lightweight_exact_pair_deltas.csv").open("r", encoding="utf-8", newline="") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pair_rows), 1)
            self.assertEqual(pair_rows[0]["family"], "phi")
            self.assertEqual(pair_rows[0]["delta_driving_score_v2"], "0.0419")

            with (bundle_dir / "table_4_midclass_screening_summary.csv").open("r", encoding="utf-8", newline="") as handle:
                mid_rows = list(csv.DictReader(handle))
            self.assertEqual(mid_rows[0]["family"], "deepseek")
            self.assertEqual(mid_rows[0]["base_model_status"], "quarantined+incomplete+invalid")
            self.assertEqual(mid_rows[0]["fine_tuned_model_status"], "valid")

            figure_plan = (bundle_dir / "figure_plan.md").read_text(encoding="utf-8")
            self.assertIn("Figure 1", figure_plan)
            self.assertIn("Figure 4", figure_plan)
            self.assertIn("Pareto", figure_plan)
            self.assertIn("valid-only ranking", figure_plan)


if __name__ == "__main__":
    unittest.main()
