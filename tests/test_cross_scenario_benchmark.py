import json
import os
import tempfile
import unittest

import gymnasium as gym

from dilu.runtime.highway_env_config import resolve_simulation_env_bundle
from dilu.runtime.task_benchmark import (
    BenchmarkEpisodeEvaluator,
    benchmark_metric_config,
    load_benchmark_case_set,
    validate_benchmark_case,
)
from dilu.scenario.envScenario import EnvScenario


def _write_case_set(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class CrossScenarioBenchmarkTests(unittest.TestCase):
    def test_intersection_available_actions_follow_env_action_semantics(self):
        env = gym.make("intersection-v0", render_mode="rgb_array")
        try:
            env.reset(seed=123)
            scenario = EnvScenario(env, "intersection-v0", 123, enable_db=False)
            desc = scenario.availableActionsDescription()
        finally:
            env.close()

        self.assertIn("IDLE - remain in the current lane with current speed Action_id: 1", desc)
        self.assertIn("Deceleration - decelerate the vehicle Action_id: 0", desc)
        self.assertNotIn("Turn-left", desc)

    def test_intersection_description_mentions_intersection_context(self):
        env = gym.make("intersection-v0", render_mode="rgb_array")
        try:
            env.reset(seed=123)
            scenario = EnvScenario(env, "intersection-v0", 123, enable_db=False)
            desc = scenario.describe(0)
        finally:
            env.close()

        self.assertIn("intersection", desc.lower())

    def test_load_benchmark_case_set_preserves_target_env_and_scenario_family(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cases.json")
            _write_case_set(
                path,
                {
                    "benchmark_name": "lampilot_merge_v1",
                    "target_env_id": "merge-v0",
                    "scenario_family": "merge",
                    "defaults": {
                        "time_limit_sec": 12,
                        "success_criteria": {"type": "merge_complete", "hold_steps": 2},
                    },
                    "cases": [
                        {
                            "case_id": "merge_clear_gap_0001",
                            "category": "clear_gap",
                            "instruction": "Merge safely into the mainline traffic flow.",
                            "seed": 1,
                        }
                    ],
                },
            )

            loaded = load_benchmark_case_set(path)

        self.assertEqual(loaded["target_env_id"], "merge-v0")
        self.assertEqual(loaded["scenario_family"], "merge")

    def test_validate_benchmark_case_supports_merge_complete(self):
        reasons = validate_benchmark_case(
            {
                "case_id": "merge_case",
                "success_criteria": {
                    "type": "merge_complete",
                    "target_lane_offset": -1,
                    "hold_steps": 2,
                    "min_progress_m": 20,
                },
            },
            {
                "initial_lane_rank": 1,
                "available_actions": [0, 1, 4],
                "can_change_left": True,
                "can_change_right": False,
            },
        )

        self.assertEqual(reasons, [])

    def test_benchmark_evaluator_supports_arrive_for_intersection(self):
        env = gym.make("intersection-v0", render_mode="rgb_array")
        try:
            env.reset(seed=123)
            case = {
                "case_id": "arrive_case",
                "category": "clear_cross",
                "instruction": "Cross the intersection safely and arrive.",
                "time_limit_sec": 15,
                "difficulty": "medium",
                "case_group": "intersection_progress",
                "success_criteria": {"type": "arrive", "hold_steps": 1},
                "tags": [],
            }
            evaluator = BenchmarkEpisodeEvaluator(case, env)
            info = {"speed": 4.0, "crashed": False}
            while not evaluator.task_completed:
                env.step(1)
                evaluator.update(
                    env,
                    step_idx=len(evaluator.speed_history) + 1,
                    step_metrics={"front_gap_m": None, "ttc_sec": None},
                    crashed=False,
                    info=info,
                )
                if len(evaluator.speed_history) > evaluator.max_steps:
                    break
            result = evaluator.finalize(crashed=False, episode_stop_reason="completed")
        finally:
            env.close()

        self.assertTrue(result["task_completed"])
        self.assertEqual(result["benchmark_failure_reason"], None)

    def test_resolve_simulation_env_bundle_rejects_continuous_envs_for_benchmark_use(self):
        with self.assertRaisesRegex(ValueError, "unsupported.*continuous"):
            resolve_simulation_env_bundle(
                {"sim_env_id": "parking-v0", "sim_use_native_env_defaults": True},
                show_trajectories=False,
                render_agent=False,
                require_discrete_meta_action=True,
            )

    def test_non_highway_metric_config_uses_legacy_driving_score_headline(self):
        merge_metrics = benchmark_metric_config("merge")
        intersection_metrics = benchmark_metric_config("intersection")

        self.assertEqual(merge_metrics["recommended_headline_metric"], "driving_score")
        self.assertEqual(intersection_metrics["recommended_headline_metric"], "driving_score")

    def test_cross_scenario_case_sets_exist_and_expose_metadata(self):
        merge_case_set = load_benchmark_case_set("benchmarks/lampilot_merge_v1/cases.json")
        intersection_case_set = load_benchmark_case_set("benchmarks/lampilot_intersection_v1/cases.json")

        self.assertEqual(merge_case_set["target_env_id"], "merge-v0")
        self.assertEqual(merge_case_set["scenario_family"], "merge")
        self.assertTrue(merge_case_set["cases"])

        self.assertEqual(intersection_case_set["target_env_id"], "intersection-v0")
        self.assertEqual(intersection_case_set["scenario_family"], "intersection")
        self.assertTrue(intersection_case_set["cases"])


if __name__ == "__main__":
    unittest.main()
