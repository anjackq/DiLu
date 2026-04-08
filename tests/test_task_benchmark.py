import copy
import os
import unittest

import numpy as np
import yaml

from dilu.runtime.highway_env_config import resolve_simulation_env_bundle
from dilu.runtime.task_benchmark import (
    BenchmarkEpisodeEvaluator,
    augment_behavior_aware_benchmark_episode,
    benchmark_result_validity,
    bootstrap_ci95,
    compute_benchmark_case_scores,
    load_benchmark_case_set,
    summarize_benchmark_episodes,
    validate_benchmark_case_set,
)
from evaluate_models_ollama import aggregate_results
from evaluate_models_ollama import extract_step_traffic_metrics


class _DummyVehicle:
    def __init__(self, lane_rank, speed, x):
        self.lane_index = ("a", "b", int(lane_rank))
        self.speed = float(speed)
        self.position = np.array([float(x), float(lane_rank) * 4.0], dtype=float)


class _DummyRoad:
    def __init__(self, ego_vehicle, front_vehicle):
        self.ego_vehicle = ego_vehicle
        self.front_vehicle = front_vehicle
        self.vehicles = [ego_vehicle]
        if front_vehicle is not None:
            self.vehicles.append(front_vehicle)

    def neighbour_vehicles(self, vehicle, lane_index):
        return self.front_vehicle, None


class _DummyUnwrapped:
    def __init__(self, ego_vehicle, front_vehicle, available_actions):
        self.vehicle = ego_vehicle
        self.road = _DummyRoad(ego_vehicle, front_vehicle)
        self.config = {"policy_frequency": 1}
        self._available_actions = list(available_actions)

    def get_available_actions(self):
        return list(self._available_actions)


class _DummyEnv:
    def __init__(self, ego_vehicle, front_vehicle, available_actions):
        self.unwrapped = _DummyUnwrapped(ego_vehicle, front_vehicle, available_actions)


class TaskBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(repo_root, "config.example.yaml")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        cls.env_bundle = resolve_simulation_env_bundle(
            config,
            show_trajectories=False,
            render_agent=False,
        )

    def test_builtin_case_set_validates(self):
        case_set = load_benchmark_case_set("lampilot_highway_v1")
        result = validate_benchmark_case_set(
            case_set,
            self.env_bundle["env_config_map"],
            self.env_bundle["env_id"],
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["total_cases"], 40)
        self.assertEqual(result["summary"]["invalid_case_count"], 0)

    def test_invalid_case_set_fails_prevalidation(self):
        case_set = load_benchmark_case_set("lampilot_highway_v1")
        bad_case = copy.deepcopy(case_set["cases"][0])
        bad_case["success_criteria"]["min_speed_mps"] = 24.0
        bad_case["success_criteria"]["max_speed_mps"] = 26.0
        custom_case_set = {
            "benchmark_name": "invalid_speed_case",
            "cases": [bad_case],
        }
        result = validate_benchmark_case_set(
            custom_case_set,
            self.env_bundle["env_config_map"],
            self.env_bundle["env_id"],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["summary"]["invalid_case_count"], 1)
        self.assertIn(
            "initial_speed_inside_target_band",
            result["invalid_cases"][0]["reasons"],
        )

    def test_native_env_bundle_preserves_current_default_target_speeds_without_override(self):
        target_speeds = self.env_bundle["env_config_snapshot"]["action"]["target_speeds"]
        self.assertEqual(list(target_speeds), [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        self.assertEqual(self.env_bundle["env_profile_label"], "default_stop_capable")

    def test_native_env_bundle_accepts_stop_capable_target_speed_override(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(repo_root, "config.example.yaml")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["sim_action_target_speeds"] = [0, 5, 10, 15, 20, 25, 30]
        bundle = resolve_simulation_env_bundle(
            config,
            show_trajectories=False,
            render_agent=False,
        )
        target_speeds = bundle["env_config_snapshot"]["action"]["target_speeds"]
        self.assertEqual(list(target_speeds), [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

    def test_step_metrics_detect_stop_and_near_stop_independently_from_low_speed_blocking(self):
        ego = _DummyVehicle(lane_rank=1, speed=0.1, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=0.0, x=120.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        step_metrics = extract_step_traffic_metrics(
            env=env,
            ttc_threshold_sec=2.0,
            headway_threshold_m=15.0,
            rear_ttc_threshold_sec=2.5,
            rear_headway_threshold_m=12.0,
            low_speed_blocking_threshold_mps=8.5,
            blocking_front_gap_safe_m=25.0,
            blocking_front_ttc_safe_sec=4.0,
            stop_threshold_mps=0.5,
            near_stop_threshold_mps=2.0,
        )
        self.assertTrue(step_metrics["stopped"])
        self.assertTrue(step_metrics["near_stop"])
        self.assertTrue(step_metrics["low_speed_blocking"])

    def test_score_semantics_make_completion_decisive(self):
        crash_case = compute_benchmark_case_scores(
            task_completed=True,
            crashed=True,
            min_positive_ttc_sec=4.0,
            speed_history=[25.0, 25.0, 25.0],
            completion_time_sec=2.0,
            time_limit_sec=10.0,
        )
        incomplete_case = compute_benchmark_case_scores(
            task_completed=False,
            crashed=False,
            min_positive_ttc_sec=4.0,
            speed_history=[25.0, 25.0, 25.0],
            completion_time_sec=None,
            time_limit_sec=10.0,
        )
        weak_case = compute_benchmark_case_scores(
            task_completed=True,
            crashed=False,
            min_positive_ttc_sec=1.0,
            speed_history=[20.0, 30.0, 20.0, 30.0],
            completion_time_sec=8.0,
            time_limit_sec=10.0,
        )
        strong_case = compute_benchmark_case_scores(
            task_completed=True,
            crashed=False,
            min_positive_ttc_sec=3.0,
            speed_history=[28.0, 28.0, 29.0, 28.0],
            completion_time_sec=4.0,
            time_limit_sec=10.0,
        )
        self.assertEqual(crash_case["driving_score"], 0.0)
        self.assertEqual(incomplete_case["driving_score"], 0.0)
        self.assertGreater(strong_case["driving_score"], weak_case["driving_score"])

    def test_behavior_aware_v2_penalizes_qwen_like_conservative_timeout_behavior(self):
        episode = {
            "category": "speed_increase",
            "task_completed": True,
            "crashed": False,
            "overall_score": 0.58,
            "driving_score": 0.58,
            "stop_rate": 0.45,
            "near_stop_rate": 0.65,
            "low_speed_blocking_rate": 0.55,
            "decision_timeout_rate": 0.16,
            "fallback_action_rate": 0.16,
        }
        scored = augment_behavior_aware_benchmark_episode(episode)
        self.assertLess(scored["behavior_penalty_factor_v2"], 1.0)
        self.assertLess(scored["overall_score_v2"], scored["overall_score"])
        self.assertLess(scored["driving_score_v2"], scored["driving_score"])

    def test_behavior_aware_v2_is_task_aware_for_defensive_categories(self):
        defensive_episode = {
            "category": "speed_decrease",
            "task_completed": True,
            "crashed": False,
            "overall_score": 0.58,
            "driving_score": 0.58,
            "stop_rate": 0.12,
            "near_stop_rate": 0.22,
            "low_speed_blocking_rate": 0.02,
            "decision_timeout_rate": 0.0,
            "fallback_action_rate": 0.0,
        }
        assertive_episode = dict(defensive_episode)
        assertive_episode["category"] = "speed_increase"
        defensive_scored = augment_behavior_aware_benchmark_episode(defensive_episode)
        assertive_scored = augment_behavior_aware_benchmark_episode(assertive_episode)
        self.assertGreater(defensive_scored["behavior_penalty_factor_v2"], assertive_scored["behavior_penalty_factor_v2"])
        self.assertGreater(defensive_scored["driving_score_v2"], assertive_scored["driving_score_v2"])

    def test_bootstrap_ci_is_deterministic(self):
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        first = bootstrap_ci95(values, iterations=500, seed=123)
        second = bootstrap_ci95(values, iterations=500, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertLessEqual(first[0], first[1])

    def test_speed_band_predicate_requires_hold_steps(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=20.0, x=80.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "speed_band_test",
            "category": "speed_increase",
            "instruction": "speed test",
            "time_limit_sec": 10,
            "success_criteria": {
                "type": "speed_band",
                "min_speed_mps": 27.0,
                "max_speed_mps": 30.0,
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        ego.speed = 28.0
        evaluator.update(env, 1, {"front_gap_m": 80.0, "ttc_sec": 5.0}, crashed=False)
        self.assertFalse(evaluator.task_completed)
        evaluator.update(env, 2, {"front_gap_m": 80.0, "ttc_sec": 5.0}, crashed=False)
        self.assertTrue(evaluator.task_completed)

    def test_front_gap_band_predicate_requires_hold_steps(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=20.0, x=40.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "front_gap_test",
            "category": "follow_gap_increase",
            "instruction": "gap test",
            "time_limit_sec": 12,
            "success_criteria": {
                "type": "front_gap_band",
                "min_gap_m": 70.0,
                "max_gap_m": 110.0,
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        front.position[0] = 85.0
        evaluator.update(env, 1, {"front_gap_m": 85.0, "ttc_sec": 6.0}, crashed=False)
        self.assertFalse(evaluator.task_completed)
        evaluator.update(env, 2, {"front_gap_m": 85.0, "ttc_sec": 6.0}, crashed=False)
        self.assertTrue(evaluator.task_completed)

    def test_lane_change_predicate_requires_hold_steps(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=20.0, x=60.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "lane_change_test",
            "category": "lane_change_left",
            "instruction": "lane test",
            "time_limit_sec": 10,
            "success_criteria": {
                "type": "lane_change",
                "direction": "left",
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        ego.lane_index = ("a", "b", 0)
        ego.position[1] = 0.0
        evaluator.update(env, 1, {"front_gap_m": 60.0, "ttc_sec": 4.0}, crashed=False)
        self.assertFalse(evaluator.task_completed)
        evaluator.update(env, 2, {"front_gap_m": 60.0, "ttc_sec": 4.0}, crashed=False)
        self.assertTrue(evaluator.task_completed)

    def test_overtake_predicate_requires_lane_use_and_pass_margin(self):
        ego = _DummyVehicle(lane_rank=1, speed=25.0, x=0.0)
        front = _DummyVehicle(lane_rank=1, speed=20.0, x=45.0)
        env = _DummyEnv(ego, front, available_actions=[0, 1, 2, 3, 4])
        case = {
            "case_id": "overtake_test",
            "category": "overtake_left",
            "instruction": "overtake test",
            "time_limit_sec": 16,
            "success_criteria": {
                "type": "overtake",
                "direction": "left",
                "pass_margin_m": 5.0,
                "hold_steps": 2,
            },
        }
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        ego.lane_index = ("a", "b", 0)
        ego.position[1] = 0.0
        ego.position[0] = 60.0
        evaluator.update(env, 1, {"front_gap_m": None, "ttc_sec": 5.0}, crashed=False)
        self.assertFalse(evaluator.task_completed)
        ego.position[0] = 65.0
        evaluator.update(env, 2, {"front_gap_m": None, "ttc_sec": 5.0}, crashed=False)
        self.assertTrue(evaluator.task_completed)

    def test_benchmark_aggregate_includes_category_breakdown_and_ci(self):
        episodes = [
            {
                "category": "speed_increase",
                "task_completed": True,
                "ttc_score": 1.0,
                "speed_variance_score": 0.9,
                "time_efficiency_score": 0.8,
                "overall_score": 0.93,
                "driving_score": 0.93,
                "overall_score_v2": 0.75,
                "driving_score_v2": 0.75,
                "behavior_penalty_factor_v2": 0.8,
                "conservative_penalty_severity_v2": 0.1,
                "runtime_penalty_severity_v2": 0.125,
                "benchmark_failure_reason": "",
            },
            {
                "category": "speed_increase",
                "task_completed": False,
                "ttc_score": 0.9,
                "speed_variance_score": 0.8,
                "time_efficiency_score": 0.0,
                "overall_score": 0.69,
                "driving_score": 0.0,
                "overall_score_v2": 0.2,
                "driving_score_v2": 0.0,
                "behavior_penalty_factor_v2": 0.29,
                "conservative_penalty_severity_v2": 0.5,
                "runtime_penalty_severity_v2": 0.42,
                "benchmark_failure_reason": "task_not_completed",
            },
            {
                "category": "lane_change_left",
                "task_completed": True,
                "ttc_score": 0.8,
                "speed_variance_score": 0.7,
                "time_efficiency_score": 0.6,
                "overall_score": 0.73,
                "driving_score": 0.73,
                "overall_score_v2": 0.65,
                "driving_score_v2": 0.65,
                "behavior_penalty_factor_v2": 0.89,
                "conservative_penalty_severity_v2": 0.08,
                "runtime_penalty_severity_v2": 0.03,
                "benchmark_failure_reason": "",
            },
        ]
        summary = summarize_benchmark_episodes(episodes)
        self.assertEqual(summary["benchmark_case_count"], 3)
        self.assertAlmostEqual(summary["task_completion_rate"], 0.6667, places=4)
        self.assertIsNotNone(summary["task_completion_rate_ci95"])
        self.assertIsNotNone(summary["driving_score_ci95"])
        self.assertIn("driving_score_v2", summary)
        self.assertIn("driving_score_v2_ci95", summary)
        self.assertIn("behavior_penalty_factor_v2_mean", summary)
        self.assertIn("speed_increase", summary["benchmark_by_category"])
        self.assertIn("lane_change_left", summary["benchmark_by_category"])
        self.assertIn("driving_score_v2", summary["benchmark_by_category"]["speed_increase"])
        self.assertEqual(
            summary["benchmark_by_category"]["speed_increase"]["benchmark_case_count"],
            2,
        )

    def test_benchmark_result_validity_thresholds(self):
        valid, reason = benchmark_result_validity(
            decision_timeout_rate_mean=0.49,
            fallback_action_rate_mean=0.1,
            timeout_episode_rate=0.2,
        )
        self.assertTrue(valid)
        self.assertIsNone(reason)

        valid, reason = benchmark_result_validity(
            decision_timeout_rate_mean=0.5,
            fallback_action_rate_mean=0.2,
            timeout_episode_rate=0.1,
        )
        self.assertFalse(valid)
        self.assertIn("decision_timeout_rate_mean>=0.5", reason)

    def test_benchmark_finalize_preserves_timeout_cap_failure_reason(self):
        case = {
            "case_id": "speed_increase_case",
            "instruction": "Speed up safely.",
            "category": "speed_increase",
            "seed": 1,
            "success_criteria": {"type": "speed_band", "min_speed_mps": 20.0, "max_speed_mps": 30.0, "hold_steps": 2},
            "time_limit_sec": 8,
        }

        class _DummyEnv:
            config = {"policy_frequency": 1}
            unwrapped = None

        env = _DummyEnv()
        env.unwrapped = env
        evaluator = BenchmarkEpisodeEvaluator(case, env)
        metrics = evaluator.finalize(crashed=False, episode_stop_reason="episode_timeout_cap")
        self.assertEqual(metrics["benchmark_failure_reason"], "episode_timeout_cap")

    def test_seed_mode_aggregate_has_no_benchmark_fields(self):
        episode = {
            "crashed": False,
            "error": None,
            "success_no_collision": True,
            "truncated": False,
            "terminated": True,
            "steps": 10,
            "episode_runtime_sec": 1.0,
            "decisions_made": 10,
            "decision_calls_total": 10,
            "decision_timeout_count": 0,
        }
        summary = aggregate_results("seed_only_model", [episode])
        self.assertNotIn("task_completion_rate", summary)
        self.assertNotIn("benchmark_result_valid", summary)

    def test_aggregate_includes_stop_and_near_stop_metrics(self):
        episodes = [
            {
                "crashed": False,
                "error": None,
                "success_no_collision": True,
                "truncated": False,
                "terminated": True,
                "steps": 10,
                "episode_runtime_sec": 1.0,
                "decisions_made": 10,
                "decision_calls_total": 10,
                "decision_timeout_count": 0,
                "fallback_action_count": 0,
                "ollama_native_retry_count": 0,
                "ollama_openai_fallback_count": 0,
                "ollama_native_decision_count": 0,
                "ollama_native_timeout_count": 0,
                "ollama_native_timeout_short_circuit_count": 0,
                "ollama_downgrade_triggered": False,
                "responses_with_delimiter": 0,
                "responses_strict_format": 0,
                "responses_direct_parseable": 0,
                "format_failure_count": 0,
                "episode_reward_sum": 0.0,
                "avg_ego_speed_mps": 0.2,
                "ttc_danger_rate": 0.0,
                "headway_violation_rate": 0.0,
                "rear_ttc_danger_rate": 0.0,
                "rear_headway_violation_rate": 0.0,
                "low_speed_blocking_rate": 1.0,
                "lane_change_rate": 0.0,
                "flap_accel_decel_rate": 0.0,
                "decision_latency_ms_avg": 5.0,
                "timeout_penalty_events": 0,
                "timeout_penalty_timeout_triggers": 0,
                "timeout_penalty_slow_triggers": 0,
                "timeout_penalty_stage_max": 0,
                "min_ego_speed_mps": 0.0,
                "stopped_ever": True,
                "stop_steps": 8,
                "stop_rate": 0.8,
                "near_stop_steps": 10,
                "near_stop_rate": 1.0,
            },
            {
                "crashed": False,
                "error": None,
                "success_no_collision": True,
                "truncated": False,
                "terminated": True,
                "steps": 10,
                "episode_runtime_sec": 1.0,
                "decisions_made": 10,
                "decision_calls_total": 10,
                "decision_timeout_count": 0,
                "fallback_action_count": 0,
                "ollama_native_retry_count": 0,
                "ollama_openai_fallback_count": 0,
                "ollama_native_decision_count": 0,
                "ollama_native_timeout_count": 0,
                "ollama_native_timeout_short_circuit_count": 0,
                "ollama_downgrade_triggered": False,
                "responses_with_delimiter": 0,
                "responses_strict_format": 0,
                "responses_direct_parseable": 0,
                "format_failure_count": 0,
                "episode_reward_sum": 0.0,
                "avg_ego_speed_mps": 12.0,
                "ttc_danger_rate": 0.0,
                "headway_violation_rate": 0.0,
                "rear_ttc_danger_rate": 0.0,
                "rear_headway_violation_rate": 0.0,
                "low_speed_blocking_rate": 0.0,
                "lane_change_rate": 0.0,
                "flap_accel_decel_rate": 0.0,
                "decision_latency_ms_avg": 5.0,
                "timeout_penalty_events": 0,
                "timeout_penalty_timeout_triggers": 0,
                "timeout_penalty_slow_triggers": 0,
                "timeout_penalty_stage_max": 0,
                "min_ego_speed_mps": 10.0,
                "stopped_ever": False,
                "stop_steps": 0,
                "stop_rate": 0.0,
                "near_stop_steps": 0,
                "near_stop_rate": 0.0,
            },
        ]
        summary = aggregate_results("stop_metrics_model", episodes)
        self.assertEqual(summary["min_ego_speed_mps_mean"], 5.0)
        self.assertEqual(summary["stop_episode_rate"], 0.5)
        self.assertEqual(summary["stop_rate_mean"], 0.4)
        self.assertEqual(summary["near_stop_episode_rate"], 0.5)
        self.assertEqual(summary["near_stop_rate_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
