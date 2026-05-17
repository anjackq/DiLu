import unittest
from unittest.mock import patch

import gymnasium as gym

from dilu.runtime.lampilot_highway.benchmark import (
    BENCHMARK_ID,
    DEMO_ITEM_IDS,
    build_benchmark_fingerprint,
    load_dataset,
    load_source_manifest,
)
from dilu.runtime.lampilot_highway.runner import _resolve_progress_enabled, build_parser
from dilu.runtime.lampilot_highway.envs import ensure_envs_registered
from dilu.runtime.lampilot_highway.evaluators import (
    ACCEvalbyDistance,
    ACCEvalbySpeed,
    BaseHighwayEvaluator,
    LaneChangeEval,
    OvertakeEval,
    PullOverEval,
)
from dilu.runtime.lampilot_highway.policy import (
    PolicyCompilationError,
    compile_policy_response,
)


class _DummyLane:
    def local_coordinates(self, position):
        return (0.0, 0.0)

    def heading_at(self, _s):
        return 0.0


class _DummyNetwork:
    def __init__(self):
        self.graph = {"a": {"b": [object(), object(), object()]}}

    def get_lane(self, _lane_index):
        return _DummyLane()


class _DummyRoad:
    def __init__(self, front_vehicle=None):
        self._front_vehicle = front_vehicle
        self.network = _DummyNetwork()

    def neighbour_vehicles(self, _ego, _lane_index):
        return (self._front_vehicle, None)


class _DummyVehicle:
    LENGTH = 5.0

    def __init__(self, *, speed=0.0, lane_index=("a", "b", 1), front_gap=30.0):
        self.speed = speed
        self.lane_index = lane_index
        self.heading = 0.0
        self.position = (0.0, 0.0)
        self._front_gap = front_gap
        self.road = _DummyRoad()

    def lane_distance_to(self, _other, _lane=None):
        return self._front_gap


class LaMPilotHighwayPortBenchmarkTests(unittest.TestCase):
    def test_source_manifest_matches_expected_parity(self):
        manifest = load_source_manifest(BENCHMARK_ID)
        dataset = load_dataset(BENCHMARK_ID)

        self.assertEqual(manifest["benchmark_name"], BENCHMARK_ID)
        self.assertEqual(manifest["expected_config_count"], 29)
        self.assertEqual(manifest["expected_item_count"], 3400)
        self.assertEqual(len(manifest["included_config_filenames"]), 29)
        self.assertEqual(len(dataset), 3400)
        self.assertNotIn("go_straight.json", manifest["included_config_filenames"])
        self.assertNotIn("turn_left.json", manifest["included_config_filenames"])
        self.assertNotIn("turn_right.json", manifest["included_config_filenames"])

    def test_dataset_item_ids_are_deterministic_and_demo_subset_is_supported(self):
        dataset = load_dataset(BENCHMARK_ID)
        self.assertEqual(dataset[0]["item_id"], "dec_abs_dis15:s0:c0")
        self.assertEqual(dataset[-1]["item_id"], "right_overtake:s19:c9")
        self.assertEqual(len({item["item_id"] for item in dataset}), len(dataset))

        demo_dataset = load_dataset(BENCHMARK_ID, use_demo=True)
        self.assertEqual(sorted(item["item_id"] for item in demo_dataset), sorted(DEMO_ITEM_IDS))

    def test_benchmark_fingerprint_is_stable(self):
        first = build_benchmark_fingerprint(BENCHMARK_ID)
        second = build_benchmark_fingerprint(BENCHMARK_ID)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith(f"{BENCHMARK_ID}:"))

    def test_progress_resolution_defaults_to_interactive_config_value(self):
        self.assertTrue(
            _resolve_progress_enabled(
                config={"progress_bar": True},
                force_progress=False,
                disable_progress=False,
                interactive_output=True,
            )
        )
        self.assertFalse(
            _resolve_progress_enabled(
                config={"progress_bar": True},
                force_progress=False,
                disable_progress=False,
                interactive_output=False,
            )
        )

    def test_progress_resolution_honors_cli_overrides(self):
        self.assertTrue(
            _resolve_progress_enabled(
                config={"progress_bar": False},
                force_progress=True,
                disable_progress=False,
                interactive_output=False,
            )
        )
        self.assertFalse(
            _resolve_progress_enabled(
                config={"progress_bar": True},
                force_progress=False,
                disable_progress=True,
                interactive_output=True,
            )
        )

    def test_parser_accepts_progress_flags(self):
        parser = build_parser()
        args = parser.parse_args(["--models", "qwen3:1.7b", "--progress"])
        self.assertTrue(args.progress)
        self.assertFalse(args.no_progress)

        args = parser.parse_args(["--models", "qwen3:1.7b", "--no-progress"])
        self.assertFalse(args.progress)
        self.assertTrue(args.no_progress)


class LaMPilotHighwayPortEnvTests(unittest.TestCase):
    def test_custom_envs_register_and_reset(self):
        ensure_envs_registered()

        for env_id in ("ramp-merge-v0", "dt-highway-v0"):
            env = gym.make(env_id)
            try:
                env.reset(seed=123)
            finally:
                env.close()


class LaMPilotHighwayPortPolicyTests(unittest.TestCase):
    def test_compile_policy_allows_whitelisted_api_program_and_preserves_state(self):
        response = """
Plan:
1) Keep the current lane.
Code:
```python
def hold_lane_temporarily():
    ego = get_ego_vehicle()
    lane = get_lane_of(ego)
    count = 0
    while count < 2:
        set_target_lane(lane)
        count += 1
        yield autopilot()
```
"""
        compiled = compile_policy_response(response)
        calls = []
        bindings = {
            "get_ego_vehicle": lambda: "ego",
            "get_lane_of": lambda ego: ("a", "b", 1),
            "set_target_lane": lambda lane: calls.append(("lane", lane)),
            "autopilot": lambda: ("control", 1),
        }

        program = compiled.instantiate(bindings)
        self.assertEqual(next(program), ("control", 1))
        self.assertEqual(next(program), ("control", 1))
        with self.assertRaises(StopIteration):
            next(program)
        self.assertEqual(calls, [("lane", ("a", "b", 1)), ("lane", ("a", "b", 1))])

    def test_compile_policy_rejects_imports(self):
        response = """
Plan:
1) Break out.
Code:
```python
def bad_policy():
    import os
    yield autopilot()
```
"""
        with self.assertRaises(PolicyCompilationError):
            compile_policy_response(response)

    def test_compile_policy_rejects_dunder_and_forbidden_builtins(self):
        response = """
Plan:
1) Escape sandbox.
Code:
```python
def bad_policy():
    value = __import__("os")
    yield autopilot()
```
"""
        with self.assertRaises(PolicyCompilationError):
            compile_policy_response(response)


class LaMPilotHighwayPortEvaluatorTests(unittest.TestCase):
    def test_lane_change_marks_success_only_when_heading_aligns(self):
        evaluator = LaneChangeEval.__new__(LaneChangeEval)
        evaluator.done = False
        evaluator.success = False
        evaluator.target_lane_index = ("a", "b", 0)
        evaluator.target_lane = _DummyLane()
        evaluator.ego_vehicle = _DummyVehicle(lane_index=("a", "b", 0))
        with patch.object(BaseHighwayEvaluator, "step", lambda self, agent: None):
            evaluator.step(None)
        self.assertTrue(evaluator.done)
        self.assertTrue(evaluator.success)

    def test_overtake_requires_passing_margin_on_target_lane(self):
        evaluator = OvertakeEval.__new__(OvertakeEval)
        evaluator.done = False
        evaluator.success = False
        evaluator.target_lane_index = ("a", "b", 0)
        evaluator.target_lane = _DummyLane()
        evaluator.ego_vehicle = _DummyVehicle(lane_index=("a", "b", 0), front_gap=-11.0)
        evaluator.front_vehicle = object()
        with patch.object(BaseHighwayEvaluator, "step", lambda self, agent: None):
            evaluator.step(None)
        self.assertTrue(evaluator.done)
        self.assertTrue(evaluator.success)

    def test_pull_over_requires_emergency_lane_and_stop(self):
        evaluator = PullOverEval.__new__(PullOverEval)
        evaluator.done = False
        evaluator.success = False
        evaluator.emergency_lane_index = ("a", "b", 2)
        evaluator.ego_vehicle = _DummyVehicle(lane_index=("a", "b", 2), speed=0.0)
        with patch.object(BaseHighwayEvaluator, "step", lambda self, agent: None):
            evaluator.step(None)
        self.assertTrue(evaluator.done)
        self.assertTrue(evaluator.success)

    def test_acc_by_speed_holds_target_for_five_seconds(self):
        front = _DummyVehicle(speed=40.0)
        ego = _DummyVehicle(speed=30.0, front_gap=120.0)
        ego.road = _DummyRoad(front_vehicle=None)
        evaluator = ACCEvalbySpeed.__new__(ACCEvalbySpeed)
        evaluator.config = {"eval": {"speed": 30.0}}
        evaluator.ego_vehicle = ego
        evaluator.desired_speed = 30.0
        evaluator.last_time = -1
        evaluator.time_duration = 5
        evaluator.failure_time = 60
        evaluator.max_gap = 100
        evaluator.failure_start_time = 0
        evaluator.done = False
        evaluator.success = False
        evaluator.frame = 1
        evaluator.simulation_frequency = 1
        with patch.object(BaseHighwayEvaluator, "step", lambda self, agent: None):
            evaluator.step(None)
            evaluator.frame = 7
            evaluator.step(None)
        self.assertTrue(evaluator.done)
        self.assertTrue(evaluator.success)

    def test_acc_by_distance_holds_gap_within_tolerance(self):
        front = _DummyVehicle(speed=20.0)
        ego = _DummyVehicle(speed=20.0, front_gap=20.0)
        ego.road = _DummyRoad(front_vehicle=front)
        evaluator = ACCEvalbyDistance.__new__(ACCEvalbyDistance)
        evaluator.ego_vehicle = ego
        evaluator.front_vehicle = front
        evaluator.desired_distance = 20.0
        evaluator.last_time = -1
        evaluator.time_duration = 5
        evaluator.failure_time = 60
        evaluator.failure_start_time = 0
        evaluator.init_ego_speed = 20.0
        evaluator.dis_tol = 6.0
        evaluator.done = False
        evaluator.success = False
        evaluator.frame = 1
        evaluator.simulation_frequency = 1
        with patch.object(BaseHighwayEvaluator, "step", lambda self, agent: None):
            evaluator.step(None)
            evaluator.frame = 7
            evaluator.step(None)
        self.assertTrue(evaluator.done)
        self.assertTrue(evaluator.success)


if __name__ == "__main__":
    unittest.main()
