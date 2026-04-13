import copy
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


DEFAULT_BENCHMARK_CASE_SET = "lampilot_highway_v1"
DEFAULT_TARGET_ENV_ID = "highway-fast-v0"
BENCHMARK_TTC_SAFE_THRESHOLD_SEC = 2.0
BENCHMARK_SPEED_STD_SAFE_MPS = 4.0
BENCHMARK_OVERALL_WEIGHTS = {
    "ttc": 0.5,
    "speed_variance": 0.3,
    "time_efficiency": 0.2,
}
BENCHMARK_SCORING_POLICY_VERSION = "v2_behavior_aware"
BENCHMARK_RECOMMENDED_HEADLINE_METRIC = "driving_score_v2"
BENCHMARK_BOOTSTRAP_ITERATIONS = 2000
BENCHMARK_BOOTSTRAP_SEED = 20260326
BENCHMARK_V2_ASSERTIVE_CATEGORIES = (
    "follow_gap_decrease",
    "lane_change_left",
    "lane_change_right",
    "overtake_left",
    "overtake_right",
    "speed_increase",
)
BENCHMARK_V2_DEFENSIVE_CATEGORIES = (
    "follow_gap_increase",
    "speed_decrease",
)
BENCHMARK_V2_CONSERVATIVE_PROFILES = {
    "assertive": {
        "stop_rate": {"weight": 0.40, "grace": 0.02},
        "near_stop_rate": {"weight": 0.20, "grace": 0.05},
        "low_speed_blocking_rate": {"weight": 0.40, "grace": 0.05},
    },
    "defensive": {
        "stop_rate": {"weight": 0.20, "grace": 0.10},
        "near_stop_rate": {"weight": 0.10, "grace": 0.20},
        "low_speed_blocking_rate": {"weight": 0.70, "grace": 0.10},
    },
}
BENCHMARK_V2_RUNTIME_PENALTY = {
    "decision_timeout_rate": {"weight": 0.40},
    "fallback_action_rate": {"weight": 0.60},
    "grace": 0.01,
    "cap": 0.25,
}

_ENV_OVERRIDE_ALIASES = {
    "simulation_duration": "duration",
    "vehicle_count": "vehicles_count",
    "other_vehicle_type": "other_vehicles_type",
}
_ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _infer_scenario_family_from_env_id(env_id: str) -> str:
    text = str(env_id or "").strip().lower()
    if text.startswith("merge-"):
        return "merge"
    if text.startswith("intersection-"):
        return "intersection"
    if text.startswith("parking-"):
        return "parking"
    return "highway"


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _normalize_env_overrides(raw_overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw_overrides = raw_overrides or {}
    normalized: Dict[str, Any] = {}
    for key, value in raw_overrides.items():
        mapped_key = _ENV_OVERRIDE_ALIASES.get(str(key), str(key))
        if isinstance(value, dict):
            normalized[mapped_key] = _normalize_env_overrides(value)
        else:
            normalized[mapped_key] = value
    return normalized


def resolve_benchmark_case_set_path(identifier: str) -> str:
    raw = str(identifier or "").strip()
    if not raw:
        raise ValueError("Benchmark case set identifier cannot be empty.")
    if os.path.isfile(raw):
        return os.path.abspath(raw)
    candidate = os.path.join(_repo_root(), "benchmarks", raw, "cases.json")
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Benchmark case set not found: {identifier}. "
        f"Expected a JSON file path or benchmarks/<name>/cases.json."
    )


def load_benchmark_case_set(identifier: str) -> Dict[str, Any]:
    case_set_path = resolve_benchmark_case_set_path(identifier)
    with open(case_set_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Benchmark case set must be a JSON object.")

    defaults = raw.get("defaults") or {}
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("Benchmark case set must define a non-empty `cases` list.")

    normalized_cases: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases_raw, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Benchmark case #{idx} must be a JSON object.")
        case_id = str(case.get("case_id") or f"case_{idx:03d}").strip()
        category = str(case.get("category") or "").strip()
        instruction = str(case.get("instruction") or "").strip()
        if not category:
            raise ValueError(f"Benchmark case `{case_id}` is missing `category`.")
        if not instruction:
            raise ValueError(f"Benchmark case `{case_id}` is missing `instruction`.")
        if case.get("seed") is None:
            raise ValueError(f"Benchmark case `{case_id}` is missing `seed`.")

        success_criteria = copy.deepcopy(defaults.get("success_criteria") or {})
        if case.get("success_criteria"):
            success_criteria = _deep_update(success_criteria, dict(case["success_criteria"]))

        env_overrides = copy.deepcopy(defaults.get("env_overrides") or {})
        if case.get("env_overrides"):
            env_overrides = _deep_update(env_overrides, dict(case["env_overrides"]))
        env_overrides = _normalize_env_overrides(env_overrides)

        difficulty = str(case.get("difficulty") or defaults.get("difficulty") or "medium").strip().lower()
        if difficulty not in _ALLOWED_DIFFICULTIES:
            raise ValueError(
                f"Benchmark case `{case_id}` has invalid difficulty `{difficulty}`. "
                f"Allowed: {sorted(_ALLOWED_DIFFICULTIES)}"
            )
        case_group = str(case.get("case_group") or defaults.get("case_group") or category).strip() or category

        normalized_case = {
            "case_id": case_id,
            "category": category,
            "instruction": instruction,
            "seed": int(case["seed"]),
            "time_limit_sec": float(case.get("time_limit_sec", defaults.get("time_limit_sec", 12.0))),
            "success_criteria": success_criteria,
            "env_overrides": env_overrides,
            "tags": [str(tag) for tag in (case.get("tags") or [])],
            "difficulty": difficulty,
            "case_group": case_group,
        }
        normalized_cases.append(normalized_case)

    categories = sorted({case["category"] for case in normalized_cases})
    benchmark_name = str(raw.get("benchmark_name") or os.path.basename(os.path.dirname(case_set_path)) or DEFAULT_BENCHMARK_CASE_SET)
    target_env_id = str(raw.get("target_env_id") or DEFAULT_TARGET_ENV_ID).strip() or DEFAULT_TARGET_ENV_ID
    scenario_family = str(raw.get("scenario_family") or _infer_scenario_family_from_env_id(target_env_id)).strip().lower()
    return {
        "benchmark_name": benchmark_name,
        "case_set_path": case_set_path,
        "version": str(raw.get("version") or "1.0"),
        "description": str(raw.get("description") or "").strip(),
        "target_env_id": target_env_id,
        "scenario_family": scenario_family,
        "defaults": defaults,
        "categories": categories,
        "cases": normalized_cases,
    }


def build_case_env_config(
    base_env_config_map: Dict[str, Dict[str, Any]],
    env_type: str,
    case: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    env_config_map = copy.deepcopy(base_env_config_map)
    env_cfg = copy.deepcopy(env_config_map[env_type])
    overrides = dict((case or {}).get("env_overrides") or {})
    if overrides:
        _deep_update(env_cfg, overrides)

    time_limit_sec = float((case or {}).get("time_limit_sec") or 0.0)
    if time_limit_sec > 0:
        env_cfg["duration"] = max(float(env_cfg.get("duration", 0.0) or 0.0), time_limit_sec)

    if isinstance(env_cfg.get("observation"), dict) and env_cfg.get("vehicles_count") is not None:
        env_cfg["observation"] = dict(env_cfg["observation"])
        env_cfg["observation"]["vehicles_count"] = int(env_cfg["vehicles_count"])

    env_config_map[env_type] = env_cfg
    return env_config_map, env_cfg


def benchmark_max_steps(case: Optional[Dict[str, Any]], env_config_snapshot: Dict[str, Any], default_steps: int) -> int:
    if not case:
        return int(default_steps)
    policy_frequency = float(env_config_snapshot.get("policy_frequency", 1) or 1)
    time_limit_sec = float(case.get("time_limit_sec") or env_config_snapshot.get("duration") or default_steps)
    return max(1, int(math.ceil(time_limit_sec * max(policy_frequency, 1.0))))


def build_benchmark_instruction(case: Dict[str, Any]) -> str:
    return (
        f"Primary task: {case['instruction']} "
        "Complete the task while driving safely, obeying lane boundaries, and avoiding collisions."
    )


def benchmark_metric_config(scenario_family: str = "highway") -> Dict[str, Any]:
    scenario_family = str(scenario_family or "highway").strip().lower()
    recommended_headline_metric = (
        BENCHMARK_RECOMMENDED_HEADLINE_METRIC if scenario_family == "highway" else "driving_score"
    )
    return {
        "ttc_safe_threshold_sec": BENCHMARK_TTC_SAFE_THRESHOLD_SEC,
        "speed_std_safe_mps": BENCHMARK_SPEED_STD_SAFE_MPS,
        "overall_weights": dict(BENCHMARK_OVERALL_WEIGHTS),
        "driving_score_formula": "0 if crashed else completion_rate * overall_score",
        "benchmark_scoring_policy_version": BENCHMARK_SCORING_POLICY_VERSION,
        "recommended_headline_metric": recommended_headline_metric,
        "scenario_family": scenario_family,
        "behavior_aware_v2": {
            "formula": "overall_score_v2 = overall_score * (1 - conservative_penalty_severity_v2) * (1 - runtime_penalty_severity_v2); driving_score_v2 = driving_score * (1 - conservative_penalty_severity_v2) * (1 - runtime_penalty_severity_v2)",
            "category_groups": {
                "assertive": list(BENCHMARK_V2_ASSERTIVE_CATEGORIES),
                "defensive": list(BENCHMARK_V2_DEFENSIVE_CATEGORIES),
            },
            "conservative_profiles": copy.deepcopy(BENCHMARK_V2_CONSERVATIVE_PROFILES),
            "runtime_penalty": copy.deepcopy(BENCHMARK_V2_RUNTIME_PENALTY),
        },
        "bootstrap_iterations": int(BENCHMARK_BOOTSTRAP_ITERATIONS),
        "bootstrap_seed": int(BENCHMARK_BOOTSTRAP_SEED),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _norm_with_grace(rate: Any, grace: float) -> float:
    return _clamp01((float(rate or 0.0) - float(grace)) / max(1e-9, 1.0 - float(grace)))


def _runtime_norm(rate: Any) -> float:
    grace = float(BENCHMARK_V2_RUNTIME_PENALTY["grace"])
    cap = float(BENCHMARK_V2_RUNTIME_PENALTY["cap"])
    return _clamp01((float(rate or 0.0) - grace) / max(1e-9, cap - grace))


def _conservative_profile_name(category: Any) -> str:
    category_name = str(category or "").strip()
    if category_name in BENCHMARK_V2_DEFENSIVE_CATEGORIES:
        return "defensive"
    return "assertive"


def compute_behavior_aware_penalty_v2(
    *,
    category: Any,
    stop_rate: Any,
    near_stop_rate: Any,
    low_speed_blocking_rate: Any,
    decision_timeout_rate: Any,
    fallback_action_rate: Any,
) -> Dict[str, Any]:
    profile_name = _conservative_profile_name(category)
    profile = BENCHMARK_V2_CONSERVATIVE_PROFILES[profile_name]

    conservative_penalty_severity_v2 = 0.0
    for metric_name, metric_cfg in profile.items():
        conservative_penalty_severity_v2 += float(metric_cfg["weight"]) * _norm_with_grace(
            {
                "stop_rate": stop_rate,
                "near_stop_rate": near_stop_rate,
                "low_speed_blocking_rate": low_speed_blocking_rate,
            }[metric_name],
            float(metric_cfg["grace"]),
        )

    runtime_penalty_severity_v2 = (
        float(BENCHMARK_V2_RUNTIME_PENALTY["decision_timeout_rate"]["weight"]) * _runtime_norm(decision_timeout_rate)
        + float(BENCHMARK_V2_RUNTIME_PENALTY["fallback_action_rate"]["weight"]) * _runtime_norm(fallback_action_rate)
    )

    conservative_penalty_severity_v2 = _clamp01(conservative_penalty_severity_v2)
    runtime_penalty_severity_v2 = _clamp01(runtime_penalty_severity_v2)
    conservative_factor_v2 = 1.0 - conservative_penalty_severity_v2
    runtime_factor_v2 = 1.0 - runtime_penalty_severity_v2
    behavior_penalty_factor_v2 = conservative_factor_v2 * runtime_factor_v2

    return {
        "behavior_penalty_profile_v2": profile_name,
        "conservative_penalty_severity_v2": round(conservative_penalty_severity_v2, 4),
        "runtime_penalty_severity_v2": round(runtime_penalty_severity_v2, 4),
        "behavior_penalty_factor_v2": round(behavior_penalty_factor_v2, 4),
    }


def augment_behavior_aware_benchmark_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    if "task_completed" not in episode:
        return dict(episode)
    if str(episode.get("scenario_family") or "highway").strip().lower() != "highway":
        return dict(episode)

    scored = dict(episode)
    penalty_metrics = compute_behavior_aware_penalty_v2(
        category=scored.get("category"),
        stop_rate=scored.get("stop_rate"),
        near_stop_rate=scored.get("near_stop_rate"),
        low_speed_blocking_rate=scored.get("low_speed_blocking_rate"),
        decision_timeout_rate=scored.get("decision_timeout_rate"),
        fallback_action_rate=scored.get("fallback_action_rate"),
    )
    behavior_penalty_factor_v2 = float(penalty_metrics["behavior_penalty_factor_v2"])
    overall_score = float(scored.get("overall_score", 0.0) or 0.0)
    driving_score = float(scored.get("driving_score", 0.0) or 0.0)

    scored.update(penalty_metrics)
    scored["overall_score_v2"] = round(overall_score * behavior_penalty_factor_v2, 4)
    scored["driving_score_v2"] = round(driving_score * behavior_penalty_factor_v2, 4)
    return scored


def _lane_rank(vehicle) -> Optional[int]:
    lane_index = getattr(vehicle, "lane_index", None)
    if not lane_index or len(lane_index) < 3:
        return None
    try:
        return int(lane_index[2])
    except Exception:
        return None


def _vehicle_x(vehicle) -> Optional[float]:
    if vehicle is None:
        return None
    try:
        return float(vehicle.position[0])
    except Exception:
        return None


def _vehicle_by_runtime_id(road, runtime_id: Optional[int]):
    if road is None or runtime_id is None:
        return None
    for vehicle in getattr(road, "vehicles", []):
        if id(vehicle) == runtime_id:
            return vehicle
    return None


def _resolve_direction_offset(criteria: Dict[str, Any]) -> int:
    if "target_lane_offset" in criteria:
        return int(criteria.get("target_lane_offset") or 0)
    direction = str(criteria.get("direction") or criteria.get("target_lane") or "").strip().lower()
    if direction == "left":
        return -1
    if direction == "right":
        return 1
    return 0


def inspect_benchmark_initial_state(env) -> Dict[str, Any]:
    uenv = env.unwrapped
    ego = getattr(uenv, "vehicle", None)
    road = getattr(uenv, "road", None)
    available_actions = list(getattr(uenv, "get_available_actions", lambda: [])())
    front_vehicle = None
    front_gap_m = None
    front_is_ahead = False
    if ego is not None and road is not None:
        front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
        if front_vehicle is not None:
            front_gap_m = float(np.linalg.norm(ego.position - front_vehicle.position))
            ego_x = _vehicle_x(ego)
            front_x = _vehicle_x(front_vehicle)
            front_is_ahead = bool(
                ego_x is not None and front_x is not None and front_x > ego_x
            )
    return {
        "initial_lane_rank": _lane_rank(ego),
        "initial_speed_mps": float(getattr(ego, "speed", 0.0) or 0.0) if ego is not None else None,
        "initial_front_vehicle_exists": bool(front_vehicle is not None),
        "initial_front_gap_m": front_gap_m,
        "initial_front_vehicle_is_ahead": bool(front_is_ahead),
        "available_actions": available_actions,
        "can_change_left": 0 in available_actions,
        "can_change_right": 2 in available_actions,
    }


def validate_benchmark_case(case: Dict[str, Any], initial_state: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    criteria = dict(case.get("success_criteria") or {})
    criteria_type = str(criteria.get("type") or "").strip().lower()

    if criteria_type == "speed_band":
        speed = initial_state.get("initial_speed_mps")
        min_speed = float(criteria.get("min_speed_mps", 0.0))
        max_speed = float(criteria.get("max_speed_mps", 999.0))
        if speed is None:
            reasons.append("missing_initial_speed")
        elif min_speed <= float(speed) <= max_speed:
            reasons.append("initial_speed_inside_target_band")

    elif criteria_type == "front_gap_band":
        if not initial_state.get("initial_front_vehicle_exists"):
            reasons.append("missing_initial_front_vehicle")
        else:
            front_gap_m = initial_state.get("initial_front_gap_m")
            min_gap = float(criteria.get("min_gap_m", 0.0))
            max_gap = float(criteria.get("max_gap_m", 1e9))
            if front_gap_m is None:
                reasons.append("missing_initial_front_gap")
            elif min_gap <= float(front_gap_m) <= max_gap:
                reasons.append("initial_front_gap_inside_target_band")

    elif criteria_type == "lane_change":
        lane_rank = initial_state.get("initial_lane_rank")
        target_offset = _resolve_direction_offset(criteria)
        if target_offset == 0:
            reasons.append("invalid_target_lane_offset")
        elif target_offset < 0 and not initial_state.get("can_change_left"):
            reasons.append("target_left_lane_unavailable")
        elif target_offset > 0 and not initial_state.get("can_change_right"):
            reasons.append("target_right_lane_unavailable")
        elif lane_rank is None:
            reasons.append("missing_initial_lane_rank")
        else:
            target_lane_rank = int(lane_rank) + int(target_offset)
            if target_lane_rank == int(lane_rank):
                reasons.append("ego_already_in_target_lane")

    elif criteria_type == "overtake":
        target_offset = _resolve_direction_offset(criteria)
        if target_offset == 0:
            reasons.append("invalid_target_lane_offset")
        elif target_offset < 0 and not initial_state.get("can_change_left"):
            reasons.append("target_left_lane_unavailable")
        elif target_offset > 0 and not initial_state.get("can_change_right"):
            reasons.append("target_right_lane_unavailable")
        if not initial_state.get("initial_front_vehicle_exists"):
            reasons.append("missing_initial_front_vehicle")
        elif not initial_state.get("initial_front_vehicle_is_ahead"):
            reasons.append("initial_front_vehicle_not_ahead")

    elif criteria_type == "merge_complete":
        lane_rank = initial_state.get("initial_lane_rank")
        target_offset = _resolve_direction_offset(criteria)
        if target_offset == 0:
            reasons.append("invalid_target_lane_offset")
        elif target_offset < 0 and not initial_state.get("can_change_left"):
            reasons.append("target_left_lane_unavailable")
        elif target_offset > 0 and not initial_state.get("can_change_right"):
            reasons.append("target_right_lane_unavailable")
        elif lane_rank is None:
            reasons.append("missing_initial_lane_rank")

    elif criteria_type == "arrive":
        pass

    else:
        reasons.append(f"unsupported_success_criteria_type:{criteria_type or 'missing'}")

    return reasons


def validate_benchmark_case_set(
    case_set: Dict[str, Any],
    base_env_config_map: Dict[str, Dict[str, Any]],
    env_type: str,
) -> Dict[str, Any]:
    target_env_id = str(case_set.get("target_env_id") or "").strip()
    if target_env_id and target_env_id != str(env_type):
        raise ValueError(
            f"Benchmark case set {case_set.get('benchmark_name')!r} targets env_id={target_env_id!r}, "
            f"but evaluation resolved env_id={env_type!r}."
        )
    invalid_cases: List[Dict[str, Any]] = []
    valid_cases: List[Dict[str, Any]] = []

    for case in case_set.get("cases", []):
        case_env_config_map, _ = build_case_env_config(base_env_config_map, env_type, case)
        env = gym.make(env_type, render_mode="rgb_array")
        initial_state: Dict[str, Any] = {}
        reasons: List[str] = []
        try:
            env.unwrapped.configure(case_env_config_map[env_type])
            env.reset(seed=int(case["seed"]))
            initial_state = inspect_benchmark_initial_state(env)
            reasons = validate_benchmark_case(case, initial_state)
        finally:
            env.close()

        item = {
            "case_id": case["case_id"],
            "category": case["category"],
            "seed": int(case["seed"]),
            "difficulty": case.get("difficulty"),
            "case_group": case.get("case_group"),
            "reasons": reasons,
            "initial_state": initial_state,
        }
        if reasons:
            invalid_cases.append(item)
        else:
            valid_cases.append(item)

    summary = {
        "benchmark_name": str(case_set.get("benchmark_name") or ""),
        "total_cases": len(case_set.get("cases", [])),
        "valid_case_count": len(valid_cases),
        "invalid_case_count": len(invalid_cases),
        "valid_categories": sorted({item["category"] for item in valid_cases}),
        "invalid_categories": sorted({item["category"] for item in invalid_cases}),
        "case_group_count": len({str(case.get("case_group") or case.get("category") or "") for case in case_set.get("cases", [])}),
    }
    return {
        "passed": len(invalid_cases) == 0,
        "invalid_cases": invalid_cases,
        "valid_cases": valid_cases,
        "summary": summary,
    }


def bootstrap_ci95(
    values: List[float],
    *,
    iterations: int = BENCHMARK_BOOTSTRAP_ITERATIONS,
    seed: int = BENCHMARK_BOOTSTRAP_SEED,
) -> Optional[List[float]]:
    if not values:
        return None
    values_arr = np.array(list(values), dtype=float)
    if values_arr.size == 1:
        only = round(float(values_arr[0]), 4)
        return [only, only]
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(max(1, int(iterations))):
        sample = rng.choice(values_arr, size=values_arr.size, replace=True)
        means.append(float(np.mean(sample)))
    lower, upper = np.percentile(np.array(means, dtype=float), [2.5, 97.5])
    return [round(float(lower), 4), round(float(upper), 4)]


def compute_benchmark_case_scores(
    *,
    task_completed: bool,
    crashed: bool,
    min_positive_ttc_sec: Optional[float],
    speed_history: List[float],
    completion_time_sec: Optional[float],
    time_limit_sec: float,
) -> Dict[str, Any]:
    completion_rate = 1.0 if bool(task_completed) else 0.0
    if crashed:
        ttc_score = 0.0
    elif min_positive_ttc_sec is None:
        ttc_score = 1.0
    else:
        ttc_score = max(
            0.0,
            min(1.0, float(min_positive_ttc_sec) / BENCHMARK_TTC_SAFE_THRESHOLD_SEC),
        )

    if len(speed_history) <= 1:
        speed_std_mps = 0.0
    else:
        speed_std_mps = float(np.std(np.array(speed_history, dtype=float)))
    speed_variance_score = max(
        0.0,
        min(1.0, 1.0 - (speed_std_mps / BENCHMARK_SPEED_STD_SAFE_MPS)),
    )

    if completion_time_sec is None or float(time_limit_sec) <= 0:
        time_efficiency_score = 0.0
    else:
        time_efficiency_score = max(
            0.0,
            min(1.0, 1.0 - (float(completion_time_sec) / float(time_limit_sec))),
        )

    overall_score = (
        BENCHMARK_OVERALL_WEIGHTS["ttc"] * ttc_score
        + BENCHMARK_OVERALL_WEIGHTS["speed_variance"] * speed_variance_score
        + BENCHMARK_OVERALL_WEIGHTS["time_efficiency"] * time_efficiency_score
    )
    driving_score = 0.0 if crashed else (completion_rate * overall_score)
    return {
        "completion_rate": round(completion_rate, 4),
        "ttc_score": round(ttc_score, 4),
        "speed_std_mps": round(speed_std_mps, 4),
        "speed_variance_score": round(speed_variance_score, 4),
        "time_efficiency_score": round(time_efficiency_score, 4),
        "overall_score": round(overall_score, 4),
        "driving_score": round(driving_score, 4),
    }


def _mean_metric(episodes: List[Dict[str, Any]], key: str) -> float:
    return float(sum(float(item.get(key, 0.0) or 0.0) for item in episodes) / max(len(episodes), 1))


def _failure_reason_counts(episodes: List[Dict[str, Any]]) -> Dict[str, int]:
    failure_reasons: Dict[str, int] = {}
    for episode in episodes:
        reason = str(episode.get("benchmark_failure_reason") or "").strip()
        if reason:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    return failure_reasons


def summarize_benchmark_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episodes:
        return {}

    benchmark_total = len(episodes)
    completion_values = [1.0 if episode.get("task_completed") else 0.0 for episode in episodes]
    driving_values = [float(episode.get("driving_score", 0.0) or 0.0) for episode in episodes]
    driving_v2_values = [
        float(episode.get("driving_score_v2"))
        for episode in episodes
        if episode.get("driving_score_v2") is not None
    ]
    overall_v2_values = [
        float(episode.get("overall_score_v2"))
        for episode in episodes
        if episode.get("overall_score_v2") is not None
    ]
    behavior_penalty_values = [
        float(episode.get("behavior_penalty_factor_v2"))
        for episode in episodes
        if episode.get("behavior_penalty_factor_v2") is not None
    ]
    conservative_penalty_values = [
        float(episode.get("conservative_penalty_severity_v2"))
        for episode in episodes
        if episode.get("conservative_penalty_severity_v2") is not None
    ]
    runtime_penalty_values = [
        float(episode.get("runtime_penalty_severity_v2"))
        for episode in episodes
        if episode.get("runtime_penalty_severity_v2") is not None
    ]

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in sorted({str(episode.get("category") or "uncategorized") for episode in episodes}):
        subset = [episode for episode in episodes if str(episode.get("category") or "uncategorized") == category]
        category_summary = {
            "benchmark_case_count": len(subset),
            "task_completion_count": sum(1 for episode in subset if episode.get("task_completed")),
            "task_completion_rate": round(sum(1 for episode in subset if episode.get("task_completed")) / max(len(subset), 1), 4),
            "ttc_score_mean": round(_mean_metric(subset, "ttc_score"), 4),
            "speed_variance_score_mean": round(_mean_metric(subset, "speed_variance_score"), 4),
            "time_efficiency_score_mean": round(_mean_metric(subset, "time_efficiency_score"), 4),
            "overall_score_mean": round(_mean_metric(subset, "overall_score"), 4),
            "driving_score": round(_mean_metric(subset, "driving_score"), 4),
            "benchmark_failure_reasons": _failure_reason_counts(subset),
        }
        category_driving_v2_values = [
            float(episode.get("driving_score_v2"))
            for episode in subset
            if episode.get("driving_score_v2") is not None
        ]
        category_overall_v2_values = [
            float(episode.get("overall_score_v2"))
            for episode in subset
            if episode.get("overall_score_v2") is not None
        ]
        category_behavior_penalty_values = [
            float(episode.get("behavior_penalty_factor_v2"))
            for episode in subset
            if episode.get("behavior_penalty_factor_v2") is not None
        ]
        if category_driving_v2_values:
            category_summary["driving_score_v2"] = round(
                float(np.mean(np.array(category_driving_v2_values, dtype=float))),
                4,
            )
        if category_overall_v2_values:
            category_summary["overall_score_v2_mean"] = round(
                float(np.mean(np.array(category_overall_v2_values, dtype=float))),
                4,
            )
        if category_behavior_penalty_values:
            category_summary["behavior_penalty_factor_v2_mean"] = round(
                float(np.mean(np.array(category_behavior_penalty_values, dtype=float))),
                4,
            )
        by_category[category] = category_summary

    summary = {
        "benchmark_case_count": benchmark_total,
        "task_completion_count": int(sum(completion_values)),
        "task_completion_rate": round(float(np.mean(np.array(completion_values, dtype=float))), 4),
        "task_completion_rate_ci95": bootstrap_ci95(
            completion_values,
            iterations=BENCHMARK_BOOTSTRAP_ITERATIONS,
            seed=BENCHMARK_BOOTSTRAP_SEED,
        ),
        "ttc_score_mean": round(_mean_metric(episodes, "ttc_score"), 4),
        "speed_variance_score_mean": round(_mean_metric(episodes, "speed_variance_score"), 4),
        "time_efficiency_score_mean": round(_mean_metric(episodes, "time_efficiency_score"), 4),
        "overall_score_mean": round(_mean_metric(episodes, "overall_score"), 4),
        "driving_score": round(float(np.mean(np.array(driving_values, dtype=float))), 4),
        "driving_score_ci95": bootstrap_ci95(
            driving_values,
            iterations=BENCHMARK_BOOTSTRAP_ITERATIONS,
            seed=BENCHMARK_BOOTSTRAP_SEED + 1,
        ),
        "benchmark_failure_reasons": _failure_reason_counts(episodes),
        "benchmark_by_category": by_category,
    }
    if overall_v2_values:
        summary["overall_score_v2_mean"] = round(
            float(np.mean(np.array(overall_v2_values, dtype=float))),
            4,
        )
    if driving_v2_values:
        summary["driving_score_v2"] = round(
            float(np.mean(np.array(driving_v2_values, dtype=float))),
            4,
        )
        summary["driving_score_v2_ci95"] = bootstrap_ci95(
            driving_v2_values,
            iterations=BENCHMARK_BOOTSTRAP_ITERATIONS,
            seed=BENCHMARK_BOOTSTRAP_SEED + 2,
        )
    if behavior_penalty_values:
        summary["behavior_penalty_factor_v2_mean"] = round(
            float(np.mean(np.array(behavior_penalty_values, dtype=float))),
            4,
        )
    if conservative_penalty_values:
        summary["conservative_penalty_severity_v2_mean"] = round(
            float(np.mean(np.array(conservative_penalty_values, dtype=float))),
            4,
        )
    if runtime_penalty_values:
        summary["runtime_penalty_severity_v2_mean"] = round(
            float(np.mean(np.array(runtime_penalty_values, dtype=float))),
            4,
        )
    return summary


def benchmark_result_validity(
    *,
    decision_timeout_rate_mean: Optional[float],
    fallback_action_rate_mean: Optional[float],
    timeout_episode_rate: Optional[float],
) -> Tuple[bool, Optional[str]]:
    reasons: List[str] = []
    if decision_timeout_rate_mean is not None and float(decision_timeout_rate_mean) >= 0.5:
        reasons.append("decision_timeout_rate_mean>=0.5")
    if fallback_action_rate_mean is not None and float(fallback_action_rate_mean) >= 0.5:
        reasons.append("fallback_action_rate_mean>=0.5")
    if timeout_episode_rate is not None and float(timeout_episode_rate) >= 0.5:
        reasons.append("timeout_episode_rate>=0.5")
    if reasons:
        return False, "; ".join(reasons)
    return True, None


class BenchmarkEpisodeEvaluator:
    def __init__(self, case: Dict[str, Any], env) -> None:
        self.case = copy.deepcopy(case)
        self.case_id = str(case["case_id"])
        self.category = str(case["category"])
        self.instruction = str(case["instruction"])
        self.success_criteria = dict(case.get("success_criteria") or {})
        self.time_limit_sec = float(case.get("time_limit_sec") or 0.0)
        self.difficulty = str(case.get("difficulty") or "medium")
        self.case_group = str(case.get("case_group") or self.category)
        self.scenario_family = str(case.get("scenario_family") or _infer_scenario_family_from_env_id(getattr(env.unwrapped, "spec", None).id if getattr(getattr(env.unwrapped, "spec", None), "id", None) else "") or "highway")

        uenv = env.unwrapped
        env_cfg = dict(getattr(uenv, "config", {}) or {})
        self.policy_frequency = float(env_cfg.get("policy_frequency", 1) or 1)
        self.max_steps = benchmark_max_steps(case, env_cfg, default_steps=int(math.ceil(self.time_limit_sec or 1.0)))

        initial_state = inspect_benchmark_initial_state(env)
        self.initial_lane_rank = initial_state.get("initial_lane_rank")
        self.initial_speed_mps = float(initial_state.get("initial_speed_mps") or 0.0)
        self.initial_front_gap_m = initial_state.get("initial_front_gap_m")
        self.initial_front_vehicle_exists = bool(initial_state.get("initial_front_vehicle_exists"))
        self.initial_front_vehicle_is_ahead = bool(initial_state.get("initial_front_vehicle_is_ahead"))
        self.available_actions = list(initial_state.get("available_actions") or [])
        self.initial_x = _vehicle_x(getattr(uenv, "vehicle", None)) or 0.0
        self.initial_front_vehicle_id = None
        self.initial_front_x = None

        ego = getattr(uenv, "vehicle", None)
        road = getattr(uenv, "road", None)
        if ego is not None and road is not None:
            front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
            if front_vehicle is not None:
                self.initial_front_vehicle_id = id(front_vehicle)
                self.initial_front_x = _vehicle_x(front_vehicle)

        self.hold_steps_required = max(1, int(self.success_criteria.get("hold_steps", 2)))
        self.hold_streak = 0
        self.completion_step = None
        self.completion_time_sec = None
        self.task_completed = False
        self.failure_reason = None
        self.visited_left_lane = False
        self.visited_right_lane = False
        self.speed_history: List[float] = []
        self.front_gap_history: List[float] = []
        self.min_positive_ttc_sec = None
        self.max_progress_m = 0.0
        self.last_info: Dict[str, Any] = {}

    def _completion_predicate(self, env, step_metrics: Dict[str, Any]) -> bool:
        ego = getattr(env.unwrapped, "vehicle", None)
        road = getattr(env.unwrapped, "road", None)
        lane_rank = _lane_rank(ego)
        current_speed = float(getattr(ego, "speed", 0.0) or 0.0) if ego is not None else 0.0
        front_gap_m = step_metrics.get("front_gap_m")
        criteria_type = str(self.success_criteria.get("type") or "").strip().lower()

        if criteria_type == "speed_band":
            min_speed = float(self.success_criteria.get("min_speed_mps", 0.0))
            max_speed = float(self.success_criteria.get("max_speed_mps", 999.0))
            return min_speed <= current_speed <= max_speed

        if criteria_type == "front_gap_band":
            if front_gap_m is None:
                return False
            min_gap = float(self.success_criteria.get("min_gap_m", 0.0))
            max_gap = float(self.success_criteria.get("max_gap_m", 1e9))
            return min_gap <= float(front_gap_m) <= max_gap

        if criteria_type == "lane_change":
            if lane_rank is None or self.initial_lane_rank is None:
                return False
            target_offset = _resolve_direction_offset(self.success_criteria)
            target_lane_rank = self.initial_lane_rank + target_offset
            return lane_rank == target_lane_rank

        if criteria_type == "overtake":
            if lane_rank is None or self.initial_lane_rank is None:
                return False
            target_offset = _resolve_direction_offset(self.success_criteria)
            used_required_lane = (
                self.visited_left_lane if target_offset < 0 else self.visited_right_lane
            )
            target_vehicle = _vehicle_by_runtime_id(road, self.initial_front_vehicle_id)
            if target_vehicle is None or ego is None:
                return False
            target_x = _vehicle_x(target_vehicle)
            ego_x = _vehicle_x(ego)
            if target_x is None or ego_x is None:
                return False
            pass_margin_m = float(self.success_criteria.get("pass_margin_m", 5.0))
            return bool(used_required_lane and target_x <= (ego_x - pass_margin_m))

        if criteria_type == "merge_complete":
            if lane_rank is None or self.initial_lane_rank is None:
                return False
            target_offset = _resolve_direction_offset(self.success_criteria)
            target_lane_rank = self.initial_lane_rank + target_offset
            min_progress_m = float(self.success_criteria.get("min_progress_m", 0.0) or 0.0)
            return lane_rank == target_lane_rank and float(self.max_progress_m) >= min_progress_m

        if criteria_type == "arrive":
            has_arrived = getattr(env.unwrapped, "has_arrived", None)
            if callable(has_arrived) and ego is not None:
                try:
                    return bool(has_arrived(ego))
                except TypeError:
                    return bool(has_arrived())
            return bool((self.last_info or {}).get("is_success"))

        return False

    def update(self, env, step_idx: int, step_metrics: Dict[str, Any], crashed: bool, info: Optional[Dict[str, Any]] = None) -> None:
        ego = getattr(env.unwrapped, "vehicle", None)
        self.last_info = dict(info or {})
        lane_rank = _lane_rank(ego)
        if lane_rank is not None and self.initial_lane_rank is not None:
            if lane_rank < self.initial_lane_rank:
                self.visited_left_lane = True
            if lane_rank > self.initial_lane_rank:
                self.visited_right_lane = True

        current_speed = float(getattr(ego, "speed", 0.0) or 0.0) if ego is not None else 0.0
        self.speed_history.append(current_speed)
        front_gap_m = step_metrics.get("front_gap_m")
        if front_gap_m is not None:
            self.front_gap_history.append(float(front_gap_m))
        ttc_sec = step_metrics.get("ttc_sec")
        if ttc_sec is not None and float(ttc_sec) > 0:
            positive_ttc = float(ttc_sec)
            if self.min_positive_ttc_sec is None:
                self.min_positive_ttc_sec = positive_ttc
            else:
                self.min_positive_ttc_sec = min(self.min_positive_ttc_sec, positive_ttc)

        ego_x = _vehicle_x(ego)
        if ego_x is not None:
            self.max_progress_m = max(self.max_progress_m, float(ego_x - self.initial_x))

        if crashed:
            self.failure_reason = self.failure_reason or "crash"
            self.hold_streak = 0
            return

        if self._completion_predicate(env, step_metrics):
            self.hold_streak += 1
            if (not self.task_completed) and self.hold_streak >= self.hold_steps_required:
                self.task_completed = True
                self.completion_step = int(step_idx)
                self.completion_time_sec = round(step_idx / max(self.policy_frequency, 1.0), 3)
        else:
            self.hold_streak = 0

    def finalize(self, crashed: bool, episode_stop_reason: str) -> Dict[str, Any]:
        score_metrics = compute_benchmark_case_scores(
            task_completed=bool(self.task_completed),
            crashed=bool(crashed),
            min_positive_ttc_sec=self.min_positive_ttc_sec,
            speed_history=self.speed_history,
            completion_time_sec=self.completion_time_sec,
            time_limit_sec=self.time_limit_sec,
        )

        if self.failure_reason is None and not self.task_completed:
            if episode_stop_reason == "crash":
                self.failure_reason = "crash"
            elif episode_stop_reason == "episode_timeout_cap":
                self.failure_reason = "episode_timeout_cap"
            elif self.initial_front_vehicle_id is None and str(self.success_criteria.get("type") or "").lower() in {
                "front_gap_band",
                "overtake",
            }:
                self.failure_reason = "missing_initial_front_vehicle"
            else:
                self.failure_reason = "task_not_completed"

        return {
            "case_id": self.case_id,
            "instruction": self.instruction,
            "category": self.category,
            "scenario_family": self.scenario_family,
            "tags": list(self.case.get("tags") or []),
            "difficulty": self.difficulty,
            "case_group": self.case_group,
            "time_limit_sec": round(float(self.time_limit_sec), 3),
            "benchmark_case_env_overrides": copy.deepcopy(self.case.get("env_overrides") or {}),
            "benchmark_success_criteria": copy.deepcopy(self.success_criteria),
            "benchmark_initial_lane_rank": self.initial_lane_rank,
            "benchmark_initial_front_gap_m": (
                round(float(self.initial_front_gap_m), 4)
                if self.initial_front_gap_m is not None
                else None
            ),
            "benchmark_completion_step": self.completion_step,
            "benchmark_completion_time_sec": self.completion_time_sec,
            "task_completed": bool(self.task_completed),
            "completion_rate": score_metrics["completion_rate"],
            "ttc_score": score_metrics["ttc_score"],
            "speed_variance_score": score_metrics["speed_variance_score"],
            "time_efficiency_score": score_metrics["time_efficiency_score"],
            "overall_score": score_metrics["overall_score"],
            "driving_score": score_metrics["driving_score"],
            "benchmark_failure_reason": self.failure_reason,
            "benchmark_speed_std_mps": score_metrics["speed_std_mps"],
            "benchmark_min_positive_ttc_sec": (
                round(float(self.min_positive_ttc_sec), 4)
                if self.min_positive_ttc_sec is not None
                else None
            ),
            "benchmark_max_progress_m": round(float(self.max_progress_m), 4),
        }
