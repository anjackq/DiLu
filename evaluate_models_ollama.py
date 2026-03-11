import argparse
import copy
import fnmatch
import json
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.wrappers import RecordVideo
from rich import print

from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.runtime import (
    configure_runtime_env,
    build_highway_env_config,
    DEFAULT_DILU_SEEDS,
    ensure_dir,
    ensure_parent_dir,
    timestamped_results_path,
    current_timestamp,
    slugify_model_name,
    build_experiment_root,
    build_model_root,
    build_model_run_dir,
    ensure_experiment_layout,
    write_json_atomic,
    read_json,
)
from dilu.scenario.envScenario import EnvScenario


STRICT_RESPONSE_PATTERN = re.compile(r"Response to user:\s*\#{4}\s*([0-4])\s*$", re.IGNORECASE)


def build_env_config(config: Dict) -> Dict:
    return build_highway_env_config(
        config,
        show_trajectories=False,
        render_agent=False,
    )


def parse_seeds(raw: Optional[str]) -> List[int]:
    if not raw:
        return DEFAULT_DILU_SEEDS
    seeds = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model_override(model_name: str, overrides: Dict) -> Dict:
    if not isinstance(overrides, dict):
        return {}
    lower_name = str(model_name or "").strip().lower()
    if not lower_name:
        return {}

    for key, value in overrides.items():
        if str(key).strip().lower() == lower_name and isinstance(value, dict):
            return dict(value)

    best_key = None
    best_value = None
    best_len = -1
    for key, value in overrides.items():
        if not isinstance(value, dict):
            continue
        key_str = str(key).strip()
        if not key_str:
            continue
        key_lower = key_str.lower()
        matched = False
        if "*" in key_lower or "?" in key_lower:
            matched = fnmatch.fnmatchcase(lower_name, key_lower)
        else:
            matched = lower_name.startswith(key_lower)
        if matched and len(key_lower) > best_len:
            best_key = key_str
            best_value = value
            best_len = len(key_lower)

    if best_key is None:
        return {}
    return dict(best_value or {})


def _safe_int_action(action) -> int:
    if isinstance(action, str):
        action = action.strip()
    action = int(action)
    if action < 0 or action > 4:
        raise ValueError(f"Invalid action id: {action}")
    return action


def _response_format_metrics(response_content: str) -> Dict:
    response_content = (response_content or "").strip()
    has_delimiter = "####" in response_content
    strict_match = STRICT_RESPONSE_PATTERN.search(response_content)
    direct_action_parseable = False
    parsed_action = None

    if has_delimiter:
        tail = response_content.split("####")[-1].strip()
        try:
            parsed_action = int(tail)
            if 0 <= parsed_action <= 4:
                direct_action_parseable = True
        except Exception:
            direct_action_parseable = False

    return {
        "has_delimiter": has_delimiter,
        "strict_format_match": bool(strict_match),
        "direct_action_parseable": direct_action_parseable,
        "strict_action": int(strict_match.group(1)) if strict_match else None,
        "direct_parsed_action": parsed_action,
    }


def extract_step_traffic_metrics(env, ttc_threshold_sec: float, headway_threshold_m: float) -> Dict:
    ego_speed_mps = None
    front_gap_m = None
    relative_speed_mps = None
    ttc_sec = None
    ttc_danger = False
    headway_violation = False

    try:
        uenv = env.unwrapped
        ego = getattr(uenv, "vehicle", None)
        road = getattr(uenv, "road", None)
        if ego is not None:
            ego_speed_mps = float(getattr(ego, "speed", 0.0))
        if ego is not None and road is not None:
            front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
            if front_vehicle is not None:
                front_gap_m = float(np.linalg.norm(ego.position - front_vehicle.position))
                relative_speed_mps = float(ego.speed - front_vehicle.speed)
                if relative_speed_mps > 0:
                    ttc_sec = front_gap_m / max(relative_speed_mps, 1e-6)
                    ttc_danger = bool(ttc_sec < ttc_threshold_sec)
                headway_violation = bool(front_gap_m < headway_threshold_m)
    except Exception:
        pass

    return {
        "ego_speed_mps": ego_speed_mps,
        "front_gap_m": front_gap_m,
        "relative_speed_mps": relative_speed_mps,
        "ttc_sec": ttc_sec,
        "ttc_danger": ttc_danger,
        "headway_violation": headway_violation,
    }


def run_episode(
    config: Dict,
    env_config: Dict,
    agent_memory: DrivingMemory,
    seed: int,
    few_shot_num: int,
    temp_dir: str,
    ttc_threshold_sec: float,
    headway_threshold_m: float,
    alignment_sample_rate: float,
    alignment_max_samples: int,
    slow_decision_threshold_sec: float,
    save_artifacts: bool = False,
    run_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict:
    env_type = "highway-v0"
    env = None
    result_prefix = f"highway_seed_{seed}"
    if save_artifacts:
        if not run_dir:
            raise ValueError("run_dir is required when save_artifacts is enabled.")
        ensure_dir(run_dir)
        database_path = os.path.join(run_dir, f"{result_prefix}.db")
    else:
        database_path = os.path.join(temp_dir, f"eval_{seed}_{int(time.time() * 1000)}.db")
    started = time.time()
    error = None
    crashed = False
    truncated = False
    terminated = False
    steps = 0
    final_info = {}
    episode_stop_reason = "completed"
    decisions_made = 0
    decision_calls_total = 0
    decision_timeout_count = 0
    fallback_action_count = 0
    first_timeout_step = None
    ollama_requested_think_mode = None
    ollama_effective_think_modes_seen = set()
    ollama_native_retry_count = 0
    ollama_openai_fallback_count = 0
    ollama_native_decision_count = 0
    ollama_native_timeout_count = 0
    ollama_native_timeout_short_circuit_count = 0
    responses_with_delimiter = 0
    responses_strict_format = 0
    responses_direct_parseable = 0
    format_failure_count = 0
    episode_reward_sum = 0.0
    ego_speed_sum = 0.0
    ego_speed_count = 0
    ttc_danger_steps = 0
    headway_violation_steps = 0
    lane_change_count = 0
    flap_accel_decel_count = 0
    prev_action_id = None
    alignment_samples = []
    decision_latencies_sec = []
    slow_decision_count = 0

    try:
        env = gym.make(env_type, render_mode="rgb_array")
        env.configure(env_config[env_type])
        if save_artifacts:
            env = RecordVideo(
                env,
                run_dir,
                episode_trigger=lambda episode_id: True,
                name_prefix=result_prefix,
            )
            try:
                env.unwrapped.set_record_video_wrapper(env)
            except Exception:
                pass
        obs, info = env.reset(seed=seed)
        final_info = info

        sce = EnvScenario(env, env_type, seed, database_path)
        agent = DriverAgent(sce, verbose=False)

        prev_action = "Not available"
        for frame_id in range(config["simulation_duration"]):
            _ = np.array(obs, dtype=float)

            fewshot_results = (
                agent_memory.retriveMemory(sce, frame_id, few_shot_num)
                if few_shot_num > 0 else []
            )
            fewshot_messages = [x["human_question"] for x in fewshot_results]
            fewshot_answers = [x["LLM_response"] for x in fewshot_results]

            sce_descrip = sce.describe(frame_id)
            avail_action = sce.availableActionsDescription()
            action, response, human_question, fewshot_answer = agent.few_shot_decision(
                scenario_description=sce_descrip,
                available_actions=avail_action,
                previous_decisions=prev_action,
                fewshot_messages=fewshot_messages,
                driving_intensions="Drive safely and avoid collisons",
                fewshot_answers=fewshot_answers,
            )
            prev_action = action
            decision_calls_total += 1
            decisions_made += 1
            decision_meta = getattr(agent, "last_decision_meta", {}) or {}
            timed_out = bool(decision_meta.get("timed_out", False))
            used_fallback = bool(decision_meta.get("used_fallback", False))
            ollama_requested_think_mode = decision_meta.get("ollama_requested_think_mode") or ollama_requested_think_mode
            ollama_effective_mode = decision_meta.get("ollama_effective_think_mode")
            ollama_transport = decision_meta.get("ollama_transport")
            ollama_native_retry_used = bool(decision_meta.get("ollama_native_retry_used", False))
            ollama_native_timeout = bool(decision_meta.get("ollama_native_timeout", False))
            ollama_native_timeout_short_circuit = bool(
                decision_meta.get("ollama_native_timeout_short_circuit", False)
            )
            decision_elapsed_sec = float(decision_meta.get("decision_elapsed_sec", 0.0) or 0.0)
            decision_timeout_count += int(timed_out)
            fallback_action_count += int(used_fallback)
            ollama_native_retry_count += int(ollama_native_retry_used)
            ollama_openai_fallback_count += int(ollama_transport == "openai_compat_fallback")
            ollama_native_decision_count += int(ollama_transport == "native")
            ollama_native_timeout_count += int(ollama_native_timeout)
            ollama_native_timeout_short_circuit_count += int(ollama_native_timeout_short_circuit)
            decision_latencies_sec.append(decision_elapsed_sec)
            slow_decision_count += int(decision_elapsed_sec >= max(0.001, slow_decision_threshold_sec))
            if ollama_effective_mode:
                ollama_effective_think_modes_seen.add(str(ollama_effective_mode))
            if timed_out and first_timeout_step is None:
                first_timeout_step = int(frame_id)

            fmt = _response_format_metrics(response)
            responses_with_delimiter += int(fmt["has_delimiter"])
            responses_strict_format += int(fmt["strict_format_match"])
            responses_direct_parseable += int(fmt["direct_action_parseable"])
            format_failure_count += int(not fmt["strict_format_match"])

            action = _safe_int_action(action)
            lane_change_count += int(action in (0, 2))
            if prev_action_id is not None and ((prev_action_id == 3 and action == 4) or (prev_action_id == 4 and action == 3)):
                flap_accel_decel_count += 1
            prev_action_id = action

            if alignment_sample_rate > 0 and len(alignment_samples) < alignment_max_samples and random.random() < alignment_sample_rate:
                alignment_samples.append({
                    "scenario_summary": (sce_descrip or "")[:800],
                    "model_response": response,
                    "action_id": int(action),
                    "step_idx": int(frame_id),
                    "seed": int(seed),
                })

            obs, reward, terminated, truncated, info = env.step(action)
            final_info = info
            crashed = bool(info.get("crashed", False))
            done = terminated or truncated
            steps += 1
            episode_reward_sum += float(reward)

            step_metrics = extract_step_traffic_metrics(env, ttc_threshold_sec, headway_threshold_m)
            if step_metrics["ego_speed_mps"] is not None:
                ego_speed_sum += float(step_metrics["ego_speed_mps"])
                ego_speed_count += 1
            ttc_danger_steps += int(step_metrics["ttc_danger"])
            headway_violation_steps += int(step_metrics["headway_violation"])

            # Keep DB prompt logs for replay/debugging if needed.
            try:
                sce.promptsCommit(frame_id, None, done, human_question, fewshot_answer, response)
            except Exception:
                pass

            if done:
                break

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        episode_stop_reason = "error"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if (not save_artifacts) and os.path.exists(database_path):
            try:
                os.remove(database_path)
            except Exception:
                pass

    duration_sec = time.time() - started
    episode_reward_avg = episode_reward_sum / max(steps, 1)
    avg_ego_speed_mps = ego_speed_sum / max(ego_speed_count, 1)
    ttc_danger_rate = ttc_danger_steps / max(steps, 1)
    headway_violation_rate = headway_violation_steps / max(steps, 1)
    lane_change_rate = lane_change_count / max(steps, 1)
    flap_accel_decel_rate = flap_accel_decel_count / max(steps, 1)
    decision_latency_ms_avg = (duration_sec / max(steps, 1)) * 1000.0
    format_failure_rate = format_failure_count / max(decisions_made, 1)
    decision_timeout_rate = decision_timeout_count / max(decision_calls_total, 1)
    fallback_action_rate = fallback_action_count / max(decision_calls_total, 1)
    ollama_native_retry_rate = ollama_native_retry_count / max(decision_calls_total, 1)
    ollama_openai_fallback_rate = ollama_openai_fallback_count / max(decision_calls_total, 1)
    p95_decision_latency_sec = float(np.percentile(decision_latencies_sec, 95)) if decision_latencies_sec else 0.0
    timeout_triggered = decision_timeout_count > 0

    if episode_stop_reason != "error":
        if crashed:
            episode_stop_reason = "crash"
        elif truncated:
            episode_stop_reason = "truncated"
        elif terminated:
            episode_stop_reason = "terminated"
        else:
            episode_stop_reason = "completed"

    return {
        "seed": seed,
        "steps": steps,
        "max_steps": int(config["simulation_duration"]),
        "crashed": crashed,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success_no_collision": (error is None and not crashed),
        "episode_runtime_sec": round(duration_sec, 3),
        "avg_step_runtime_sec": round(duration_sec / max(steps, 1), 3),
        "episode_stop_reason": episode_stop_reason,
        "timeout_triggered": bool(timeout_triggered),
        "first_timeout_step": first_timeout_step,
        "decision_calls_total": decision_calls_total,
        "decision_timeout_count": decision_timeout_count,
        "decision_timeout_rate": round(decision_timeout_rate, 4),
        "fallback_action_count": fallback_action_count,
        "fallback_action_rate": round(fallback_action_rate, 4),
        "ollama_requested_think_mode": ollama_requested_think_mode,
        "ollama_effective_think_modes_seen": sorted(ollama_effective_think_modes_seen),
        "ollama_native_retry_count": ollama_native_retry_count,
        "ollama_native_retry_rate": round(ollama_native_retry_rate, 4),
        "ollama_openai_fallback_count": ollama_openai_fallback_count,
        "ollama_openai_fallback_rate": round(ollama_openai_fallback_rate, 4),
        "ollama_native_decision_count": ollama_native_decision_count,
        "ollama_native_timeout_count": ollama_native_timeout_count,
        "ollama_native_timeout_short_circuit_count": ollama_native_timeout_short_circuit_count,
        "ollama_downgrade_triggered": bool(ollama_native_retry_count > 0 or ("auto" in ollama_effective_think_modes_seen and ollama_requested_think_mode == "think")),
        "slow_decision_count": int(slow_decision_count),
        "p95_decision_latency_sec": round(p95_decision_latency_sec, 4),
        "decisions_made": decisions_made,
        "responses_with_delimiter": responses_with_delimiter,
        "responses_strict_format": responses_strict_format,
        "responses_direct_parseable": responses_direct_parseable,
        "format_failure_count": format_failure_count,
        "format_failure_rate": round(format_failure_rate, 4),
        "episode_reward_sum": round(episode_reward_sum, 4),
        "episode_reward_avg": round(episode_reward_avg, 4),
        "avg_ego_speed_mps": round(avg_ego_speed_mps, 4),
        "ttc_danger_steps": ttc_danger_steps,
        "ttc_danger_rate": round(ttc_danger_rate, 4),
        "headway_violation_steps": headway_violation_steps,
        "headway_violation_rate": round(headway_violation_rate, 4),
        "lane_change_count": lane_change_count,
        "lane_change_rate": round(lane_change_rate, 4),
        "flap_accel_decel_count": flap_accel_decel_count,
        "flap_accel_decel_rate": round(flap_accel_decel_rate, 4),
        "decision_latency_ms_avg": round(decision_latency_ms_avg, 3),
        "alignment_samples": alignment_samples,
        "model": model_name,
        "database_path": database_path if save_artifacts else None,
        "video_prefix": result_prefix if save_artifacts else None,
        "run_id": run_id if save_artifacts else None,
        "run_dir": run_dir if save_artifacts else None,
        "error": error,
        "final_info": copy.deepcopy(final_info),
    }


def aggregate_results(model_name: str, episodes: List[Dict]) -> Dict:
    total = len(episodes)
    crashes = sum(1 for e in episodes if e["crashed"])
    errors = sum(1 for e in episodes if e["error"])
    no_collision = sum(1 for e in episodes if e["success_no_collision"])
    truncations = sum(1 for e in episodes if e["truncated"])
    terminations = sum(1 for e in episodes if e["terminated"])
    total_steps = sum(e["steps"] for e in episodes)
    total_runtime = sum(e["episode_runtime_sec"] for e in episodes)
    total_decisions = sum(e.get("decisions_made", 0) for e in episodes)
    total_decision_calls = sum(e.get("decision_calls_total", e.get("decisions_made", 0)) for e in episodes)
    total_decision_timeouts = sum(e.get("decision_timeout_count", 0) for e in episodes)
    timeout_episode_count = sum(1 for e in episodes if e.get("timeout_triggered", False))
    total_fallback_actions = sum(e.get("fallback_action_count", 0) for e in episodes)
    total_ollama_native_retries = sum(e.get("ollama_native_retry_count", 0) for e in episodes)
    total_ollama_openai_fallbacks = sum(e.get("ollama_openai_fallback_count", 0) for e in episodes)
    total_ollama_native_decisions = sum(e.get("ollama_native_decision_count", 0) for e in episodes)
    total_ollama_native_timeouts = sum(e.get("ollama_native_timeout_count", 0) for e in episodes)
    total_ollama_native_timeout_short_circuits = sum(
        e.get("ollama_native_timeout_short_circuit_count", 0) for e in episodes
    )
    ollama_native_timeout_episode_count = sum(1 for e in episodes if e.get("ollama_native_timeout_count", 0) > 0)
    ollama_downgrade_episode_count = sum(1 for e in episodes if e.get("ollama_downgrade_triggered", False))
    timeout_cap_stops = sum(1 for e in episodes if e.get("episode_stop_reason") == "episode_timeout_cap")
    total_delimiters = sum(e.get("responses_with_delimiter", 0) for e in episodes)
    total_strict = sum(e.get("responses_strict_format", 0) for e in episodes)
    total_direct = sum(e.get("responses_direct_parseable", 0) for e in episodes)
    total_format_failures = sum(e.get("format_failure_count", 0) for e in episodes)
    total_reward_sum = sum(float(e.get("episode_reward_sum", 0.0)) for e in episodes)
    total_speed = sum(float(e.get("avg_ego_speed_mps", 0.0)) for e in episodes)
    total_ttc_danger_rate = sum(float(e.get("ttc_danger_rate", 0.0)) for e in episodes)
    total_headway_rate = sum(float(e.get("headway_violation_rate", 0.0)) for e in episodes)
    total_lane_change_rate = sum(float(e.get("lane_change_rate", 0.0)) for e in episodes)
    total_flap_rate = sum(float(e.get("flap_accel_decel_rate", 0.0)) for e in episodes)
    total_decision_latency_ms = sum(float(e.get("decision_latency_ms_avg", 0.0)) for e in episodes)

    return {
        "model": model_name,
        "episodes": total,
        "crashes": crashes,
        "errors": errors,
        "no_collision_episodes": no_collision,
        "crash_rate": round(crashes / total, 4) if total else None,
        "no_collision_rate": round(no_collision / total, 4) if total else None,
        "error_rate": round(errors / total, 4) if total else None,
        "truncation_count": truncations,
        "termination_count": terminations,
        "avg_steps": round(total_steps / total, 2) if total else None,
        "avg_episode_runtime_sec": round(total_runtime / total, 3) if total else None,
        "avg_step_runtime_sec": round(total_runtime / max(total_steps, 1), 3),
        "decisions_total": total_decisions,
        "decision_calls_total": total_decision_calls,
        "decision_timeouts_total": total_decision_timeouts,
        "decision_timeout_rate_mean": round(total_decision_timeouts / max(total_decision_calls, 1), 4),
        "timeout_episode_count": timeout_episode_count,
        "timeout_episode_rate": round(timeout_episode_count / total, 4) if total else None,
        "fallback_actions_total": total_fallback_actions,
        "fallback_action_rate_mean": round(total_fallback_actions / max(total_decision_calls, 1), 4),
        "ollama_native_retries_total": total_ollama_native_retries,
        "ollama_native_retry_rate_mean": round(total_ollama_native_retries / max(total_decision_calls, 1), 4),
        "ollama_openai_fallbacks_total": total_ollama_openai_fallbacks,
        "ollama_openai_fallback_rate_mean": round(total_ollama_openai_fallbacks / max(total_decision_calls, 1), 4),
        "ollama_native_decisions_total": total_ollama_native_decisions,
        "ollama_native_decision_rate_mean": round(total_ollama_native_decisions / max(total_decision_calls, 1), 4),
        "ollama_native_timeouts_total": total_ollama_native_timeouts,
        "ollama_native_timeout_rate_mean": round(total_ollama_native_timeouts / max(total_decision_calls, 1), 4),
        "ollama_native_timeout_short_circuits_total": total_ollama_native_timeout_short_circuits,
        "ollama_native_timeout_episode_count": ollama_native_timeout_episode_count,
        "ollama_native_timeout_episode_rate": round(ollama_native_timeout_episode_count / total, 4) if total else None,
        "ollama_downgrade_episode_count": ollama_downgrade_episode_count,
        "ollama_downgrade_episode_rate": round(ollama_downgrade_episode_count / total, 4) if total else None,
        "episodes_stopped_by_timeout_cap": timeout_cap_stops,
        "response_delimiter_rate": round(total_delimiters / total_decisions, 4) if total_decisions else None,
        "response_strict_format_rate": round(total_strict / total_decisions, 4) if total_decisions else None,
        "response_direct_parseable_rate": round(total_direct / total_decisions, 4) if total_decisions else None,
        "avg_reward_sum": round(total_reward_sum / total, 4) if total else None,
        "avg_reward_per_step": round(total_reward_sum / max(total_steps, 1), 4),
        "avg_ego_speed_mps": round(total_speed / total, 4) if total else None,
        "ttc_danger_rate_mean": round(total_ttc_danger_rate / total, 4) if total else None,
        "headway_violation_rate_mean": round(total_headway_rate / total, 4) if total else None,
        "lane_change_rate_mean": round(total_lane_change_rate / total, 4) if total else None,
        "flap_accel_decel_rate_mean": round(total_flap_rate / total, 4) if total else None,
        "format_failure_rate_mean": round(total_format_failures / max(total_decisions, 1), 4),
        "decision_latency_ms_avg": round(total_decision_latency_ms / total, 3) if total else None,
    }


def _append_eval_run_log(log_path: str, model_name: str, episode: Dict) -> None:
    ensure_parent_dir(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            "Model {model} | Seed {seed} | Stop {stop} | Steps {steps}/{max_steps} | "
            "Crash {crashed} | Error {error} | Runtime {runtime}s | DB {db} | Video {video}\n".format(
                model=model_name,
                seed=episode.get("seed"),
                stop=episode.get("episode_stop_reason"),
                steps=episode.get("steps"),
                max_steps=episode.get("max_steps"),
                crashed=episode.get("crashed"),
                error=episode.get("error"),
                runtime=episode.get("episode_runtime_sec"),
                db=episode.get("database_path"),
                video=episode.get("video_prefix"),
            )
        )


def _build_eval_run_metrics_report(
    model_name: str,
    experiment_id: str,
    experiment_root: Optional[str],
    run_id: str,
    run_dir: str,
    config_path: str,
    openai_api_type: str,
    few_shot_num: int,
    memory_path: str,
    simulation_duration: int,
    metrics_config: Dict,
    episodes: List[Dict],
    aggregate: Dict,
) -> Dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_schema_version": "2.0",
        "source": "evaluate_models_ollama:run_artifacts",
        "config_path": config_path,
        "openai_api_type": openai_api_type,
        "model": model_name,
        "experiment_id": experiment_id,
        "experiment_root": experiment_root,
        "run_id": run_id,
        "run_dir": run_dir,
        "few_shot_num": int(few_shot_num),
        "memory_path": memory_path,
        "simulation_duration": int(simulation_duration),
        "metrics_config": metrics_config,
        "episodes": episodes,
        "aggregate": aggregate,
    }


def _build_model_extract(
    model_name: str,
    experiment_id: str,
    source_compare_report: str,
    aggregate: Dict,
    episodes: List[Dict],
    metrics_config: Dict,
) -> Dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_schema_version": "2.0",
        "source": "evaluate_models_ollama:model_extract",
        "experiment_id": experiment_id,
        "model": model_name,
        "source_compare_report": source_compare_report,
        "aggregate": aggregate,
        "episodes": episodes,
        "metrics_config": metrics_config,
    }


def _update_experiment_manifest_for_eval(
    experiment_root: str,
    experiment_id: str,
    config_path: str,
    memory_path: str,
    few_shot_num: int,
    simulation_duration: int,
    compare_report_path: str,
    model_summaries: Dict[str, Dict[str, str]],
    model_run_outputs: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    manifest_path = os.path.join(experiment_root, "manifest.json")
    manifest = read_json(manifest_path, default={})

    manifest.setdefault("experiment_id", experiment_id)
    manifest.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["config_path"] = config_path
    manifest["memory_path"] = memory_path
    manifest["few_shot_num"] = int(few_shot_num)
    manifest["simulation_duration"] = int(simulation_duration)

    compare_meta = manifest.setdefault("compare", {})
    compare_meta["latest_report"] = compare_report_path
    compare_meta.setdefault("history", [])
    if compare_report_path not in compare_meta["history"]:
        compare_meta["history"].append(compare_report_path)

    models = manifest.setdefault("models", {})
    for model_name, paths in model_summaries.items():
        model_item = models.setdefault(model_name, {})
        model_item.setdefault("slug", slugify_model_name(model_name))
        model_item.setdefault("root", os.path.join(experiment_root, "models", slugify_model_name(model_name)))
        model_item["latest_eval_summary"] = paths["summary"]
        model_item["latest_eval_episodes"] = paths["episodes"]
        if model_run_outputs and model_name in model_run_outputs:
            run_paths = model_run_outputs[model_name]
            model_item["latest_eval_run_id"] = run_paths.get("run_id")
            model_item["latest_eval_run_dir"] = run_paths.get("run_dir")
            model_item["latest_eval_run_metrics"] = run_paths.get("run_metrics")

    write_json_atomic(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DiLu agent behavior across Ollama models on fixed seeds.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to compare (e.g. deepseek-r1:14b dilu-llama3_1-8b-v1)")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to DiLu fixed seed list.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of seeds after parsing.")
    parser.add_argument("--few-shot-num", type=int, default=None, help="Override config few_shot_num.")
    parser.add_argument("--memory-path", default=None, help="Override config memory_path.")
    parser.add_argument("--output", default=None, help="Write JSON report to this file (default: results/eval_compare_<timestamp>.json)")
    parser.add_argument("--experiment-id", default=None, help="Experiment id. Defaults to config or timestamp.")
    parser.add_argument("--results-root", default=None, help="Structured results root. Defaults to config or results/experiments.")
    parser.add_argument("--output-root", default=None, help="Optional compare-output folder override.")
    parser.add_argument("--no-structured-output", action="store_true", help="Disable structured experiment/model outputs.")
    parser.add_argument("--save-run-artifacts", action="store_true", help="Save run-style artifacts (video/db/log/run_metrics) per model during evaluation.")
    parser.add_argument("--eval-run-id", default=None, help="Run id used under models/<slug>/runs/<eval_run_id> when --save-run-artifacts is enabled.")
    parser.add_argument("--decision-timeout-sec", type=float, default=None, help="Hard timeout per model decision call. Default: config eval_decision_timeout_sec or 60.")
    parser.add_argument("--decision-max-output-tokens", type=int, default=None, help="Per-decision max output tokens. Default: config eval_decision_max_output_tokens or 512.")
    parser.add_argument("--disable-streaming", action="store_true", help="Disable streaming inference in evaluation to reduce hangs.")
    parser.add_argument("--disable-checker-llm", action="store_true", help="Disable second checker LLM call; use local parse/fallback only.")
    parser.add_argument("--ollama-think-mode", choices=["auto", "think", "no_think"], default=None, help="Ollama native think mode override for driver agent.")
    parser.add_argument("--ollama-use-native-chat", action="store_true", help="Force native Ollama /api/chat driver path.")
    parser.add_argument("--ollama-disable-native-chat", action="store_true", help="Disable native Ollama /api/chat and use OpenAI-compatible /v1 path.")
    parser.add_argument("--alignment-sample-rate", type=float, default=0.0, help="Sampling probability [0,1] for reasoning-alignment sample collection.")
    parser.add_argument("--alignment-max-samples", type=int, default=0, help="Max alignment samples per model.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seeds = parse_seeds(args.seeds)
    if args.limit is not None:
        seeds = seeds[:args.limit]
    if not seeds:
        raise ValueError("No seeds to evaluate.")

    few_shot_num = config["few_shot_num"] if args.few_shot_num is None else args.few_shot_num
    if args.memory_path:
        config["memory_path"] = args.memory_path
    ttc_threshold_sec = float(config.get("metrics_ttc_threshold_sec", 2.0))
    headway_threshold_m = float(config.get("metrics_headway_threshold_m", 15.0))
    alignment_sample_rate = max(0.0, min(1.0, float(args.alignment_sample_rate)))
    alignment_max_samples = max(0, int(args.alignment_max_samples))
    structured_output = not args.no_structured_output
    save_run_artifacts = bool(config.get("eval_save_run_artifacts", False)) or bool(args.save_run_artifacts)
    eval_run_id = str(
        args.eval_run_id
        or config.get("eval_run_id")
        or f"eval_run_{current_timestamp()}"
    ).strip()
    if not eval_run_id:
        eval_run_id = f"eval_run_{current_timestamp()}"
    if save_run_artifacts and not structured_output:
        raise ValueError("--save-run-artifacts requires structured output. Remove --no-structured-output.")
    default_decision_timeout_sec = float(config.get("eval_decision_timeout_sec", 60.0))
    default_decision_max_output_tokens = int(config.get("eval_decision_max_output_tokens", 512))
    default_disable_streaming = bool(config.get("eval_disable_streaming", True))
    default_disable_checker_llm = bool(config.get("eval_disable_checker_llm", True))
    default_ollama_think_mode = str(config.get("OLLAMA_THINK_MODE", "auto")).strip().lower()
    if default_ollama_think_mode not in {"auto", "think", "no_think"}:
        default_ollama_think_mode = "auto"
    default_ollama_use_native_chat = bool(config.get("OLLAMA_USE_NATIVE_CHAT", True))
    default_ollama_native_chat_timeout_sec = float(
        config.get("OLLAMA_NATIVE_CHAT_TIMEOUT_SEC", default_decision_timeout_sec)
    )
    slow_decision_threshold_sec = float(config.get("eval_slow_decision_threshold_sec", 5.0))
    model_overrides = config.get("eval_model_overrides", {})
    if not isinstance(model_overrides, dict):
        model_overrides = {}

    cli_decision_timeout_sec = (
        float(args.decision_timeout_sec) if args.decision_timeout_sec is not None else None
    )
    cli_decision_max_output_tokens = (
        int(args.decision_max_output_tokens) if args.decision_max_output_tokens is not None else None
    )
    cli_disable_streaming = bool(args.disable_streaming)
    cli_disable_checker_llm = bool(args.disable_checker_llm)
    cli_ollama_think_mode = str(args.ollama_think_mode).strip().lower() if args.ollama_think_mode else None
    if cli_ollama_think_mode and cli_ollama_think_mode not in {"auto", "think", "no_think"}:
        cli_ollama_think_mode = "auto"
    cli_ollama_use_native_chat = None
    if args.ollama_use_native_chat:
        cli_ollama_use_native_chat = True
    if args.ollama_disable_native_chat:
        cli_ollama_use_native_chat = False

    config["eval_save_run_artifacts"] = bool(save_run_artifacts)
    config["eval_run_id"] = eval_run_id

    results_root = (
        args.results_root
        or config.get("results_root")
        or os.path.join("results", "experiments")
    )
    experiment_id = (
        args.experiment_id
        or config.get("experiment_id")
        or current_timestamp()
    )

    experiment_root = None
    model_roots: Dict[str, str] = {}
    compare_dir = None
    if structured_output:
        experiment_root = build_experiment_root(results_root, experiment_id)
        experiment_id = os.path.basename(experiment_root)
        model_roots = ensure_experiment_layout(experiment_root, args.models)
        compare_dir = ensure_dir(args.output_root) if args.output_root else ensure_dir(os.path.join(experiment_root, "compare"))
    else:
        compare_dir = ensure_dir(args.output_root) if args.output_root else ensure_dir("results")

    env_config = build_env_config(config)
    temp_dir = ensure_dir(os.path.join("temp", "eval_compare"))

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_schema_version": "2.0",
        "config_path": args.config,
        "source": "evaluate_models_ollama",
        "experiment_id": experiment_id,
        "experiment_root": experiment_root,
        "compare_dir": compare_dir,
        "structured_output": structured_output,
        "save_run_artifacts": bool(save_run_artifacts),
        "eval_run_id": eval_run_id if save_run_artifacts else None,
        "openai_api_type": config["OPENAI_API_TYPE"],
        "models": args.models,
        "model_roots": model_roots,
        "seeds": seeds,
        "few_shot_num": few_shot_num,
        "memory_path": config["memory_path"],
        "simulation_duration": int(config["simulation_duration"]),
        "metrics_config": {
            "ttc_threshold_sec": ttc_threshold_sec,
            "headway_threshold_m": headway_threshold_m,
            "flapping_mode": "accel_decel",
            "decision_timeout_sec": round(max(1.0, default_decision_timeout_sec), 3),
            "decision_max_output_tokens": int(max(32, default_decision_max_output_tokens)),
            "disable_streaming": bool(default_disable_streaming),
            "disable_checker_llm": bool(default_disable_checker_llm),
            "ollama_think_mode": default_ollama_think_mode,
            "ollama_use_native_chat": bool(default_ollama_use_native_chat),
            "ollama_native_chat_timeout_sec": round(max(1.0, default_ollama_native_chat_timeout_sec), 3),
            "slow_decision_threshold_sec": round(max(0.001, slow_decision_threshold_sec), 3),
            "model_overrides_enabled": bool(model_overrides),
            "model_overrides_keys": sorted(list(model_overrides.keys())),
            "save_run_artifacts": bool(save_run_artifacts),
            "eval_run_id": eval_run_id if save_run_artifacts else None,
            "alignment_sample_rate": alignment_sample_rate,
            "alignment_max_samples": alignment_max_samples,
        },
        "per_model": {},
        "aggregates": [],
        "alignment_samples": [],
        "model_eval_outputs": {},
        "model_run_outputs": {},
        "model_runtime_policies": {},
    }

    aggregate_by_model: Dict[str, Dict] = {}
    model_run_outputs: Dict[str, Dict[str, str]] = {}
    model_metrics_configs: Dict[str, Dict] = {}
    for model_name in args.models:
        model_override = _resolve_model_override(model_name, model_overrides)
        resolved_decision_timeout_sec = float(
            cli_decision_timeout_sec
            if cli_decision_timeout_sec is not None
            else model_override.get("decision_timeout_sec", default_decision_timeout_sec)
        )
        resolved_decision_timeout_sec = max(1.0, resolved_decision_timeout_sec)
        resolved_decision_max_output_tokens = int(
            cli_decision_max_output_tokens
            if cli_decision_max_output_tokens is not None
            else model_override.get("decision_max_output_tokens", default_decision_max_output_tokens)
        )
        resolved_decision_max_output_tokens = max(32, resolved_decision_max_output_tokens)
        resolved_disable_streaming = (
            True if cli_disable_streaming else _as_bool(model_override.get("disable_streaming"), default_disable_streaming)
        )
        resolved_disable_checker_llm = (
            True if cli_disable_checker_llm else _as_bool(model_override.get("disable_checker_llm"), default_disable_checker_llm)
        )
        resolved_ollama_think_mode = str(
            cli_ollama_think_mode
            if cli_ollama_think_mode is not None
            else model_override.get("ollama_think_mode", default_ollama_think_mode)
        ).strip().lower()
        if resolved_ollama_think_mode not in {"auto", "think", "no_think"}:
            resolved_ollama_think_mode = "auto"
        resolved_ollama_use_native_chat = (
            cli_ollama_use_native_chat
            if cli_ollama_use_native_chat is not None
            else _as_bool(model_override.get("ollama_use_native_chat"), default_ollama_use_native_chat)
        )
        resolved_ollama_native_chat_timeout_sec = float(
            model_override.get("ollama_native_chat_timeout_sec", default_ollama_native_chat_timeout_sec)
        )
        resolved_ollama_native_chat_timeout_sec = max(1.0, resolved_ollama_native_chat_timeout_sec)

        # Apply resolved model policy for this model before env/provider setup.
        config["OLLAMA_THINK_MODE"] = resolved_ollama_think_mode
        config["OLLAMA_USE_NATIVE_CHAT"] = bool(resolved_ollama_use_native_chat)
        config["OLLAMA_NATIVE_CHAT_TIMEOUT_SEC"] = float(resolved_ollama_native_chat_timeout_sec)
        os.environ["DILU_DECISION_TIMEOUT_SEC"] = str(resolved_decision_timeout_sec)
        os.environ["DILU_MAX_OUTPUT_TOKENS"] = str(resolved_decision_max_output_tokens)
        os.environ["DILU_USE_STREAMING"] = "0" if resolved_disable_streaming else "1"
        os.environ["DILU_ENABLE_CHECKER_LLM"] = "0" if resolved_disable_checker_llm else "1"
        os.environ["OLLAMA_THINK_MODE"] = resolved_ollama_think_mode
        os.environ["OLLAMA_USE_NATIVE_CHAT"] = "1" if resolved_ollama_use_native_chat else "0"
        os.environ["OLLAMA_NATIVE_CHAT_TIMEOUT_SEC"] = str(resolved_ollama_native_chat_timeout_sec)
        report["model_runtime_policies"][model_name] = {
            "matched_override": model_override,
            "decision_timeout_sec": round(resolved_decision_timeout_sec, 3),
            "decision_max_output_tokens": int(resolved_decision_max_output_tokens),
            "disable_streaming": bool(resolved_disable_streaming),
            "disable_checker_llm": bool(resolved_disable_checker_llm),
            "ollama_think_mode": resolved_ollama_think_mode,
            "ollama_use_native_chat": bool(resolved_ollama_use_native_chat),
            "ollama_native_chat_timeout_sec": round(resolved_ollama_native_chat_timeout_sec, 3),
        }
        model_metrics_configs[model_name] = {
            **dict(report["metrics_config"]),
            "resolved_model_policy": dict(report["model_runtime_policies"][model_name]),
        }
        print(f"\n[bold cyan]Evaluating model[/bold cyan]: {model_name}")
        print(
            "[dim]  Policy: timeout={timeout}s, max_tokens={tokens}, streaming={streaming}, "
            "checker={checker}, think_mode={think}, native_chat={native}, native_timeout={native_timeout}s[/dim]".format(
                timeout=round(resolved_decision_timeout_sec, 3),
                tokens=int(resolved_decision_max_output_tokens),
                streaming=not resolved_disable_streaming,
                checker=not resolved_disable_checker_llm,
                think=resolved_ollama_think_mode,
                native=bool(resolved_ollama_use_native_chat),
                native_timeout=round(resolved_ollama_native_chat_timeout_sec, 3),
            )
        )
        configure_runtime_env(config, chat_model_override=model_name)
        agent_memory = DrivingMemory(db_path=config["memory_path"])
        model_run_dir = None
        model_log_path = None
        if save_run_artifacts:
            model_run_dir = build_model_run_dir(experiment_root, model_name, eval_run_id)
            model_log_path = os.path.join(model_run_dir, "log.txt")
            with open(model_log_path, "a", encoding="utf-8") as f:
                f.write(
                    "=== Eval Run {run_id} | Model {model} | Created {created} ===\n".format(
                        run_id=eval_run_id,
                        model=model_name,
                        created=datetime.now().isoformat(timespec="seconds"),
                    )
                )

        episodes = []
        model_alignment_samples = []
        for idx, seed in enumerate(seeds, start=1):
            print(f"[dim]  Seed {idx}/{len(seeds)}: {seed}[/dim]")
            episode_result = run_episode(
                config=config,
                env_config=env_config,
                agent_memory=agent_memory,
                seed=seed,
                few_shot_num=few_shot_num,
                temp_dir=temp_dir,
                ttc_threshold_sec=ttc_threshold_sec,
                headway_threshold_m=headway_threshold_m,
                alignment_sample_rate=alignment_sample_rate,
                alignment_max_samples=alignment_max_samples,
                slow_decision_threshold_sec=slow_decision_threshold_sec,
                save_artifacts=save_run_artifacts,
                run_dir=model_run_dir,
                run_id=eval_run_id if save_run_artifacts else None,
                model_name=model_name,
            )
            episode_alignment_samples = episode_result.pop("alignment_samples", [])
            for sample in episode_alignment_samples:
                sample["model"] = model_name
                model_alignment_samples.append(sample)
            episodes.append(episode_result)
            if save_run_artifacts and model_log_path:
                _append_eval_run_log(model_log_path, model_name, episode_result)
            status = "CRASH" if episode_result["crashed"] else ("ERROR" if episode_result["error"] else ("TIMEOUT" if episode_result.get("timeout_triggered") else "OK"))
            print(
                f"    -> {status} | steps={episode_result['steps']}/{episode_result['max_steps']} "
                f"| t={episode_result['episode_runtime_sec']}s | timeout_steps={episode_result.get('decision_timeout_count', 0)}"
            )
            if episode_result["error"]:
                print(f"    -> [red]{episode_result['error']}[/red]")

        report["per_model"][model_name] = episodes
        agg = aggregate_results(model_name, episodes)
        report["aggregates"].append(agg)
        aggregate_by_model[model_name] = agg
        report["alignment_samples"].extend(model_alignment_samples[:alignment_max_samples] if alignment_max_samples > 0 else [])
        if save_run_artifacts and model_run_dir:
            run_metrics_report = _build_eval_run_metrics_report(
                model_name=model_name,
                experiment_id=experiment_id,
                experiment_root=experiment_root,
                run_id=eval_run_id,
                run_dir=model_run_dir,
                config_path=args.config,
                openai_api_type=str(config.get("OPENAI_API_TYPE", "")),
                few_shot_num=int(few_shot_num),
                memory_path=str(config.get("memory_path", "")),
                simulation_duration=int(config["simulation_duration"]),
                metrics_config=model_metrics_configs.get(model_name, report["metrics_config"]),
                episodes=episodes,
                aggregate=agg,
            )
            run_metrics_path = timestamped_results_path("run_metrics", ext=".json", results_dir=model_run_dir)
            write_json_atomic(run_metrics_path, run_metrics_report)
            model_run_outputs[model_name] = {
                "run_id": eval_run_id,
                "run_dir": model_run_dir,
                "log_path": model_log_path,
                "run_metrics": run_metrics_path,
            }

    report["model_run_outputs"] = model_run_outputs

    user_out_path = None
    if args.output:
        user_out_path = ensure_parent_dir(args.output)
        write_json_atomic(user_out_path, report)

    if structured_output:
        out_path = timestamped_results_path("eval_compare", ext=".json", results_dir=compare_dir)
        write_json_atomic(out_path, report)
    else:
        if user_out_path:
            out_path = user_out_path
        else:
            out_path = timestamped_results_path("eval_compare", ext=".json", results_dir=compare_dir)
            write_json_atomic(out_path, report)

    model_summary_paths: Dict[str, Dict[str, str]] = {}
    compare_base = os.path.basename(out_path)
    compare_name, _ = os.path.splitext(compare_base)
    compare_ts = compare_name.replace("eval_compare_", "")

    if structured_output and experiment_root:
        for model_name in args.models:
            model_root = build_model_root(experiment_root, model_name)
            eval_dir = ensure_dir(os.path.join(model_root, "eval"))
            summary_path = os.path.join(eval_dir, f"eval_summary_{compare_ts}.json")
            episodes_path = os.path.join(eval_dir, f"eval_episodes_{compare_ts}.json")

            model_extract = _build_model_extract(
                model_name=model_name,
                experiment_id=experiment_id,
                source_compare_report=out_path,
                aggregate=aggregate_by_model[model_name],
                episodes=report["per_model"][model_name],
                metrics_config=model_metrics_configs.get(model_name, report["metrics_config"]),
            )
            write_json_atomic(summary_path, model_extract)
            write_json_atomic(episodes_path, {
                "model": model_name,
                "experiment_id": experiment_id,
                "source_compare_report": out_path,
                "episodes": report["per_model"][model_name],
            })
            model_summary_paths[model_name] = {
                "summary": summary_path,
                "episodes": episodes_path,
            }

        report["model_eval_outputs"] = model_summary_paths
        write_json_atomic(out_path, report)

        _update_experiment_manifest_for_eval(
            experiment_root=experiment_root,
            experiment_id=experiment_id,
            config_path=args.config,
            memory_path=config["memory_path"],
            few_shot_num=int(few_shot_num),
            simulation_duration=int(config["simulation_duration"]),
            compare_report_path=out_path,
            model_summaries=model_summary_paths,
            model_run_outputs=model_run_outputs if save_run_artifacts else None,
        )

    print("\n[bold green]Aggregate Summary[/bold green]")
    for row in report["aggregates"]:
        print(
            f"- {row['model']}: crashes={row['crashes']}/{row['episodes']} "
            f"(rate={row['crash_rate']}), no_collision_rate={row['no_collision_rate']}, "
            f"avg_steps={row['avg_steps']}, strict_format_rate={row['response_strict_format_rate']}, "
                f"ttc_danger_rate={row['ttc_danger_rate_mean']}, headway_violation_rate={row['headway_violation_rate_mean']}, "
                f"decision_timeout_rate={row.get('decision_timeout_rate_mean')}, "
                f"native_timeout_rate={row.get('ollama_native_timeout_rate_mean')}, "
                f"fallback_action_rate={row.get('fallback_action_rate_mean')}, "
                f"avg_episode_runtime_sec={row['avg_episode_runtime_sec']}"
            )
    print(f"\nSaved report: [bold]{out_path}[/bold]")
    if user_out_path and user_out_path != out_path:
        print(f"Saved user-requested output copy: [bold]{user_out_path}[/bold]")


if __name__ == "__main__":
    main()
