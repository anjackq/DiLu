import argparse
import copy
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import yaml
from rich import print

from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.scenario.envScenario import EnvScenario


DEFAULT_SEEDS = [
    5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348,
    4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31
]
STRICT_RESPONSE_PATTERN = re.compile(r"Response to user:\s*\#{4}\s*([0-4])\s*$", re.IGNORECASE)


def setup_runtime_env(config: Dict, chat_model: str) -> None:
    api_type = config["OPENAI_API_TYPE"]
    if api_type == "azure":
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = str(config["AZURE_API_VERSION"])
        os.environ["OPENAI_API_BASE"] = str(config["AZURE_API_BASE"])
        os.environ["OPENAI_API_KEY"] = str(config["AZURE_API_KEY"])
        os.environ["AZURE_CHAT_DEPLOY_NAME"] = str(config["AZURE_CHAT_DEPLOY_NAME"])
        os.environ["AZURE_EMBED_DEPLOY_NAME"] = str(config["AZURE_EMBED_DEPLOY_NAME"])
    elif api_type == "openai":
        os.environ["OPENAI_API_TYPE"] = "openai"
        os.environ["OPENAI_API_KEY"] = str(config["OPENAI_KEY"])
        os.environ["OPENAI_CHAT_MODEL"] = chat_model
        if config.get("OPENAI_REFLECTION_MODEL"):
            os.environ["OPENAI_REFLECTION_MODEL"] = str(config["OPENAI_REFLECTION_MODEL"])
    elif api_type == "ollama":
        os.environ["OPENAI_API_TYPE"] = "ollama"
        os.environ["OLLAMA_API_BASE"] = str(config["OLLAMA_API_BASE"])
        os.environ["OPENAI_BASE_URL"] = str(config["OLLAMA_API_BASE"])
        os.environ["OLLAMA_CHAT_MODEL"] = chat_model
        os.environ["OLLAMA_API_KEY"] = str(config["OLLAMA_API_KEY"])
        os.environ["OLLAMA_EMBED_MODEL"] = str(config["OLLAMA_EMBED_MODEL"])
        if config.get("OLLAMA_REFLECTION_MODEL"):
            os.environ["OLLAMA_REFLECTION_MODEL"] = str(config["OLLAMA_REFLECTION_MODEL"])
    else:
        raise ValueError(f"Unsupported OPENAI_API_TYPE: {api_type}")


def build_env_config(config: Dict) -> Dict:
    return {
        "highway-v0": {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": config["vehicle_count"],
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(5, 32, 9),
            },
            "lanes_count": 4,
            "other_vehicles_type": config["other_vehicle_type"],
            "duration": config["simulation_duration"],
            "vehicles_density": config["vehicles_density"],
            "show_trajectories": False,
            "render_agent": False,
            "scaling": 5,
            "initial_lane_id": None,
            "ego_spacing": 4,
        }
    }


def parse_seeds(raw: Optional[str]) -> List[int]:
    if not raw:
        return DEFAULT_SEEDS
    seeds = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


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


def run_episode(
    config: Dict,
    env_config: Dict,
    agent_memory: DrivingMemory,
    seed: int,
    few_shot_num: int,
    temp_dir: str,
) -> Dict:
    env_type = "highway-v0"
    env = None
    temp_db_path = os.path.join(temp_dir, f"eval_{seed}_{int(time.time() * 1000)}.db")
    started = time.time()
    error = None
    crashed = False
    truncated = False
    terminated = False
    steps = 0
    final_info = {}
    decisions_made = 0
    responses_with_delimiter = 0
    responses_strict_format = 0
    responses_direct_parseable = 0

    try:
        env = gym.make(env_type, render_mode="rgb_array")
        env.configure(env_config[env_type])
        obs, info = env.reset(seed=seed)
        final_info = info

        sce = EnvScenario(env, env_type, seed, temp_db_path)
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
            decisions_made += 1
            fmt = _response_format_metrics(response)
            responses_with_delimiter += int(fmt["has_delimiter"])
            responses_strict_format += int(fmt["strict_format_match"])
            responses_direct_parseable += int(fmt["direct_action_parseable"])

            action = _safe_int_action(action)
            obs, reward, terminated, truncated, info = env.step(action)
            final_info = info
            crashed = bool(info.get("crashed", False))
            done = terminated or truncated
            steps += 1

            # Keep DB prompt logs for replay/debugging if needed.
            try:
                sce.promptsCommit(frame_id, None, done, human_question, fewshot_answer, response)
            except Exception:
                pass

            if done:
                break

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except Exception:
                pass

    duration_sec = time.time() - started
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
        "decisions_made": decisions_made,
        "responses_with_delimiter": responses_with_delimiter,
        "responses_strict_format": responses_strict_format,
        "responses_direct_parseable": responses_direct_parseable,
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
    total_delimiters = sum(e.get("responses_with_delimiter", 0) for e in episodes)
    total_strict = sum(e.get("responses_strict_format", 0) for e in episodes)
    total_direct = sum(e.get("responses_direct_parseable", 0) for e in episodes)

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
        "response_delimiter_rate": round(total_delimiters / total_decisions, 4) if total_decisions else None,
        "response_strict_format_rate": round(total_strict / total_decisions, 4) if total_decisions else None,
        "response_direct_parseable_rate": round(total_direct / total_decisions, 4) if total_decisions else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DiLu agent behavior across Ollama models on fixed seeds.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to compare (e.g. deepseek-r1:14b dilu-llama3_1-8b-v1)")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to DiLu fixed seed list.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of seeds after parsing.")
    parser.add_argument("--few-shot-num", type=int, default=None, help="Override config few_shot_num.")
    parser.add_argument("--memory-path", default=None, help="Override config memory_path.")
    parser.add_argument("--output", default=None, help="Write JSON report to this file (default: results/eval_compare_<timestamp>.json)")
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

    env_config = build_env_config(config)
    temp_dir = os.path.join("temp", "eval_compare")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": args.config,
        "openai_api_type": config["OPENAI_API_TYPE"],
        "models": args.models,
        "seeds": seeds,
        "few_shot_num": few_shot_num,
        "memory_path": config["memory_path"],
        "simulation_duration": int(config["simulation_duration"]),
        "per_model": {},
        "aggregates": [],
    }

    for model_name in args.models:
        print(f"\n[bold cyan]Evaluating model[/bold cyan]: {model_name}")
        setup_runtime_env(config, model_name)
        agent_memory = DrivingMemory(db_path=config["memory_path"])

        episodes = []
        for idx, seed in enumerate(seeds, start=1):
            print(f"[dim]  Seed {idx}/{len(seeds)}: {seed}[/dim]")
            episode_result = run_episode(
                config=config,
                env_config=env_config,
                agent_memory=agent_memory,
                seed=seed,
                few_shot_num=few_shot_num,
                temp_dir=temp_dir,
            )
            episodes.append(episode_result)
            status = "CRASH" if episode_result["crashed"] else ("ERROR" if episode_result["error"] else "OK")
            print(
                f"    -> {status} | steps={episode_result['steps']}/{episode_result['max_steps']} "
                f"| t={episode_result['episode_runtime_sec']}s"
            )
            if episode_result["error"]:
                print(f"    -> [red]{episode_result['error']}[/red]")

        report["per_model"][model_name] = episodes
        report["aggregates"].append(aggregate_results(model_name, episodes))

    if args.output:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("results", f"eval_compare_{ts}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[bold green]Aggregate Summary[/bold green]")
    for row in report["aggregates"]:
        print(
            f"- {row['model']}: crashes={row['crashes']}/{row['episodes']} "
            f"(rate={row['crash_rate']}), no_collision_rate={row['no_collision_rate']}, "
            f"avg_steps={row['avg_steps']}, strict_format_rate={row['response_strict_format_rate']}, "
            f"avg_episode_runtime_sec={row['avg_episode_runtime_sec']}"
        )
    print(f"\nSaved report: [bold]{out_path}[/bold]")


if __name__ == "__main__":
    main()
