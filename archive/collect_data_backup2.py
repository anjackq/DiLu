import gymnasium as gym
import json
import os
import random
import yaml
import numpy as np
import tempfile  # <--- NEW IMPORT
from rich import print
from tqdm import tqdm

# Import DiLu modules
from dilu.scenario.envScenario import EnvScenario

# --- CONFIGURATION ---
TOTAL_EPISODES = 100
OUTPUT_FILE = "../data/gold_standard_data.jsonl"
SEEDS = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348, 4213, 2572]

SYSTEM_PROMPT = """You are an autonomous driving decision engine.
Analyze the scenario and checkpoints the single best Action_id integer (0-4).

Strict Output Format:
Reasoning: <one sentence>
Action_id: <integer>"""


def get_expert_action(sce):
    # (Same expert logic as before)
    ego = sce.env.unwrapped.vehicle
    road = sce.env.unwrapped.road

    front_vehicle, rear_vehicle = road.neighbour_vehicles(ego, ego.lane_index)

    front_dist = 1000
    front_speed = 30
    if front_vehicle:
        front_dist = np.linalg.norm(ego.position - front_vehicle.position)
        front_speed = front_vehicle.speed

    current_speed = ego.speed
    target_speed = 28.0

    # Fix for lane index
    current_lane_id = ego.lane_index[2]

    # Logic Tree
    if front_dist < 15:
        return 4, f"CRITICAL: Vehicle ahead is too close ({front_dist:.1f}m). Emergency braking."

    if front_dist < 30 and front_speed < current_speed:
        if current_lane_id > 0:
            return 0, f"Vehicle ahead is slow ({front_speed:.1f}m/s). Changing lane to left."
        elif current_lane_id < 3:
            return 2, f"Vehicle ahead is slow. Changing lane to right."
        else:
            return 4, f"Vehicle ahead is slow and lanes are blocked. Decelerating."

    if current_speed < target_speed - 2:
        return 3, f"Road is clear ahead. Accelerating."

    if current_speed > target_speed + 2:
        return 4, f"Speed ({current_speed:.1f}m/s) exceeds target. Decelerating."

    return 1, f"Traffic flow is stable. Maintaining current lane and speed."


def collect():
    config_env = {
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,
            "normalize": False,
            "vehicles_count": 10,
            "see_behind": True,
        },
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": np.linspace(0, 32, 9),
        },
        "duration": 40,
        "vehicles_density": 1.5,
        "show_trajectories": False,
        "render_agent": False,
    }

    env = gym.make('highway-v0', render_mode=None)
    env.configure(config_env)

    successful_samples = []
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    pbar = tqdm(total=TOTAL_EPISODES)

    # --- TEMP DIRECTORY CONTEXT MANAGER ---
    # This creates a folder that AUTOMATICALLY deletes itself when this block ends
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[green]Using temporary directory for .db files: {temp_dir}[/green]")

        while len(successful_samples) < TOTAL_EPISODES:
            seed = random.choice(SEEDS)
            obs, info = env.reset(seed=seed)

            # Create unique DB path INSIDE the temp directory
            # We use a random name to avoid conflicts
            db_name = f"ep_{len(successful_samples)}_{random.randint(1000, 9999)}.db"
            temp_db_path = os.path.join(temp_dir, db_name)

            try:
                # Initialize EnvScenario with the temp path
                sce = EnvScenario(env, 'highway-v0', seed, temp_db_path)

                episode_buffer = []
                done = False
                truncated = False
                step = 0

                while not (done or truncated):
                    try:
                        description = sce.describe(step)
                        available_actions = sce.availableActionsDescription()
                    except Exception:
                        break

                    action_id, reasoning = get_expert_action(sce)

                    full_input = f"""Driving scenario description:
{description}

Available actions:
{available_actions}

Decision:
"""
                    full_output = f"Reasoning: {reasoning}\nAction_id: {action_id}"

                    data_row = {
                        "instruction": SYSTEM_PROMPT,
                        "input": full_input,
                        "checkpoints": full_output
                    }
                    episode_buffer.append(data_row)

                    obs, reward, done, truncated, info = env.step(action_id)
                    step += 1

                    if info.get('crashed', False):
                        episode_buffer = []
                        break

                if episode_buffer:
                    successful_samples.extend(episode_buffer)
                    pbar.update(len(episode_buffer))

                    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        for entry in successful_samples:
                            f.write(json.dumps(entry) + "\n")

            except Exception as e:
                # If an episode crashes, just print and move to next
                # The temp dir will still handle cleanup later
                # print(f"[red]Episode error: {e}[/red]")
                pass

            # Optional: We can close the sce to release the file lock,
            # allowing the OS to clean it up more easily if we wanted to delete it manually.
            # But with TemporaryDirectory, we can just leave it until the end.
            if 'sce' in locals():
                del sce

    # When we exit the 'with' block, temp_dir is deleted!
    env.close()
    pbar.close()
    print(f"[green]Successfully collected {len(successful_samples)} samples to {OUTPUT_FILE}[/green]")


if __name__ == "__main__":
    collect()