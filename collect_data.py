import gymnasium as gym
import numpy as np
import json
import os
import random
import yaml
import shutil
from rich import print
from tqdm import tqdm

# Import DiLu modules
from dilu.scenario.envScenario import EnvScenario
from dilu.runtime import (
    build_highway_env_config,
    DEFAULT_DILU_SEEDS,
    ensure_dir,
    ensure_parent_dir,
)

# --- CONFIGURATION ---
TOTAL_EPISODES = 50  # Number of successful episodes to collect
OUTPUT_FILE = "data/gold_standard_data.jsonl"
def setup_env(config):
    """
    Sets up the environment configuration matching run_dilu.py
    """
    return build_highway_env_config(
        config,
        show_trajectories=True,
        render_agent=False,  # No need to render agent sprite for data collection
    )


def get_expert_decision(sce):
    """
    A Rule-Based Expert that calculates the optimal action based on safety and efficiency.
    Returns: (action_id, thought_string)

    Actions:
    0: LANE_LEFT
    1: IDLE
    2: LANE_RIGHT
    3: FASTER
    4: SLOWER
    """
    # Use .unwrapped to avoid Gymnasium wrapper warnings
    vehicle = sce.env.unwrapped.vehicle
    road = sce.env.unwrapped.road

    # 1. Get neighbors (Tuple: front, rear)
    # highway-env 1.8.2 returns (front_vehicle, rear_vehicle)
    front_vehicle, rear_vehicle = road.neighbour_vehicles(vehicle, vehicle.lane_index)

    # 2. Calculate Distance Manually
    front_dist = 1000  # Default large distance
    if front_vehicle:
        front_dist = np.linalg.norm(vehicle.position - front_vehicle.position)

    # Default: IDLE (Maintain Course)
    action = 1
    reasoning = "Maintaining current speed and lane as conditions are nominal."

    # --- SAFETY CHECK (Braking) ---
    if front_vehicle:
        relative_speed = vehicle.speed - front_vehicle.speed
        # If we are faster and close (< 25m), or extremely close (< 10m)
        if (front_dist < 25 and relative_speed > 0) or (front_dist < 10):
            action = 4  # DECELERATE
            reasoning = f"Safety Critical: Front vehicle is {front_dist:.1f}m ahead. Decelerating to avoid collision."
            return action, reasoning

    def lane_change_safe(target_lane_id):
        """
        Basic safety check for lane changes using nearest front/rear vehicles in target lane.
        Returns (is_safe: bool, reason: str)
        """
        current_lane = vehicle.lane_index
        target_lane_index = (current_lane[0], current_lane[1], target_lane_id)
        try:
            target_front, target_rear = road.neighbour_vehicles(vehicle, target_lane_index)
        except Exception:
            return False, "Could not query target-lane neighbors."

        if target_front is not None:
            front_gap = np.linalg.norm(vehicle.position - target_front.position)
            front_rel_speed = vehicle.speed - target_front.speed
            if front_gap < 15:
                return False, f"Target lane front gap too small ({front_gap:.1f}m)."
            if front_gap < 25 and front_rel_speed > 0:
                return False, f"Target lane front vehicle too close/slower ({front_gap:.1f}m)."

        if target_rear is not None:
            rear_gap = np.linalg.norm(vehicle.position - target_rear.position)
            rear_closing_speed = target_rear.speed - vehicle.speed
            if rear_gap < 10:
                return False, f"Target lane rear gap too small ({rear_gap:.1f}m)."
            if rear_gap < 20 and rear_closing_speed > 2.0:
                return False, f"Rear vehicle approaching too quickly ({rear_gap:.1f}m, +{rear_closing_speed:.1f}m/s)."

        return True, "Target lane appears safe for lane change."

    # --- EFFICIENCY CHECK (Overtaking & Speed) ---
    TARGET_SPEED = 28.0  # ~100 km/h

    # If stuck behind slow car, try to change lane
    if front_vehicle and front_dist < 40 and vehicle.speed < TARGET_SPEED:
        # Check current lane index (0 is usually left, 3 is right in 4-lane setup)
        # LaneIndex is a tuple (from, to, id). We want id.
        current_lane_id = vehicle.lane_index[2]

        # Try Left (if not already leftmost)
        if current_lane_id > 0:
            safe_left, left_reason = lane_change_safe(current_lane_id - 1)
            if safe_left:
                action = 0  # LANE_LEFT
                reasoning = "Impeded by slower vehicle. Left lane is safe, changing lane to maintain efficiency."
                return action, reasoning

        # Try Right (if not rightmost)
        if current_lane_id < 3:
            safe_right, right_reason = lane_change_safe(current_lane_id + 1)
            if safe_right:
                action = 2  # LANE_RIGHT
                reasoning = "Impeded by slower vehicle. Right lane is safe, changing lane to maintain efficiency."
                return action, reasoning

        # If no safe lane change exists, slow down rather than forcing a risky maneuver.
        if front_vehicle and front_dist < 25:
            action = 4
            reasoning = "Blocked by slower traffic and no safe lane change gap available. Decelerating."
            return action, reasoning

    # If free road and slow, accelerate
    if (not front_vehicle or front_dist > 50) and vehicle.speed < TARGET_SPEED:
        action = 3  # ACCELERATE
        reasoning = f"Road clear ahead and speed ({vehicle.speed:.1f}m/s) is below target. Accelerating."
        return action, reasoning

    return action, reasoning


def main():
    # 1. Load Config
    print("[cyan]Loading configuration from config.yaml...[/cyan]")
    config_data = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    env_config = setup_env(config_data)

    # 2. Setup Environment
    env_name = 'highway-v0'
    # Use rgb_array to avoid opening a window, but allow internal rendering if needed
    env = gym.make(env_name, render_mode='rgb_array')
    env.configure(env_config[env_name])

    successful_samples = []

    print(f"[cyan]Starting collection of {TOTAL_EPISODES} successful episodes...[/cyan]")
    print(f"[dim]Output file: {OUTPUT_FILE}[/dim]")

    pbar = tqdm(total=TOTAL_EPISODES)
    episode_count = 0

    # Ensure output directory and temp directory exist
    ensure_parent_dir(OUTPUT_FILE)

    # Ensure a temp directory for DBs exists
    temp_dir = ensure_dir('temp_dbs')

    while episode_count < TOTAL_EPISODES:
        # Reset Env
        seed = random.choice(DEFAULT_DILU_SEEDS)
        obs, info = env.reset(seed=seed)

        # Initialize DiLu Scenario wrapper
        # We use a unique temp db for each episode to avoid locking/conflicts
        temp_db_path = os.path.join(temp_dir, f"temp_collect_{episode_count}_{random.randint(1000, 9999)}.db")

        # EnvScenario expects the database path string
        sce = EnvScenario(env, env_name, seed, temp_db_path)

        episode_buffer = []
        done = False
        truncated = False
        step = 0

        # Simulation Loop
        while not (done or truncated) and step < config_data["simulation_duration"]:
            try:
                # A. Get Description (INPUT for LLM)
                # This matches exactly what the Agent sees in run_dilu.py
                description = sce.describe(step)
                available_actions = sce.availableActionsDescription()

                # B. Get Expert Decision (TARGET OUTPUT)
                action_id, reasoning = get_expert_decision(sce)

                # C. Step Environment
                obs, reward, done, truncated, info = env.step(action_id)

                # D. Format Data (Prompt + Completion)
                # Construct the full prompt context
                full_input = f"Driving scenario description:\n{description}\nAvailable actions:\n{available_actions}"

                # Create JSONL entry
                json_row = {
                    "instruction": "You are an expert autonomous driving agent. Analyze the scenario and output a JSON decision.",
                    "input": full_input,
                    "output": json.dumps({
                        "analysis": "Scenario analyzed via internal sensor data.",
                        "reasoning": reasoning,
                        "action_id": action_id
                    })
                }
                episode_buffer.append(json_row)
                step += 1

                # Check for Crash
                if info.get('crashed', False):
                    episode_buffer = []  # Discard failed episode
                    break

            except Exception as e:
                print(f"[red]Error in step {step}: {e}[/red]")
                episode_buffer = []
                break

        # End of Episode Handling
        if episode_buffer and not info.get('crashed', False):
            successful_samples.extend(episode_buffer)
            pbar.update(1)
            episode_count += 1

            # Save incrementally
            if episode_count % 5 == 0:
                with open(OUTPUT_FILE, 'w') as f:
                    for entry in successful_samples:
                        f.write(json.dumps(entry) + "\n")

        # Clean up temp database
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except PermissionError:
                pass  # Sometimes SQLite locks file briefly

    env.close()

    # Final Save
    with open(OUTPUT_FILE, 'w') as f:
        for entry in successful_samples:
            f.write(json.dumps(entry) + "\n")

    # Clean up temp folder
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

    print(f"[green]SUCCESS: Collected {len(successful_samples)} samples in {OUTPUT_FILE}[/green]")


if __name__ == "__main__":
    main()
