import gymnasium as gym
import numpy as np
import json
import os
import random
import yaml
import copy
from rich import print
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo

# Import DiLu modules (Must be in the same directory as this script)
from dilu.scenario.envScenario import EnvScenario

# CONFIGURATION
TOTAL_EPISODES = 50  # Number of successful episodes to collect
OUTPUT_FILE = "../data/gold_standard_data.jsonl"
SEEDS = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348, 4213, 2572]

# Mapping for reasoning generation
ACTION_MAP = {
    0: "Turn-left",
    1: "IDLE",
    2: "Turn-right",
    3: "Acceleration",
    4: "Deceleration"
}


def setup_env(config):
    """
    Sets up the environment configuration matching run_dilu.py
    """
    env_config = {
        'highway-v0':
            {
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
                "show_trajectories": True,
                "render_agent": False,
                "scaling": 5,
                'initial_lane_id': None,
                "ego_spacing": 4,
            }
    }
    return env_config


def get_expert_decision(sce):
    """
    A Rule-Based Expert that calculates the optimal action based on safety and efficiency.
    Returns: (action_id, thought_string)
    """
    vehicle = sce.env.unwrapped.vehicle
    road = sce.env.unwrapped.road


    # --- FIX START ---
    # 1. Get neighbors safely
    # road.neighbour_vehicles returns (front_vehicle, rear_vehicle)
    front_vehicle, rear_vehicle = road.neighbour_vehicles(vehicle, vehicle.lane_index)

    # 2. Calculate Distance Manually
    front_dist = 1000  # Default large distance
    if front_vehicle:
        # Calculate Euclidean distance
        front_dist = np.linalg.norm(vehicle.position - front_vehicle.position)
    # --- FIX END ---

    # Default: IDLE (Maintain Course)
    action = 1
    reasoning = "Maintaining current speed and lane as conditions are nominal."

    # --- SAFETY CHECK (Braking) ---
    if front_vehicle:
        relative_speed = vehicle.speed - front_vehicle.speed
        # If we are faster and close
        if front_dist < 25 and relative_speed > 0:
            action = 4  # DECELERATE
            reasoning = f"Safety Critical: Front vehicle is {front_dist:.1f}m ahead and slower. Decelerating to avoid collision."
            return action, reasoning
        elif front_dist < 10:  # Too close regardless of speed
            action = 4
            reasoning = f"Safety Critical: Front vehicle is dangerously close ({front_dist:.1f}m). Braking immediately."
            return action, reasoning

    # --- EFFICIENCY CHECK (Overtaking & Speed) ---
    TARGET_SPEED = 28.0

    # If stuck behind slow car, try to change lane
    if front_vehicle and front_dist < 40 and vehicle.speed < TARGET_SPEED:
        # Check Left Lane (Lane index 0 is typically leftmost in highway-env)
        # Note: Highway-env lane indices are (from_node, to_node, lane_id)
        current_lane_id = vehicle.lane_index[2]

        # Try Left (if not already leftmost)
        if current_lane_id > 0:
            action = 0  # LANE_LEFT
            reasoning = "Impeded by slower vehicle. Changing lane to left to maintain efficiency."
            return action, reasoning

        # Try Right (if not rightmost)
        elif current_lane_id < 3:
            action = 2  # LANE_RIGHT
            reasoning = "Impeded by slower vehicle. Changing lane to right to maintain efficiency."
            return action, reasoning

    # If free road and slow, accelerate
    if (not front_vehicle or front_dist > 50) and vehicle.speed < TARGET_SPEED:
        action = 3  # ACCELERATE
        reasoning = f"Road clear ahead and speed ({vehicle.speed:.1f}m/s) is below target. Accelerating."
        return action, reasoning

    return action, reasoning


def main():
    # 1. Load Config
    print("[cyan]Loading configuration...[/cyan]")
    config_data = yaml.load(open('../config.yaml'), Loader=yaml.FullLoader)
    env_config = setup_env(config_data)

    # 2. Setup Environment
    env_name = 'highway-v0'
    env = gym.make(env_name, render_mode='rgb_array')
    env.unwrapped.configure(env_config[env_name])

    successful_samples = []

    print(f"[cyan]Starting collection of {TOTAL_EPISODES} successful episodes...[/cyan]")

    pbar = tqdm(total=TOTAL_EPISODES)
    episode_count = 0

    while len(successful_samples) < TOTAL_EPISODES * config_data["simulation_duration"]:
        # Reset Env
        seed = random.choice(SEEDS)
        obs, info = env.reset(seed=seed)

        # Initialize DiLu Scenario wrapper
        temp_db = f"temp_collect_{random.randint(1000, 9999)}.db"
        sce = EnvScenario(env, env_name, seed, temp_db)

        episode_buffer = []
        done = False
        truncated = False

        step = 0
        while not (done or truncated) and step < config_data["simulation_duration"]:
            # A. Get Description (INPUT)
            try:
                description = sce.describe(step)
                available_actions = sce.availableActionsDescription()
            except Exception as e:
                # Handle occasional description errors
                print(f"[red]Warning: Description failed at step {step}: {e}[/red]")
                break

            full_input = f"""Driving scenario description:\n{description}\nAvailable actions:\n{available_actions}"""

            # B. Get Expert Decision (TARGET)
            action_id, reasoning = get_expert_decision(sce)

            # C. Step Environment
            obs, reward, done, truncated, info = env.step(action_id)

            # D. Check Crash
            if info.get('crashed', False):
                episode_buffer = []
                break

            # E. Format Data
            json_row = {
                "instruction": "You are an expert autonomous driving agent. Analyze the scenario and checkpoints a JSON decision.",
                "input": full_input,
                "checkpoints": json.dumps({
                    "analysis": "Scenario analyzed via internal sensor data.",
                    "reasoning": reasoning,
                    "action_id": action_id
                })
            }
            episode_buffer.append(json_row)
            step += 1

        # End of Episode
        if episode_buffer and not info.get('crashed', False):
            successful_samples.extend(episode_buffer)
            pbar.update(len(episode_buffer))  # Update progress by steps collected
            episode_count += 1

            if episode_count % 10 == 0:
                with open(OUTPUT_FILE, 'w') as f:
                    for entry in successful_samples:
                        f.write(json.dumps(entry) + "\n")

        # Cleanup
        if os.path.exists(temp_db):
            try:
                os.remove(temp_db)
            except:
                pass

        if len(successful_samples) >= TOTAL_EPISODES * config_data["simulation_duration"]:
            break

    env.close()

    # Final Save
    with open(OUTPUT_FILE, 'w') as f:
        for entry in successful_samples:
            f.write(json.dumps(entry) + "\n")

    print(f"[green]SUCCESS: Collected {len(successful_samples)} samples in {OUTPUT_FILE}[/green]")


if __name__ == "__main__":
    main()