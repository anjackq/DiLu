import copy
import random
import numpy as np
import yaml
import os
from rich import print
from rich.table import Table
from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from dilu.scenario.envScenario import EnvScenario
from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.driver_agent.reflectionAgent import ReflectionAgent
from dilu.runtime import (
    configure_runtime_env,
    build_highway_env_config,
    DEFAULT_DILU_SEEDS,
    ensure_dir,
)

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class TelemetryHUDWrapper(gym.Wrapper):
    """Overlay live ego telemetry on rgb_array frames for video/render output."""

    def __init__(self, env):
        super().__init__(env)
        self.step_idx = 0
        self._warned_pil = False
        self._warned_pygame = False
        self._font = None
        self._pygame_font = None

    def __getstate__(self):
        # Prevent deepcopy failures when upstream code snapshots scenario/env objects.
        state = self.__dict__.copy()
        state["_font"] = None
        return state

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_idx = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_idx += 1
        return obs, reward, terminated, truncated, info

    def _ego_telemetry_from_env(self):
        uenv = self.env.unwrapped
        ego = getattr(uenv, "vehicle", None)
        road = getattr(uenv, "road", None)
        if ego is None or road is None:
            return None

        lane_id = ego.lane_index[2] if isinstance(ego.lane_index, tuple) and len(ego.lane_index) >= 3 else "N/A"

        lane_pos = None
        try:
            lane = road.network.get_lane(ego.lane_index)
            lane_pos = float(np.linalg.norm(ego.position - lane.start))
        except Exception:
            lane_pos = None

        front_vehicle = None
        try:
            front_vehicle, _ = road.neighbour_vehicles(ego, ego.lane_index)
        except Exception:
            pass

        front_gap = None
        front_rel_speed = None
        if front_vehicle is not None:
            front_gap = float(np.linalg.norm(ego.position - front_vehicle.position))
            front_rel_speed = float(ego.speed - front_vehicle.speed)

        min_gap = None
        try:
            perception_distance = getattr(uenv, "PERCEPTION_DISTANCE", 80)
            nearby = road.close_vehicles_to(
                ego, perception_distance, count=9, see_behind=True, sort="sorted"
            )
            if nearby:
                min_gap = min(float(np.linalg.norm(ego.position - v.position)) for v in nearby)
        except Exception:
            pass

        danger = "SAFE"
        if front_gap is not None:
            if front_gap < 10 or (front_gap < 25 and (front_rel_speed or 0) > 0):
                danger = "DANGER"
            elif front_gap < 35:
                danger = "CAUTION"
        if min_gap is not None and min_gap < 8:
            danger = "DANGER"

        return {
            "step": self.step_idx,
            "speed": float(getattr(ego, "speed", 0.0)),
            "accel": float(getattr(ego, "action", {}).get("acceleration", 0.0)),
            "lane": lane_id,
            "lane_pos": lane_pos,
            "front_gap": front_gap,
            "front_rel_speed": front_rel_speed,
            "min_gap": min_gap,
            "danger": danger,
        }

    def _draw_hud(self, frame: np.ndarray, t: dict) -> np.ndarray:
        if not PIL_AVAILABLE:
            if not self._warned_pil:
                print("[yellow]Pillow not installed. Telemetry HUD overlay disabled.[/yellow]")
                self._warned_pil = True
            return frame

        h, w = frame.shape[:2]
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, "RGBA")
        if self._font is None:
            try:
                self._font = ImageFont.load_default()
            except Exception:
                self._font = None

        speed_kmh = t["speed"] * 3.6
        # Compact in-frame bottom overlay panel (same frame size, no window resize).
        hud_h = 50
        panel_y0 = h - hud_h
        panel_x0, panel_x1 = 6, w - 6
        panel_y1 = h - 4
        draw.rounded_rectangle((panel_x0, panel_y0 + 2, panel_x1, panel_y1), radius=8, fill=(20, 20, 20, 190))

        danger_color = {"SAFE": (80, 220, 120), "CAUTION": (255, 210, 80), "DANGER": (255, 90, 90)}.get(
            t["danger"], (255, 255, 255)
        )

        cells = [
            ("Step", str(t["step"])),
            ("Lane", str(t["lane"])),
            ("Danger", str(t["danger"])),
            ("Speed", f"{speed_kmh:.1f} km/h"),
            ("Accel", f"{t['accel']:.1f} m/s²"),
            ("LanePos", "N/A" if t["lane_pos"] is None else f"{t['lane_pos']:.1f} m"),
            ("FrontGap", "N/A" if t["front_gap"] is None else f"{t['front_gap']:.1f} m"),
            ("RelV", "N/A" if t["front_rel_speed"] is None else f"{t['front_rel_speed']:.1f} m/s"),
            ("Nearest", "N/A" if t["min_gap"] is None else f"{t['min_gap']:.1f} m"),
        ]

        cols = 3
        rows = 3
        inner_pad = 6
        grid_x0 = panel_x0 + 6
        grid_y0 = panel_y0 + 6
        grid_x1 = panel_x1 - 6
        grid_y1 = panel_y1 - 4
        cell_w = (grid_x1 - grid_x0) // cols
        cell_h = (grid_y1 - grid_y0) // rows

        for idx, (label, value) in enumerate(cells):
            r = idx // cols
            c = idx % cols
            x0 = grid_x0 + c * cell_w
            y0 = grid_y0 + r * cell_h
            x1 = grid_x0 + (c + 1) * cell_w - 2
            y1 = grid_y0 + (r + 1) * cell_h - 2
            draw.rounded_rectangle((x0, y0, x1, y1), radius=4, fill=(35, 35, 35, 200), outline=(90, 90, 90, 220))

            text = f"{label}: {value}"
            text_color = danger_color if label == "Danger" else (255, 255, 255)
            text_y = y0 + max(1, (cell_h - 10) // 2)
            draw.text((x0 + inner_pad, text_y), text, fill=text_color, font=self._font)

        return np.array(img)

    def _draw_pygame_hud(self, frame_with_hud: np.ndarray) -> None:
        if not PYGAME_AVAILABLE:
            if not self._warned_pygame:
                print("[yellow]pygame not available for live HUD overlay.[/yellow]")
                self._warned_pygame = True
            return

        try:
            viewer = getattr(self.env.unwrapped, "viewer", None)
            screen = getattr(viewer, "screen", None) if viewer is not None else None
            if viewer is None:
                return
            h, w = frame_with_hud.shape[:2]
            if screen is None or screen.get_width() != w or screen.get_height() != h:
                # Keep the live window exactly the same size as the rendered frame.
                viewer.screen = pygame.display.set_mode((w, h))
                screen = viewer.screen

            # pygame.surfarray expects (width, height, channels)
            surface = pygame.surfarray.make_surface(np.transpose(frame_with_hud[:, :, :3], (1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
        except Exception:
            # Do not break the simulation if live HUD drawing fails on a platform/viewer variant.
            return

    def render(self):
        frame = self.env.render()
        if isinstance(frame, np.ndarray):
            telemetry = self._ego_telemetry_from_env()
            if telemetry is not None:
                # Render telemetry as a compact in-frame overlay and mirror it to the live pygame window.
                frame_with_hud = self._draw_hud(frame, telemetry)
                self._draw_pygame_hud(frame_with_hud)
                return frame_with_hud
        return frame


def get_ego_telemetry(sce: EnvScenario):
    """Collect live ego-vehicle telemetry and a simple danger indicator."""
    ego = sce.ego
    road = sce.road

    lane_id = ego.lane_index[2] if isinstance(ego.lane_index, tuple) and len(ego.lane_index) >= 3 else "N/A"
    lane_pos = None
    try:
        lane_pos = sce.getLanePosition(ego)
    except Exception:
        lane_pos = None

    front_vehicle = None
    rear_vehicle = None
    try:
        front_vehicle, rear_vehicle = road.neighbour_vehicles(ego, ego.lane_index)
    except Exception:
        pass

    front_gap = None
    front_rel_speed = None
    if front_vehicle is not None:
        front_gap = float(np.linalg.norm(ego.position - front_vehicle.position))
        front_rel_speed = float(ego.speed - front_vehicle.speed)

    min_surrounding_gap = None
    try:
        nearby = sce.getSurrendVehicles(10)
        if nearby:
            min_surrounding_gap = min(float(np.linalg.norm(ego.position - v.position)) for v in nearby)
    except Exception:
        pass

    danger = "SAFE"
    if front_gap is not None:
        if front_gap < 10 or (front_gap < 25 and (front_rel_speed or 0) > 0):
            danger = "DANGER"
        elif front_gap < 35:
            danger = "CAUTION"
    if min_surrounding_gap is not None and min_surrounding_gap < 8:
        danger = "DANGER"

    return {
        "speed": float(ego.speed),
        "accel": float(ego.action.get("acceleration", 0.0)),
        "lane_id": lane_id,
        "lane_pos": lane_pos,
        "front_gap": front_gap,
        "front_rel_speed": front_rel_speed,
        "min_surrounding_gap": min_surrounding_gap,
        "danger": danger,
    }


def print_ego_telemetry(step_idx: int, telemetry: dict):
    color = {"SAFE": "green", "CAUTION": "yellow", "DANGER": "red"}.get(telemetry["danger"], "white")
    table = Table(title=f"Ego Telemetry | Step {step_idx}", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Speed (m/s)", f"{telemetry['speed']:.2f}")
    table.add_row("Acceleration (m/s^2)", f"{telemetry['accel']:.2f}")
    table.add_row("Lane", str(telemetry["lane_id"]))
    table.add_row("Lane Position (m)", "N/A" if telemetry["lane_pos"] is None else f"{telemetry['lane_pos']:.2f}")
    table.add_row("Front Gap (m)", "N/A" if telemetry["front_gap"] is None else f"{telemetry['front_gap']:.2f}")
    table.add_row(
        "Front Relative Speed (m/s)",
        "N/A" if telemetry["front_rel_speed"] is None else f"{telemetry['front_rel_speed']:.2f}"
    )
    table.add_row(
        "Nearest Vehicle Gap (m)",
        "N/A" if telemetry["min_surrounding_gap"] is None else f"{telemetry['min_surrounding_gap']:.2f}"
    )
    table.add_row("Danger Indicator", f"[{color}]{telemetry['danger']}[/{color}]")
    print(table)


def setup_env(config):
    selected_model = configure_runtime_env(config)
    if config['OPENAI_API_TYPE'] == 'ollama':
        print(f"[bold yellow]Configured for Local Ollama: {selected_model}[/bold yellow]")

    return build_highway_env_config(
        config,
        show_trajectories=True,
        render_agent=True,
    )


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    env_config = setup_env(config)

    REFLECTION = config["reflection_module"]
    reflection_interactive = _to_bool(config.get("reflection_interactive", True), default=True)
    reflection_auto_add = _to_bool(config.get("reflection_auto_add", False), default=False)
    reflection_add_every_n = int(config.get("reflection_add_every_n", 5))
    if reflection_add_every_n <= 0:
        reflection_add_every_n = 1
    memory_path = config["memory_path"]
    few_shot_num = config["few_shot_num"]
    result_folder = config["result_folder"]
    ensure_dir(result_folder)
    log_path = os.path.join(result_folder, "log.txt")
    with open(log_path, 'w') as f:
        f.write("memory_path {} | result_folder {} | few_shot_num: {} | lanes_count: {} \n".format(
            memory_path, result_folder, few_shot_num, env_config['highway-v0']['lanes_count']))
        f.write("reflection_module {} | reflection_interactive {} | reflection_auto_add {} | reflection_add_every_n {} \n".format(
            REFLECTION, reflection_interactive, reflection_auto_add, reflection_add_every_n
        ))

    agent_memory = DrivingMemory(db_path=memory_path)
    if REFLECTION:
        updated_memory = DrivingMemory(db_path=memory_path + "_updated")
        updated_memory.combineMemory(agent_memory)

    episode = 0
    while episode < config["episodes_num"]:
        # setup highway-env
        envType = 'highway-v0'
        env = gym.make(envType, render_mode="rgb_array")
        env.configure(env_config[envType])
        env = TelemetryHUDWrapper(env)
        result_prefix = f"highway_{episode}"
        env = RecordVideo(env, result_folder, name_prefix=result_prefix)
        env.unwrapped.set_record_video_wrapper(env)
        seed = random.choice(DEFAULT_DILU_SEEDS)
        obs, info = env.reset(seed=seed)
        env.render()

        # scenario and driver agent setting
        database_path = os.path.join(result_folder, f"{result_prefix}.db")
        sce = EnvScenario(env, envType, seed, database_path)
        DA = DriverAgent(sce, verbose=True)
        if REFLECTION:
            RA = ReflectionAgent(verbose=True)

        response = "Not available"
        action = "Not available"
        docs = []
        collision_frame = -1
        crashed = False

        try:
            already_decision_steps = 0
            for i in range(0, config["simulation_duration"]):
                obs = np.array(obs, dtype=float)

                print("[cyan]Retreive similar memories...[/cyan]")
                fewshot_results = agent_memory.retriveMemory(
                    sce, i, few_shot_num) if few_shot_num > 0 else []
                fewshot_messages = []
                fewshot_answers = []
                fewshot_actions = []
                for fewshot_result in fewshot_results:
                    fewshot_messages.append(
                        fewshot_result["human_question"])
                    fewshot_answers.append(fewshot_result["LLM_response"])
                    fewshot_actions.append(fewshot_result["action"])
                if few_shot_num == 0:
                    print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")
                else:
                    print("[green4]Successfully find[/green4]", len(
                        fewshot_actions), "[green4]similar memories![/green4]")

                sce_descrip = sce.describe(i)
                avail_action = sce.availableActionsDescription()
                print_ego_telemetry(i, get_ego_telemetry(sce))
                print('[cyan]Scenario description: [/cyan]\n', sce_descrip)
                # print('[cyan]Available actions: [/cyan]\n',avail_action)
                action, response, human_question, fewshot_answer = DA.few_shot_decision(
                    scenario_description=sce_descrip, available_actions=avail_action,
                    previous_decisions=action,
                    fewshot_messages=fewshot_messages,
                    driving_intensions="Drive safely and avoid collisons",
                    fewshot_answers=fewshot_answers,
                )
                docs.append({
                    "sce_descrip": sce_descrip,
                    "human_question": human_question,
                    "response": response,
                    "action": action,
                    # Deep-copying the scenario is unnecessary here and can fail due to wrapper internals (e.g., PIL font objects).
                    "sce": None
                })

                #obs, reward, done, info, _ = env.step(action)
                if isinstance(action, str):
                    action = int(action)
                action = int(action)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                crashed = bool(info.get("crashed", False))

                already_decision_steps += 1

                env.render()
                sce.promptsCommit(i, None, done, human_question,
                                  fewshot_answer, response)
                #env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()

                print("--------------------")

                if done:
                    print(f"[red]Episode ended at step {i}. terminated={terminated}, truncated={truncated}, info={info}[/red]")
                    if crashed:
                        print("[red]Simulation crash after running steps: [/red] ", i)
                        collision_frame = i
                    else:
                        print("[yellow]Episode ended without collision (e.g., timeout/truncation).[/yellow]")
                    break
        finally:

            with open(log_path, 'a') as f:
                f.write("Simulation {} | Seed {} | Steps: {} | File prefix: {} \n".format(
                    episode, seed, already_decision_steps, result_prefix))
                
            if REFLECTION:
                print("[yellow]Now running reflection agent...[/yellow]")
                if collision_frame != -1: # End with collision
                    for i in range(collision_frame, -1, -1):
                        if docs[i]["action"] != 4:  # not decelearate
                            corrected_response = RA.reflection(
                                docs[i]["human_question"], docs[i]["response"])

                            if reflection_auto_add:
                                choice = 'Y'
                            elif reflection_interactive:
                                choice = input("[yellow]Do you want to add this new memory item to update memory module? (Y/N): ").strip().upper()
                            else:
                                choice = 'N'

                            if choice == 'Y':
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    corrected_response,
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="mistake-correction"
                                )
                                print("[green] Successfully add a new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                            else:
                                print("[blue]Ignore this new memory item[/blue]")
                            break
                else:
                    planned_additions = len(docs) // reflection_add_every_n
                    print("[yellow]Do you want to add[/yellow]", planned_additions, "[yellow]new memory item to update memory module?[/yellow]", end="")
                    if reflection_auto_add:
                        choice = 'Y'
                    elif reflection_interactive:
                        choice = input("(Y/N): ").strip().upper()
                    else:
                        choice = 'N'
                    if choice == 'Y':
                        cnt = 0
                        for i in range(0, len(docs)):
                            if i % reflection_add_every_n == 1:
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    docs[i]["response"],
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="no-mistake-direct"
                                )
                                cnt +=1
                        print("[green] Successfully add[/green] ",cnt," [green]new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                    else:
                        print("[blue]Ignore these new memory items[/blue]")
            

            print("==========Simulation {} Done==========".format(episode))
            episode += 1

            try:
                env.close()
            except Exception:
                pass


