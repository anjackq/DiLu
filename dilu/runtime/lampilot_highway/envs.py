from __future__ import annotations

from typing import Dict

import highway_env  # noqa: F401
import numpy as np
from gymnasium.envs.registration import register, registry
from highway_env.envs.merge_env import MergeEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import class_from_path
from highway_env.vehicle.objects import Obstacle

from .env_utils import create_random_vehicle_highway


class RampMergeEnv(MergeEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {"type": "ContinuousAction"},
                "stage_length": [500, 80, 80, 2000],
                "screen_width": 2000,
                "screen_height": 300,
                "vehicles_count": 20,
                "vehicles_density": 1.0,
                "truncate_after_meter": 2400,
                "duration": 1000,
                "show_trajectories": True,
                "policy_frequency": 10,
                "simulation_frequency": 10,
                "ego_vehicle": {
                    "speed": 20,
                    "start_lane_index": ("j", "k", 0),
                    "start_position": [380, 0],
                },
            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        return 0.0

    def _rewards(self, action: int) -> Dict[str, float]:
        return {"total_reward": 0.0}

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or not self.vehicle.on_road

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"] or bool(self.vehicle.position[0] > self.config["truncate_after_meter"])

    def _make_vehicles(self) -> None:
        ego_vehicle_type = self.action_type.vehicle_class
        ego_vehicle = ego_vehicle_type(
            self.road,
            self.road.network.get_lane(self.config["ego_vehicle"]["start_lane_index"]).position(
                self.config["ego_vehicle"]["start_position"][0],
                self.config["ego_vehicle"]["start_position"][1],
            ),
            speed=self.config["ego_vehicle"]["speed"],
        )
        other_vehicle_type = class_from_path(self.config["other_vehicles_type"])
        for index in range(self.config["vehicles_count"]):
            vehicle = create_random_vehicle_highway(
                other_vehicle_type,
                self.road,
                spacing=1 / self.config["vehicles_density"],
                lane_from="a" if index == 0 else None,
                lane_to="b" if index == 0 else None,
            )
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def _make_road(self) -> None:
        net = RoadNetwork()
        highway_speed_limit = 31.2928
        ramp_speed_limit = 24.5872
        before, converging, merge, after = self.config["stage_length"]
        ends = [before, converging, merge, after]
        continuous, striped, none = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y_coords = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [(continuous, striped), (none, continuous)]
        line_type_merge = [(continuous, striped), (none, striped)]
        for lane_id in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, y_coords[lane_id]],
                    [sum(ends[:2]), y_coords[lane_id]],
                    line_types=line_type[lane_id],
                    speed_limit=highway_speed_limit,
                ),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y_coords[lane_id]],
                    [sum(ends[:3]), y_coords[lane_id]],
                    line_types=line_type_merge[lane_id],
                    speed_limit=highway_speed_limit,
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y_coords[lane_id]],
                    [sum(ends), y_coords[lane_id]],
                    line_types=line_type[lane_id],
                    speed_limit=highway_speed_limit,
                ),
            )
        amplitude = 3.25
        lane_jk = StraightLane(
            [0, 6.5 + 4 + 4],
            [ends[0], 6.5 + 4 + 4],
            line_types=(continuous, continuous),
            forbidden=True,
            speed_limit=ramp_speed_limit,
        )
        lane_kb = SineLane(
            lane_jk.position(ends[0], -amplitude),
            lane_jk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[continuous, continuous],
            forbidden=True,
            speed_limit=ramp_speed_limit,
        )
        lane_bc = StraightLane(
            lane_kb.position(ends[1], 0),
            lane_kb.position(ends[1], 0) + [ends[2], 0],
            line_types=(none, continuous),
            forbidden=True,
            speed_limit=highway_speed_limit,
        )
        net.add_lane("j", "k", lane_jk)
        net.add_lane("k", "b", lane_kb)
        net.add_lane("b", "c", lane_bc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lane_bc.position(ends[2], 0)))
        self.road = road


class DTHighwayEnv(MergeEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {"type": "ContinuousAction"},
                "stage_length": [2000],
                "num_lanes": 5,
                "emergency_lane": True,
                "screen_width": 2000,
                "screen_height": 300,
                "vehicles_count": 20,
                "vehicles_density": 1.0,
                "truncate_after_meter": 2400,
                "duration": 1000,
                "show_trajectories": True,
                "policy_frequency": 10,
                "simulation_frequency": 10,
                "ego_vehicle": {
                    "speed": 20,
                    "start_lane_index": ("a", "b", 0),
                    "start_position": [380, 0],
                },
            }
        )
        return cfg

    def _reward(self, action: int) -> float:
        return 0.0

    def _rewards(self, action: int) -> Dict[str, float]:
        return {"total_reward": 0.0}

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or not self.vehicle.on_road

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"] or bool(self.vehicle.position[0] > self.config["truncate_after_meter"])

    def _make_vehicles(self) -> None:
        ego_vehicle_type = self.action_type.vehicle_class
        ego_vehicle = ego_vehicle_type(
            self.road,
            self.road.network.get_lane(self.config["ego_vehicle"]["start_lane_index"]).position(
                self.config["ego_vehicle"]["start_position"][0],
                self.config["ego_vehicle"]["start_position"][1],
            ),
            speed=self.config["ego_vehicle"]["speed"],
        )
        other_vehicle_type = class_from_path(self.config["other_vehicles_type"])
        for index in range(self.config["vehicles_count"]):
            vehicle = create_random_vehicle_highway(
                other_vehicle_type,
                self.road,
                spacing=1 / self.config["vehicles_density"],
                lane_from="a" if index == 0 else None,
                lane_to="b" if index == 0 else None,
                exclude_emergency_lane=self.config["emergency_lane"],
            )
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def _make_road(self) -> None:
        net = RoadNetwork()
        highway_speed_limit = 31.2928
        stage_length = self.config["stage_length"]
        continuous, striped, none = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y_coords = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [(continuous, striped)]
        if self.config["emergency_lane"]:
            line_type.extend([(none, striped)] * (self.config["num_lanes"] - 2))
            line_type.extend([(none, continuous)] * 2)
        else:
            line_type.extend([(none, striped)] * (self.config["num_lanes"] - 1))
            line_type.extend([(none, continuous)])
        num_regular_lane = self.config["num_lanes"] - 1 if self.config["emergency_lane"] else self.config["num_lanes"]
        for lane_id in range(num_regular_lane):
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, lane_id * y_coords[1]],
                    [sum(stage_length), lane_id * y_coords[1]],
                    line_types=line_type[lane_id],
                    speed_limit=highway_speed_limit,
                ),
            )
        if self.config["emergency_lane"]:
            emergency_lane = StraightLane(
                [0, num_regular_lane * y_coords[1]],
                [sum(stage_length), num_regular_lane * y_coords[1]],
                line_types=(continuous, continuous),
                forbidden=True,
                speed_limit=highway_speed_limit,
            )
            net.add_lane("a", "b", emergency_lane)
        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])


def ensure_envs_registered() -> None:
    specs = {
        "ramp-merge-v0": "dilu.runtime.lampilot_highway.envs:RampMergeEnv",
        "dt-highway-v0": "dilu.runtime.lampilot_highway.envs:DTHighwayEnv",
    }
    for env_id, entry_point in specs.items():
        if env_id in registry:
            continue
        register(id=env_id, entry_point=entry_point)
