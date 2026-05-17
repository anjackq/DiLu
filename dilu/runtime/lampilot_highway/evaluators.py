from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from highway_env.envs import AbstractEnv
from highway_env.road.lane import LineType
from highway_env.road.road import LaneIndex
from highway_env.utils import not_zero
from highway_env.vehicle.kinematics import Vehicle

from .envs import ensure_envs_registered
from .metrics import compute_ttc, ordinal


class BaseHighwayEvaluator:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        show_window: bool = False,
        record_video: bool = False,
        wait_time: float = 0.0,
        safe_ttc_threshold: float = 2.0,
        speed_std_threshold: float = 10.0,
        time_threshold: float = 60.0,
        score_weights: dict[str, float] | None = None,
        video_dir: str = "",
    ) -> None:
        ensure_envs_registered()
        self.exp_time = time.strftime("%Y%m%d-%H%M%S")
        self.config = dict(config)
        self.record_video = bool(record_video)
        self.show_window = bool(show_window)
        self.wait_time = float(wait_time)
        self.video_dir = video_dir
        self.safe_ttc_threshold = float(safe_ttc_threshold)
        self.speed_std_threshold = float(speed_std_threshold)
        self.time_threshold = float(time_threshold)
        self.score_weights = score_weights or {"ttc": 0.5, "speed_variance": 0.3, "time_efficiency": 0.2}
        self._init_env(self.config)
        self.simulation_frequency = self.env.unwrapped.config["simulation_frequency"]
        self.frame = 0
        self.queue: deque[dict[str, Any]] = deque(maxlen=10000)
        self._init_ego_vehicle()
        self.done = False
        self.truncated = False
        self.success = False
        self.collision = False

    def _init_env(self, config: dict[str, Any]) -> None:
        self.env: AbstractEnv = gym.make(config["env"]["type"], render_mode="rgb_array")
        self.env.unwrapped.configure(config["env"])
        if self.record_video:
            self.env = RecordVideo(self.env, self.video_dir, name_prefix=f"{self.exp_time}")
            self.env.unwrapped.set_record_video_wrapper(self.env)
        self.env.reset(seed=config["seed"])
        for _ in range(3 * self.env.unwrapped.config["simulation_frequency"]):
            self.env.step(np.array([0.0, 0.0]))

    def _init_ego_vehicle(self) -> None:
        self.ego_vehicle = self.env.unwrapped.vehicle
        self.ego_vehicle.speed = 20
        speed_limit = self.ego_vehicle.road.network.get_lane(self.ego_vehicle.lane_index).speed_limit
        if speed_limit:
            self.ego_vehicle.speed = speed_limit
        front_vehicle = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]
        if front_vehicle is not None and self.ego_vehicle.lane_distance_to(front_vehicle) < 50:
            self.ego_vehicle.speed = front_vehicle.speed

    def step(self, agent: Any) -> None:
        if self.show_window:
            self.env.render()
            time.sleep(self.wait_time)
        action = agent.act()
        _, _, self.done, self.truncated, _ = self.env.step(action)
        self.collision = self.ego_vehicle.crashed or not self.ego_vehicle.on_road
        self.queue.append(
            {
                "acceleration": float(action[0]),
                "steering": float(action[1]),
                "speed": float(agent.speed),
                "ttc": compute_ttc(self.env),
            }
        )
        self.frame += 1

    def close(self) -> None:
        self.env.close()

    @property
    def ended(self) -> bool:
        return bool(self.done or self.truncated)

    @property
    def overall_time(self) -> float:
        return self.frame / self.simulation_frequency

    @property
    def speed_std(self) -> float:
        if not self.queue:
            return 0.0
        return float(np.std([item["speed"] for item in self.queue]).item())

    @property
    def score_ttc(self) -> float:
        ttcs = [item["ttc"] for item in self.queue if item["ttc"] > 0]
        if not ttcs:
            return 100.0
        min_ttc = min(ttcs)
        if min_ttc > self.safe_ttc_threshold:
            return 100.0
        return max(100.0 - (1 / min_ttc), -100.0)

    @property
    def score_speed_variance(self) -> float:
        return 100 * (1 - self.speed_std / self.speed_std_threshold)

    @property
    def score_time_efficiency(self) -> float:
        return 100 * (1 - self.overall_time / self.time_threshold)

    @property
    def score(self) -> float:
        if not self.success:
            return 0.0
        return max(
            0.0,
            self.score_ttc * self.score_weights["ttc"]
            + self.score_speed_variance * self.score_weights["speed_variance"]
            + self.score_time_efficiency * self.score_weights["time_efficiency"],
        )

    @property
    def _lanes(self) -> tuple[int, int, bool]:
        lane_from, lane_to, lane_id = self.ego_vehicle.lane_index
        lane_count = int(len(self.ego_vehicle.road.network.graph[lane_from][lane_to]))
        right_count = lane_count - lane_id
        right_most_lane_index = (lane_from, lane_to, lane_count - 1)
        emergency = self.ego_vehicle.road.network.get_lane(right_most_lane_index).line_types == (
            LineType.CONTINUOUS_LINE,
            LineType.CONTINUOUS_LINE,
        )
        return lane_count, right_count, emergency

    @property
    def _front_vehicle(self) -> tuple[Vehicle | None, float | None, float | None]:
        front = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]
        if front:
            front_distance = self.ego_vehicle.lane_distance_to(front)
            if front_distance < 100:
                return front, float(front_distance), float(front.speed)
        return None, None, None

    def get_context_info(self) -> str:
        env_type = str(self.config["env"]["type"])
        if env_type in {"ramp-merge-v0", "dt-highway-v0"}:
            front, front_distance, front_speed = self._front_vehicle
            lane_count, right_count, emergency = self._lanes
            context = f"My current speed is {self.ego_vehicle.speed:.1f} m/s. "
            context += (
                f"I am driving on a highway with {lane_count:d} lanes in my direction, "
                f"and I am in the {ordinal(right_count)} lane from the right. "
            )
            if emergency:
                context += "The right-most lane is an emergency lane. "
            if front:
                context += (
                    f"There is a car in front of me in my lane, at a distance of {front_distance:.1f} m, "
                    f"with a speed of {front_speed:.1f} m/s. "
                )
            else:
                context += "There is no car ahead of me in my lane. "
            return context
        raise RuntimeError(f"Unsupported env type {env_type}")


class ACCEvalbySpeed(BaseHighwayEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if "speed" in self.config["eval"]:
            self.desired_speed = self.config["eval"]["speed"]
        elif "rel_speed" in self.config["eval"]:
            self.desired_speed = self.ego_vehicle.speed + self.config["eval"]["rel_speed"]
        else:
            raise ValueError("ACCEvalbySpeed requires speed or rel_speed.")
        self.last_time = -1.0
        self.time_duration = 5.0
        self.failure_time = 60.0
        self.max_gap = 100.0
        self.failure_start_time = self.overall_time

    def step(self, agent: Any) -> None:
        super().step(agent)
        self.front_vehicle = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]
        if self.front_vehicle and self.ego_vehicle.lane_distance_to(self.front_vehicle) > 100:
            self.front_vehicle = None
        if self.last_time == -1 and self.front_vehicle is None and abs(self.ego_vehicle.speed - self.desired_speed) < 1:
            self.last_time = self.overall_time
        elif (
            self.last_time == -1
            and self.front_vehicle is not None
            and self.front_vehicle.speed > self.desired_speed
            and abs(self.desired_speed - self.ego_vehicle.speed) < 1
        ):
            self.last_time = self.overall_time
        elif (
            self.last_time == -1
            and self.front_vehicle is not None
            and self.front_vehicle.speed < self.desired_speed
            and abs(self.front_vehicle.speed - self.ego_vehicle.speed) < 1
            and abs(self.ego_vehicle.lane_distance_to(self.front_vehicle)) < self.max_gap
        ):
            self.last_time = self.overall_time
        elif self.last_time > 0:
            if self.overall_time - self.last_time > self.time_duration:
                self.done = True
                self.success = True
            elif self.front_vehicle is None and abs(self.ego_vehicle.speed - self.desired_speed) > 1:
                self.last_time = -1
            elif (
                self.front_vehicle is not None
                and self.front_vehicle.speed > self.desired_speed
                and abs(self.desired_speed - self.ego_vehicle.speed) > 1
            ):
                self.last_time = -1
            elif (
                self.front_vehicle is not None
                and self.front_vehicle.speed < self.desired_speed
                and abs(self.front_vehicle.speed - self.ego_vehicle.speed) > 1
            ):
                self.last_time = -1
        if self.overall_time - self.failure_start_time > self.failure_time:
            self.done = True
            self.success = False


class ACCEvalbyDistance(BaseHighwayEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.front_vehicle = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]
        if "distance" in self.config["eval"]:
            self.desired_distance = self.config["eval"]["distance"]
        elif "rel_distance" in self.config["eval"]:
            self.desired_distance = self.config["eval"]["rel_distance"] + self.ego_vehicle.lane_distance_to(self.front_vehicle)
        else:
            raise ValueError("ACCEvalbyDistance requires distance or rel_distance.")
        self.last_time = -1.0
        self.time_duration = 5.0
        self.failure_time = 60.0
        self.max_gap = 60.0
        self.failure_start_time = self.overall_time
        self.init_ego_speed = self.ego_vehicle.speed
        self.dis_tol = 0.3 * self.desired_distance
        if self.front_vehicle is None:
            raise AssertionError("No front vehicle detected.")
        self.start_lane_index = self.ego_vehicle.lane_index
        start_lane = self.ego_vehicle.road.network.get_lane(self.start_lane_index)
        if abs(self.ego_vehicle.lane_distance_to(self.front_vehicle, start_lane)) >= 100:
            raise AssertionError("Front vehicle is too far away from the ego vehicle.")

    def step(self, agent: Any) -> None:
        super().step(agent)
        if self.front_vehicle.lane_index != self.ego_vehicle.lane_index:
            self.front_vehicle = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]
        if self.front_vehicle is None and self.last_time == -1:
            if np.isclose(self.ego_vehicle.speed, self.init_ego_speed, atol=0.1):
                self.last_time = self.overall_time
            else:
                self.last_time = -1
        elif self.front_vehicle is not None and self.last_time == -1:
            current_gap = abs(self.ego_vehicle.lane_distance_to(self.front_vehicle))
            if (self.desired_distance - self.dis_tol) < current_gap < (self.desired_distance + self.dis_tol):
                self.last_time = self.overall_time
            else:
                self.last_time = -1
        elif self.last_time > 0:
            if self.overall_time - self.last_time > self.time_duration:
                self.done = True
                self.success = True
            elif self.front_vehicle is not None:
                current_gap = abs(self.ego_vehicle.lane_distance_to(self.front_vehicle))
                if current_gap < (self.desired_distance - self.dis_tol) or current_gap > (self.desired_distance + self.dis_tol):
                    self.last_time = -1
            elif self.front_vehicle is None and np.isclose(self.ego_vehicle.speed, self.init_ego_speed, atol=0.1):
                self.last_time = -1
        if self.overall_time - self.failure_start_time > self.failure_time:
            self.done = True
            self.success = False


class LaneChangeEval(BaseHighwayEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_lane_index: LaneIndex = self.ego_vehicle.lane_index
        lane_from, lane_to, lane_id = self.start_lane_index
        if self.config["eval"]["direction"] == "left":
            direction = -1
        elif self.config["eval"]["direction"] == "right":
            direction = 1
        else:
            raise ValueError(f"Invalid direction {self.config['eval']['direction']}")
        road = self.ego_vehicle.road
        self.target_lane_index = (
            lane_from,
            lane_to,
            int(np.clip(lane_id + direction, 0, len(road.network.graph[lane_from][lane_to]) - 1)),
        )
        self.target_lane = road.network.get_lane(self.target_lane_index)
        if self.target_lane_index == self.start_lane_index:
            raise AssertionError("Target lane index is the same as the start lane index.")
        if not road.network.get_lane(self.target_lane_index).is_reachable_from(self.ego_vehicle.position):
            raise AssertionError("Target lane index is not reachable from the current position.")

    def step(self, agent: Any) -> None:
        super().step(agent)
        if self.ego_vehicle.lane_index == self.target_lane_index:
            ego_heading = self.ego_vehicle.heading
            lane_coords = self.target_lane.local_coordinates(self.ego_vehicle.position)
            lane_heading = self.target_lane.heading_at(lane_coords[0])
            if abs(ego_heading - lane_heading) < np.deg2rad(5):
                self.done = True
                self.success = True


class OvertakeEval(BaseHighwayEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.config["eval"]["direction"] == "left":
            direction = -1
        elif self.config["eval"]["direction"] == "right":
            direction = 1
        else:
            raise ValueError(f"Invalid direction {self.config['eval']['direction']}")
        self.start_lane_index = self.ego_vehicle.lane_index
        lane_from, lane_to, lane_id = self.start_lane_index
        road = self.ego_vehicle.road
        self.target_lane_index = (
            lane_from,
            lane_to,
            int(np.clip(lane_id + direction, 0, len(road.network.graph[lane_from][lane_to]) - 1)),
        )
        self.target_lane = road.network.get_lane(self.target_lane_index)
        self.front_vehicle = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]
        if self.target_lane_index == self.start_lane_index:
            raise AssertionError("Target lane index is the same as the start lane index.")
        if not road.network.get_lane(self.target_lane_index).is_reachable_from(self.ego_vehicle.position):
            raise AssertionError("Target lane index is not reachable from the current position.")
        if self.front_vehicle is None:
            raise AssertionError("No front vehicle detected.")
        start_lane = road.network.get_lane(self.start_lane_index)
        if abs(self.ego_vehicle.lane_distance_to(self.front_vehicle, start_lane)) >= 100:
            raise AssertionError("Front vehicle is too far away from the ego vehicle.")

    def step(self, agent: Any) -> None:
        super().step(agent)
        if (
            self.ego_vehicle.lane_index == self.target_lane_index
            and self.ego_vehicle.lane_distance_to(self.front_vehicle, self.target_lane) < -2 * self.ego_vehicle.LENGTH
        ):
            self.done = True
            self.success = True


class PullOverEval(BaseHighwayEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        lane_from, lane_to, _lane_id = self.ego_vehicle.lane_index
        most_right_lane_index = (lane_from, lane_to, int(len(self.ego_vehicle.road.network.graph[lane_from][lane_to]) - 1))
        if self.ego_vehicle.road.network.get_lane(most_right_lane_index).line_types == (
            LineType.CONTINUOUS_LINE,
            LineType.CONTINUOUS_LINE,
        ):
            self.emergency_lane_index = most_right_lane_index
        else:
            self.emergency_lane_index = None
        if self.emergency_lane_index is None:
            raise AssertionError("No emergency lane")

    def step(self, agent: Any) -> None:
        super().step(agent)
        if self.ego_vehicle.lane_index == self.emergency_lane_index and np.isclose(self.ego_vehicle.speed, 0.0, atol=5e-1):
            self.done = True
            self.success = True


def get_evaluator_class(name: str) -> type[BaseHighwayEvaluator]:
    mapping: dict[str, type[BaseHighwayEvaluator]] = {
        "ACCEvalbySpeed": ACCEvalbySpeed,
        "ACCEvalbyDistance": ACCEvalbyDistance,
        "LaneChangeEval": LaneChangeEval,
        "OvertakeEval": OvertakeEval,
        "PullOverEval": PullOverEval,
    }
    if name not in mapping:
        raise KeyError(f"Unsupported evaluator class: {name}")
    return mapping[name]
