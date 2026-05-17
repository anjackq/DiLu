from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from highway_env.road.road import LaneIndex, Route
from highway_env.utils import not_zero, wrap_to_pi
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from .policy import CompiledPolicy, PolicyCompilationError


@dataclass(frozen=True)
class VehicleConfig:
    kp_a: float = ControlledVehicle.KP_A
    kp_lat: float = ControlledVehicle.KP_LATERAL
    kp_psi: float = ControlledVehicle.KP_HEADING
    acceleration: float = 6.0
    comfortable_deceleration: float = 6.0
    acceleration_exponent: float = 4.0
    desired_time_headway: float = 1.5
    minimum_spacing: float = 6.0


class ControlDigitalTwin:
    TAU_PURSUIT = ControlledVehicle.TAU_PURSUIT
    MAX_STEERING_ANGLE = ControlledVehicle.MAX_STEERING_ANGLE

    def __init__(self, ego_vehicle: Vehicle | None = None) -> None:
        self.ego_vehicle: Vehicle | None = None
        self.vehicle_cfg = VehicleConfig()
        self.target_speed = 0.0
        self.target_lane_index: LaneIndex | None = None
        self.route: Route = []
        self.policy: Iterable[Any] = iter(())
        self.policy_status = "not_started"
        self.policy_error: str | None = None
        self.last_policy_source = ""
        self.policy_failed = False
        if ego_vehicle is not None:
            self.reset(ego_vehicle)

    def reset(self, ego_vehicle: Vehicle) -> None:
        self.ego_vehicle = ego_vehicle
        self.vehicle_cfg = VehicleConfig()
        self.target_speed = float(ego_vehicle.speed)
        self.target_lane_index = ego_vehicle.lane_index
        self.route = []
        self.policy = iter(())
        self.policy_status = "not_started"
        self.policy_error = None
        self.last_policy_source = ""
        self.policy_failed = False

    @property
    def speed(self) -> float:
        return float(self.ego_vehicle.speed)

    def api_bindings(self) -> dict[str, Any]:
        return {
            "get_ego_vehicle": self.get_ego_vehicle,
            "get_desired_time_headway": self.get_desired_time_headway,
            "get_target_speed": self.get_target_speed,
            "say": self.say,
            "is_safe_enter": self.is_safe_enter,
            "set_desired_time_headway": self.set_desired_time_headway,
            "set_target_speed": self.set_target_speed,
            "set_target_lane": self.set_target_lane,
            "autopilot": self.autopilot,
            "get_speed_of": self.get_speed_of,
            "get_lane_of": self.get_lane_of,
            "detect_front_vehicle_in": self.detect_front_vehicle_in,
            "detect_rear_vehicle_in": self.detect_rear_vehicle_in,
            "get_distance_between_vehicles": self.get_distance_between_vehicles,
            "get_left_lane": self.get_left_lane,
            "get_right_lane": self.get_right_lane,
        }

    def execute(self, compiled_policy: CompiledPolicy) -> None:
        self.last_policy_source = compiled_policy.source_code
        self.policy_failed = False
        self.policy_error = None
        try:
            self.policy = compiled_policy.instantiate(self.api_bindings())
            self.policy_status = "ok"
        except PolicyCompilationError as exc:
            self.policy = iter(())
            self.policy_status = "compile_error"
            self.policy_error = str(exc)
            self.policy_failed = True

    def reset_policy(self) -> None:
        if self.target_lane_index is None and self.ego_vehicle is not None:
            self.target_lane_index = self.ego_vehicle.lane_index
        self.policy = iter(())

    def act(self) -> np.ndarray:
        try:
            _ = next(self.policy)
        except StopIteration:
            self.reset_policy()
        except Exception as exc:
            self.policy_failed = True
            self.policy_status = "runtime_error"
            self.policy_error = str(exc)
            self.reset_policy()
        return self.autopilot()

    def get_ego_vehicle(self) -> Vehicle:
        return self.ego_vehicle

    def get_desired_time_headway(self) -> float:
        return float(self.vehicle_cfg.desired_time_headway)

    def get_target_speed(self) -> float:
        return float(self.target_speed)

    @staticmethod
    def say(_text: str) -> None:
        return None

    def set_desired_time_headway(self, desired_time_headway: float) -> None:
        value = max(0.01, min(float(desired_time_headway), 2.0))
        self.vehicle_cfg = VehicleConfig(
            kp_a=self.vehicle_cfg.kp_a,
            kp_lat=self.vehicle_cfg.kp_lat,
            kp_psi=self.vehicle_cfg.kp_psi,
            acceleration=self.vehicle_cfg.acceleration,
            comfortable_deceleration=self.vehicle_cfg.comfortable_deceleration,
            acceleration_exponent=self.vehicle_cfg.acceleration_exponent,
            desired_time_headway=value,
            minimum_spacing=self.vehicle_cfg.minimum_spacing,
        )

    def set_target_speed(self, target_speed: float) -> None:
        self.target_speed = float(target_speed)

    def set_target_lane(self, target_lane: LaneIndex) -> None:
        if len(target_lane) != 3:
            raise ValueError("Invalid target_lane")
        self.target_lane_index = target_lane

    @staticmethod
    def get_speed_of(vehicle: Vehicle) -> float:
        return float(vehicle.speed)

    @staticmethod
    def get_lane_of(vehicle: Vehicle) -> LaneIndex:
        return vehicle.lane_index

    @staticmethod
    def get_distance_between_vehicles(front_vehicle: Vehicle, rear_vehicle: Vehicle) -> float:
        return float(-front_vehicle.lane_distance_to(rear_vehicle))

    def get_left_lane(self, vehicle: Vehicle | None = None, lane_index: LaneIndex | None = None) -> LaneIndex | None:
        vehicle = vehicle or self.ego_vehicle
        lane_from, lane_to, lane_id = lane_index or vehicle.lane_index
        target_lane_index = (lane_from, lane_to, int(np.clip(lane_id - 1, 0, len(vehicle.road.network.graph[lane_from][lane_to]) - 1)))
        return target_lane_index if target_lane_index[2] != int(lane_id) else None

    def get_right_lane(self, vehicle: Vehicle | None = None, lane_index: LaneIndex | None = None) -> LaneIndex | None:
        vehicle = vehicle or self.ego_vehicle
        lane_from, lane_to, lane_id = lane_index or vehicle.lane_index
        target_lane_index = (lane_from, lane_to, int(np.clip(lane_id + 1, 0, len(vehicle.road.network.graph[lane_from][lane_to]) - 1)))
        return target_lane_index if target_lane_index[2] != int(lane_id) else None

    def _detect_front_and_rear_vehicles_in_lane(self, lane_index: LaneIndex):
        return self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, lane_index)

    def detect_front_vehicle_in(self, lane: LaneIndex, distance: float = 100):
        front, _rear = self._detect_front_and_rear_vehicles_in_lane(lane)
        if front and 0 < self.ego_vehicle.lane_distance_to(front) < distance:
            return front
        return None

    def detect_rear_vehicle_in(self, lane: LaneIndex, distance: float = 100):
        _front, rear = self._detect_front_and_rear_vehicles_in_lane(lane)
        if rear and -distance < self.ego_vehicle.lane_distance_to(rear) < 0:
            return rear
        return None

    def is_safe_enter(self, lane: LaneIndex, safe_decel: float = 5) -> bool:
        if lane is None:
            return False
        front, rear = self._detect_front_and_rear_vehicles_in_lane(lane)
        if front is not None:
            ego_decel = self._intelligent_driver_model(target_speed=self.get_speed_of(self.ego_vehicle), ego=self.ego_vehicle, front=front)
            if ego_decel < -safe_decel:
                return False
        if rear is not None:
            rear_decel = self._intelligent_driver_model(target_speed=self.get_speed_of(rear), ego=rear, front=self.ego_vehicle)
            if rear_decel < -safe_decel:
                return False
        return True

    def autopilot(self) -> np.ndarray:
        self._follow_road()
        return np.array(
            [
                self._intelligent_driver_model(self.target_speed),
                self._steering_control(self.target_lane_index),
            ]
        )

    def _follow_road(self) -> None:
        if self.ego_vehicle.road.network.get_lane(self.target_lane_index).after_end(self.ego_vehicle.position):
            self.target_lane_index = self.ego_vehicle.road.network.next_lane(
                self.target_lane_index,
                position=self.ego_vehicle.position,
                np_random=self.ego_vehicle.road.np_random,
                route=self.route,
            )

    def _front_vehicle(self):
        return self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]

    def _compute_distance_headway(self, ego: Vehicle | None = None, front: Vehicle | None = None, time_headway: float | None = None) -> float:
        ego = ego or self.ego_vehicle
        front = front or self._front_vehicle()
        tau = float(time_headway if time_headway is not None else self.vehicle_cfg.desired_time_headway)
        if front is None:
            return -1.0
        d0 = self.vehicle_cfg.minimum_spacing
        ab = self.vehicle_cfg.acceleration * self.vehicle_cfg.comfortable_deceleration
        dv = np.dot(ego.velocity - front.velocity, ego.direction)
        return float(d0 + ego.speed * tau + ego.speed * dv / (2 * np.sqrt(ab)))

    def _intelligent_driver_model(self, target_speed: float, ego: Vehicle | None = None, front: Vehicle | None = None) -> float:
        ego = ego or self.ego_vehicle
        front = front or self._front_vehicle()
        desired_accel_max = self.vehicle_cfg.acceleration
        delta = self.vehicle_cfg.acceleration_exponent
        accel = desired_accel_max * (1 - np.power(max(ego.speed, 0) / abs(not_zero(target_speed)), delta))
        if front is not None:
            distance = ego.lane_distance_to(front)
            d_star = self._compute_distance_headway(ego, front)
            accel -= desired_accel_max * (np.power(d_star / not_zero(distance), 2))
        return float(accel)

    def _steering_control(self, target_lane_index: LaneIndex) -> float:
        target_lane = self.ego_vehicle.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.ego_vehicle.position)
        lane_next_coords = lane_coords[0] + self.ego_vehicle.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)
        delta_lat = lane_coords[1]
        lateral_velocity_command = -self.vehicle_cfg.kp_lat * delta_lat
        heading_command = np.arcsin(np.clip(lateral_velocity_command / not_zero(self.ego_vehicle.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
        heading_rate_command = self.vehicle_cfg.kp_psi * wrap_to_pi(heading_ref - self.ego_vehicle.heading)
        slip_angle = np.arcsin(
            np.clip(self.ego_vehicle.LENGTH / 2 / not_zero(self.ego_vehicle.speed) * heading_rate_command, -1, 1)
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)
