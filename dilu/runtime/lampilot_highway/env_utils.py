from __future__ import annotations

from typing import Optional

import numpy as np
from highway_env.road.road import Road
from highway_env.vehicle.kinematics import Vehicle


def create_random_vehicle_highway(
    vehicle_type,
    road: Road,
    *,
    speed: float | None = None,
    lane_from: Optional[str] = None,
    lane_to: Optional[str] = None,
    lane_id: Optional[int] = None,
    spacing: float = 1.0,
    exclude_emergency_lane: bool = False,
) -> Vehicle:
    lane_from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
    lane_to = lane_to or road.np_random.choice(list(road.network.graph[lane_from].keys()))
    if exclude_emergency_lane:
        lane_index = lane_id if lane_id is not None else road.np_random.choice(
            list(range(0, len(road.network.graph[lane_from][lane_to]) - 1))
        )
    else:
        lane_index = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[lane_from][lane_to]))
    lane = road.network.get_lane((lane_from, lane_to, lane_index))
    if speed is None:
        if lane.speed_limit is not None:
            speed = road.np_random.uniform(0.8 * lane.speed_limit, 1.1 * lane.speed_limit)
        else:
            speed = road.np_random.uniform(Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1])
    default_spacing = 12 + float(speed)
    offset = float(spacing) * default_spacing
    if road.vehicles:
        x0 = np.max([lane.local_coordinates(vehicle.position)[0] for vehicle in road.vehicles])
    else:
        x0 = 12
    x0 += offset * road.np_random.uniform(0.7, 1.3)
    return vehicle_type(road, lane.position(x0, 0), lane.heading_at(x0), speed)
