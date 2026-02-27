from typing import Any, Dict

import numpy as np


def build_highway_env_config(
    config: Dict[str, Any],
    *,
    show_trajectories: bool,
    render_agent: bool,
    lanes_count: int = 4,
) -> Dict[str, Dict[str, Any]]:
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
            "lanes_count": lanes_count,
            "other_vehicles_type": config["other_vehicle_type"],
            "duration": config["simulation_duration"],
            "vehicles_density": config["vehicles_density"],
            "show_trajectories": show_trajectories,
            "render_agent": render_agent,
            "scaling": 5,
            "initial_lane_id": None,
            "ego_spacing": 4,
        }
    }
