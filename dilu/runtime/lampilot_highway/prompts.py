from __future__ import annotations


API_DOCS = """
def get_ego_vehicle() -> Vehicle:
    \"\"\"Return the ego vehicle.\"\"\"

def get_desired_time_headway() -> float:
    \"\"\"Return the desired time headway in seconds.\"\"\"

def get_target_speed() -> float:
    \"\"\"Return the target speed in m/s.\"\"\"

def say(text: str):
    \"\"\"Provide a text message to the passenger.\"\"\"

def is_safe_enter(lane: Lane, safe_decel: float = 5) -> bool:
    \"\"\"Return True if the ego vehicle can safely enter the specified lane now.\"\"\"

def set_desired_time_headway(desired_time_headway: float):
    \"\"\"Set the desired time headway in seconds.\"\"\"

def set_target_speed(target_speed: float):
    \"\"\"Set the target speed in m/s.\"\"\"

def set_target_lane(target_lane: Lane):
    \"\"\"Set the target lane.\"\"\"

def autopilot() -> list[float]:
    \"\"\"Return a continuous control command for one step.\"\"\"

def get_speed_of(veh: Vehicle) -> float:
    \"\"\"Return the speed of the vehicle in m/s.\"\"\"

def get_lane_of(veh: Vehicle) -> Lane:
    \"\"\"Return the lane of the vehicle.\"\"\"

def detect_front_vehicle_in(lane: Lane, distance: float = 100) -> Vehicle:
    \"\"\"Return the closest front vehicle in lane within distance, else None.\"\"\"

def detect_rear_vehicle_in(lane: Lane, distance: float = 100) -> Vehicle:
    \"\"\"Return the closest rear vehicle in lane within distance, else None.\"\"\"

def get_distance_between_vehicles(veh1: Vehicle, veh2: Vehicle) -> float:
    \"\"\"Return positive distance if veh1 is in front of veh2.\"\"\"

def get_left_lane(veh: Vehicle) -> Lane:
    \"\"\"Return the left lane if available, else None.\"\"\"

def get_right_lane(veh: Vehicle) -> Lane:
    \"\"\"Return the right lane if available, else None.\"\"\"
""".strip()

RESPONSE_FORMAT = """
Plan:
1) ...
2) ...
Code:
```python
def your_function_name():
    ...
```
""".strip()

ZERO_SHOT_TEMPLATE = """
You are a helpful assistant that writes Python code to complete any autonomous driving task specified by me.

Here are the APIs you can use:

{apis}

I will give you
Command: ...
Context Info: ...

You should then respond to me with
Plan: How to complete the task step by step.
Code:
    1) Write exactly one function taking no arguments.
    2) Only use the APIs that I provided plus basic Python control flow.
    3) Define all variables inside your function.
    4) Do not write helper functions, classes, imports, or infinite loops.
    5) In each loop iteration, include `yield autopilot()` so the vehicle advances one time step.

You must respond in exactly this format:
{response_format}
""".strip()

FEW_SHOT_TEMPLATE = """
{zero_shot}

Here are some examples:

Command: Decrease gap to 25 meters.
Context Info: My current speed is 31.3 m/s. I am driving on a highway with 2 lanes in my direction, and I am in the 1st lane from the right. There is a car in front of me in my lane, at a distance of 57.7 m, with a speed of 21.6 m/s.
Plan:
1) Detect the front vehicle in the current lane.
2) Match the front vehicle speed and adjust desired headway until the gap is near 25 meters.
Code:
```python
def decrease_gap_to_25_meters():
    ego = get_ego_vehicle()
    current_lane = get_lane_of(ego)
    desired_gap = 25
    while True:
        front = detect_front_vehicle_in(current_lane)
        if front is None:
            return
        set_target_speed(get_speed_of(front) * 1.2)
        current_gap = get_distance_between_vehicles(front, ego)
        current_time_headway = get_desired_time_headway()
        if current_gap > desired_gap:
            set_desired_time_headway(current_time_headway - 0.05)
        else:
            set_desired_time_headway(current_time_headway + 0.05)
        yield autopilot()
```

Command: Go around the car in front of you using the left lane.
Context Info: My current speed is 31.3 m/s. I am driving on a highway with 2 lanes in my direction, and I am in the 1st lane from the right. There is a car in front of me in my lane, at a distance of 94.8 m, with a speed of 29.3 m/s.
Plan:
1) Check the left lane and verify it is safe to enter.
2) Move left, then accelerate until the target vehicle is behind the ego vehicle.
Code:
```python
def overtake_using_left_lane():
    ego = get_ego_vehicle()
    current_lane = get_lane_of(ego)
    left_lane = get_left_lane(ego)
    target_vehicle = detect_front_vehicle_in(current_lane)
    if left_lane is None or target_vehicle is None:
        return
    while True:
        if is_safe_enter(left_lane):
            set_target_lane(left_lane)
            break
        yield autopilot()
    while True:
        if get_distance_between_vehicles(ego, target_vehicle) < 0:
            set_target_speed(get_speed_of(target_vehicle) * 1.5)
            yield autopilot()
        else:
            return
```
""".strip()


def render_system_prompt(*, few_shot: bool) -> str:
    zero_shot = ZERO_SHOT_TEMPLATE.format(apis=API_DOCS, response_format=RESPONSE_FORMAT)
    if not few_shot:
        return zero_shot
    return FEW_SHOT_TEMPLATE.format(zero_shot=zero_shot)


def render_user_prompt(command: str, context_info: str) -> str:
    return f"Command: {command}\nContext Info: {context_info}\n"
