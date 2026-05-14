"""
Webots runtime controller for the policy trained in phase_1_3_corner_focused_training.py
(random spawn position + yaw with extra weight on corners and mid-edge poses; fixed
red ball — same 12-D observation as phase 1).
"""
import os
import platform
import sys


def _prepend_webots_controller_python_path():
    """So `import controller` works when the script is not started by Webots."""
    sysname = platform.system()

    if "WEBOTS_HOME" not in os.environ or not os.environ["WEBOTS_HOME"]:
        if sysname == "Darwin":
            default_bundle = "/Applications/Webots.app"
            if os.path.isdir(default_bundle):
                os.environ["WEBOTS_HOME"] = default_bundle
        elif sysname == "Linux":
            for candidate in ("/usr/local/webots",):
                if os.path.isdir(candidate):
                    os.environ["WEBOTS_HOME"] = candidate
                    break

    home = os.environ.get("WEBOTS_HOME") or ""
    if sysname == "Darwin" and home.rstrip("/").endswith("/Contents"):
        fixed = os.path.dirname(home.rstrip("/"))
        if fixed.endswith(".app"):
            os.environ["WEBOTS_HOME"] = fixed
            home = fixed

    candidates = []
    if home:
        if sysname == "Darwin":
            candidates.append(os.path.join(home, "Contents", "lib", "controller", "python"))
        else:
            candidates.append(os.path.join(home, "lib", "controller", "python"))
    if sysname == "Darwin":
        candidates.append("/Applications/Webots.app/Contents/lib/controller/python")
    elif sysname == "Windows":
        candidates.append(r"D:\Webots\lib\controller\python")
        candidates.append(r"C:\Program Files\Webots\lib\controller\python")
    elif sysname == "Linux":
        candidates.append("/usr/local/webots/lib/controller/python")

    for path in candidates:
        if path and os.path.isdir(path):
            sys.path.insert(0, path)
            return

    tried = ", ".join(p for p in candidates if p) or "(none)"
    raise ImportError(
        "Webots Python API not found (expected a folder containing controller.py). "
        f"Tried: {tried}. Install Webots or set WEBOTS_HOME to the install root "
        '(on macOS use e.g. WEBOTS_HOME=/Applications/Webots.app, not ".../Contents").'
    )


_prepend_webots_controller_python_path()

from controller import Supervisor
import numpy as np
from stable_baselines3 import SAC

MODEL_PATH = "following_red_ball_model_phase_1_3"

TIME_STEP = 64
# IMPORTANT: must match FRAME_SKIP in the training env. The training env runs
# `for _ in range(FRAME_SKIP): robot.step(TIME_STEP)` per agent action, so the
# policy was trained on a 5x slower control loop (~320 ms per decision). Without
# this here, inference runs at 5x the bandwidth the policy expects and the
# integrated motion (especially in-place rotation) does not match training.
FRAME_SKIP = 5
MAX_SPEED = 6.279
WHEEL_DISTANCE = 0.052
# When True, sample from the policy distribution instead of taking the mean
# action. Useful while the policy is still under-trained on rare states (e.g.
# wedged-blind in a corner): the mean action there can be a stuck attractor
# because that state was not seen often enough during training. Stochastic
# sampling matches what was used during training (ent_coef > 0), at the cost
# of less reproducible runs.
STOCHASTIC_INFERENCE = True
# Print one line per agent decision with the action and key sensor state so we
# can diagnose stuck behavior (is the policy commanding a spin? are the wheels
# clipping it away? is `lin` saturated forward?). Set to a larger N to print
# less often, or 0 to disable.
DEBUG_PRINT_EVERY_N_STEPS = 8  # ~2.5 s at 320 ms/step

robot = Supervisor()
robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

camera = robot.getDevice('camera')
camera.enable(TIME_STEP)
CAMERA_WIDTH = camera.getWidth()
CAMERA_HEIGHT = camera.getHeight()
TOTAL_PIXELS = CAMERA_WIDTH * CAMERA_HEIGHT

proximity_sensors = []
for sensor_name in ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']:
    sensor = robot.getDevice(sensor_name)
    sensor.enable(TIME_STEP)
    proximity_sensors.append(sensor)

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


def analyze_camera_for_red_ball():
    raw_image = camera.getImage()
    if not raw_image:
        return 0.0, 0.0

    image = np.frombuffer(raw_image, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    red_pixel_mask = (red_channel > 130) & (green_channel < 80) & (blue_channel < 80)
    red_pixel_count = np.count_nonzero(red_pixel_mask)
    red_pixel_ratio = red_pixel_count / TOTAL_PIXELS

    if red_pixel_count > 0:
        red_pixel_columns = np.where(red_pixel_mask)[1]
        average_x = np.mean(red_pixel_columns)
        goal_horizontal_position = (average_x / (CAMERA_WIDTH - 1)) * 2.0 - 1.0
    else:
        goal_horizontal_position = 0.0

    return red_pixel_ratio, goal_horizontal_position


def read_proximity_sensors():
    raw_values = np.array([sensor.getValue() for sensor in proximity_sensors], dtype=np.float32)
    return np.clip(raw_values / 4095.0, 0.0, 1.0)


model = SAC.load(MODEL_PATH)

linear_velocity = 0.0
angular_velocity = 0.0
agent_step = 0

# Prime the simulation so sensors return real values before the first inference.
# Without this, `sensor.getValue()` returns NaN for sensors that were enabled but
# never stepped, the observation becomes NaN, and SAC's policy outputs NaN actions
# (Normal(loc=NaN, scale=NaN) -> ValueError).
if robot.step(TIME_STEP) == -1:
    raise SystemExit(0)

while True:
    red_pixel_ratio, goal_horizontal_position = analyze_camera_for_red_ball()
    proximity_values = read_proximity_sensors()
    # Defensive: if any sensor still has NaN/Inf (e.g. transient camera-frame issue),
    # treat the slot as zero rather than feeding NaN into the policy.
    proximity_values = np.nan_to_num(proximity_values, nan=0.0, posinf=1.0, neginf=0.0)
    observation = np.concatenate([
        np.array(
            [red_pixel_ratio, goal_horizontal_position, linear_velocity, angular_velocity],
            dtype=np.float32,
        ),
        proximity_values,
    ])

    action, _ = model.predict(observation, deterministic=not STOCHASTIC_INFERENCE)
    linear_velocity = float(action[0])
    angular_velocity = float(action[1])

    # Match the training kinematics exactly: same conversion AND same wheel clip
    # ([-MAX_SPEED, MAX_SPEED]). This avoids a small but real distribution
    # shift versus what the policy saw during training.
    left_pre = linear_velocity - angular_velocity * WHEEL_DISTANCE / 2.0
    right_pre = linear_velocity + angular_velocity * WHEEL_DISTANCE / 2.0
    left_wheel_speed = float(np.clip(left_pre, -MAX_SPEED, MAX_SPEED))
    right_wheel_speed = float(np.clip(right_pre, -MAX_SPEED, MAX_SPEED))
    left_motor.setVelocity(left_wheel_speed)
    right_motor.setVelocity(right_wheel_speed)

    if DEBUG_PRINT_EVERY_N_STEPS > 0 and agent_step % DEBUG_PRINT_EVERY_N_STEPS == 0:
        prox_max = float(np.max(proximity_values))
        clipped = (left_pre < 0) or (right_pre < 0)
        print(
            f"[step {agent_step:4d}] lin={linear_velocity:+.2f} ang={angular_velocity:+.2f}"
            f" | wheels (L,R)=({left_wheel_speed:+.2f},{right_wheel_speed:+.2f})"
            f"{'  CLIPPED' if clipped else ''}"
            f" | red_px={red_pixel_ratio:.3f} goal_pos={goal_horizontal_position:+.2f}"
            f" prox_max={prox_max:.2f}"
        )
    agent_step += 1

    # Hold the action across FRAME_SKIP physics steps to match training.
    for _ in range(FRAME_SKIP):
        if robot.step(TIME_STEP) == -1:
            raise SystemExit(0)
