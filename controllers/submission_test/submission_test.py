"""
Local Webots driver that exercises the submission's `Agent` class through the
*exact* contract the competition evaluator uses:

    * obs is the competition 11-D vector:
        [ps0..ps7, red_detected, centroid_x_norm, red_pixel_ratio]
    * image is (H, W, 3) uint8 RGB.
    * action is (2,) float32 in [-1, 1] -- left/right wheel speed fractions.

This lets us validate that submission/marcio_gabriel/agent.py works end-to-end
*before* zipping and submitting. We deliberately do NOT bypass the adapter
inside agent.py (unlike phase_*_trained_controller.py, which talks to the SAC
policy directly with the project's internal 12-D obs).

Point the robot's `controller` field in the .wbt to "submission_test" (or run
it as `<extern>`) and press Play.
"""

import os
import pathlib
import platform
import sys


def _prepend_webots_controller_python_path() -> None:
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
    raise ImportError(
        "Webots Python API not found; install Webots or set WEBOTS_HOME "
        "(on macOS use e.g. WEBOTS_HOME=/Applications/Webots.app)."
    )


_prepend_webots_controller_python_path()


# --- Path bootstrap so we can import both the submission and the kit ---
# `epuck_agent.py` and `camera_utils.py` come from the competition kit
# (provided by the evaluator at submission time). For local testing we expect
# them in /Users/.../Downloads/Arquivo/.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SUBMISSION_DIR = PROJECT_ROOT / "submission" / "marcio_gabriel"
KIT_DIR = pathlib.Path.home() / "Downloads" / "Arquivo"

sys.path.insert(0, str(SUBMISSION_DIR))
sys.path.insert(0, str(KIT_DIR))


from controller import Robot  # noqa: E402

import numpy as np  # noqa: E402

from agent import Agent  # noqa: E402  (from submission/marcio_gabriel/agent.py)


# --- Tunables (must match the competition runner & the e-puck) ---
TIME_STEP = 64           # simulation step in ms
MAX_SPEED = 6.28         # epuck_agent.EpuckAgent.MAX_SPEED
PS_MAX = 4096.0          # epuck_agent.EpuckAgent.PS_MAX (competition value)
RED_PIXEL_R_MIN = 100    # camera_utils.is_red thresholds
RED_PIXEL_DELTA = 50

# How many physics steps to hold each action before re-querying the agent.
# Set to 5 to match our training FRAME_SKIP; 1 to mirror a per-step runner.
HOLD_STEPS = 1


def is_red(r: int, g: int, b: int) -> bool:
    """Same definition as Arquivo/camera_utils.py."""
    return r > RED_PIXEL_R_MIN and (r - g) > RED_PIXEL_DELTA and (r - b) > RED_PIXEL_DELTA


def camera_features_rgb(image_rgb: np.ndarray) -> tuple[float, float, float]:
    """
    Vectorised reimplementation of Arquivo/camera_utils.camera_features.
    Returns (red_detected, centroid_x_norm in [-1,1], red_pixel_ratio in [0,1]).
    """
    h, w, _ = image_rgb.shape
    r = image_rgb[:, :, 0].astype(np.int32)
    g = image_rgb[:, :, 1].astype(np.int32)
    b = image_rgb[:, :, 2].astype(np.int32)
    mask = (r > RED_PIXEL_R_MIN) & ((r - g) > RED_PIXEL_DELTA) & ((r - b) > RED_PIXEL_DELTA)
    count = int(mask.sum())
    if count == 0:
        return 0.0, 0.0, 0.0
    cx_sum = float(np.where(mask)[1].sum())
    cx_norm = (cx_sum / count - w / 2) / (w / 2)
    red_ratio = count / (w * h)
    return 1.0, float(cx_norm), float(red_ratio)


def bgra_to_rgb(raw: bytes, width: int, height: int) -> np.ndarray:
    """Webots returns BGRA bytes; the competition image kwarg is (H,W,3) RGB."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    return arr[:, :, [2, 1, 0]].copy()


# --- Webots device setup ---
robot = Robot()

camera = robot.getDevice("camera")
camera.enable(TIME_STEP)
CAMERA_WIDTH = camera.getWidth()
CAMERA_HEIGHT = camera.getHeight()

proximity_sensors = []
for sensor_name in ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]:
    sensor = robot.getDevice(sensor_name)
    sensor.enable(TIME_STEP)
    proximity_sensors.append(sensor)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


# --- Load the submission agent ---
print(f"Loading submission agent from {SUBMISSION_DIR}")
agent = Agent()
agent.load(str(SUBMISSION_DIR / "model"))
print("Agent loaded.")


# --- Prime the simulation so sensors return real (non-NaN) values ---
if robot.step(TIME_STEP) == -1:
    raise SystemExit(0)


def build_observation() -> tuple[np.ndarray, np.ndarray]:
    """Build the competition 11-D observation and the (H,W,3) RGB image."""
    raw_values = np.array([s.getValue() for s in proximity_sensors], dtype=np.float32)
    proximity = np.clip(raw_values / PS_MAX, 0.0, 1.0)

    raw_image = camera.getImage()
    if raw_image:
        image_rgb = bgra_to_rgb(raw_image, CAMERA_WIDTH, CAMERA_HEIGHT)
        red_detected, centroid_x_norm, red_pixel_ratio = camera_features_rgb(image_rgb)
    else:
        image_rgb = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        red_detected, centroid_x_norm, red_pixel_ratio = 0.0, 0.0, 0.0

    obs = np.empty(11, dtype=np.float32)
    obs[0:8] = proximity
    obs[8] = red_detected
    obs[9] = centroid_x_norm
    obs[10] = red_pixel_ratio
    return obs, image_rgb


tick = 0
PRINT_EVERY = 8

while True:
    obs, image_rgb = build_observation()
    action = agent.act(obs, image=image_rgb)

    # Competition runner: action is (left_frac, right_frac) in [-1, 1].
    left_frac = float(np.clip(action[0], -1.0, 1.0))
    right_frac = float(np.clip(action[1], -1.0, 1.0))
    left_motor.setVelocity(left_frac * MAX_SPEED)
    right_motor.setVelocity(right_frac * MAX_SPEED)

    if tick % PRINT_EVERY == 0:
        print(
            f"[tick {tick:5d}] "
            f"action=({left_frac:+.2f},{right_frac:+.2f}) "
            f"red={obs[10]:.3f} centroid={obs[9]:+.2f} "
            f"prox_max={float(np.max(obs[0:8])):.2f}"
        )
    tick += 1

    for _ in range(HOLD_STEPS):
        if robot.step(TIME_STEP) == -1:
            raise SystemExit(0)
