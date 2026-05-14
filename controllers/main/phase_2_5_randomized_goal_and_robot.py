"""
Webots controller — Phase 2.5: randomized robot AND goal positions.

Curriculum step that follows phase 1 (fixed start, empty arena) and phase 2
(fixed start + obstacles). Here the policy from phase 2 keeps training in the
same arena, but on every episode reset both the robot pose and the red ball
position are sampled uniformly inside the arena (with a minimum separation),
forcing the agent to generalize beyond the single fixed approach trajectory it
saw during phases 1 & 2.

Observation / action / reward functions are identical to phase 2 so the
existing weights load without shape mismatch.

Observation space:
    [red_pixel_ratio, goal_horizontal_position, linear_velocity, angular_velocity,
     ps0, ps1, ps2, ps3, ps4, ps5, ps6, ps7]

Action space:
    [linear_velocity, angular_velocity]

Reward:
    - Visibility / approach (red pixel ratio + delta)
    - Centering bonus (delta-based)
    - Loss-of-sight penalty
    - Step penalty
    - Smooth wall avoidance gradient (when ball is not dominant)
    - Wall crash penalty
    - Large bonus for physically reaching the goal
"""
import os
import platform
import sys


def _prepend_webots_controller_python_path():
    """So `import controller` works when the script is not started by Webots."""
    sysname = platform.system()

    # controller/wb.py loads libController using WEBOTS_HOME; on macOS the dylib
    # path is join(WEBOTS_HOME, "Contents/lib/controller/libController.dylib"),
    # so WEBOTS_HOME must be the .app bundle root (e.g. /Applications/Webots.app).
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

from controller import Supervisor, Motor, Camera, DistanceSensor
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

# --- Simulation constants ---
TIME_STEP = 64                   # simulation step in milliseconds
MAX_SPEED = 6.279                # max angular velocity of e-puck motors [rad/s]

# --- Episode constants ---
MAX_STEPS_PER_EPISODE = 10000
PROXIMITY_GOAL_THRESHOLD = 0.95  # normalized proximity sensor value (0.0 - 1.0)
STEPS_WITHOUT_BALL_LIMIT = 6000
FRAME_SKIP = 5

# --- Arena & randomization bounds (world is Z-up, floor is X-Y plane) ---
# RectangleArena floorSize is 2.5 x 1.5  ->  X in [-1.25, 1.25], Y in [-0.75, 0.75].
# A safety margin keeps the e-puck (radius ~0.037 m) and the ball (radius 0.06 m)
# clear of the surrounding walls.
ARENA_X_RANGE = (-1.05, 1.05)
ARENA_Y_RANGE = (-0.55, 0.55)

# Heights are kept fixed at the same values used in sample_lab.wbt
ROBOT_Z = 0.0
GOAL_Z  = 0.06

# Minimum spawn-time distance between robot center and ball center.
# Below this the camera already sees a huge red blob at episode start, which
# trivializes the task and washes out the approach reward signal.
MIN_ROBOT_GOAL_DISTANCE = 0.7

# Static obstacles (X, Y, conservative radius) that random spawn positions must
# avoid. The four wooden boxes in sample_lab.wbt are placed BEHIND the arena
# walls, so this list is empty in phase 2.5; phase 3 can extend it.
STATIC_OBSTACLES = []  # e.g. [(-0.30, 0.00, 0.15)]

# Maximum rejection-sampling attempts before falling back to a guaranteed-safe
# but less-random pose. With the bounds above this never triggers in practice.
MAX_RANDOM_TRIES = 50

# Fallback poses (used on the very first reset before any sampling, and as the
# guaranteed-safe fallback if rejection sampling somehow fails).
ROBOT_START_POSITION = [-1.0, 0.0, ROBOT_Z]
ROBOT_START_ROTATION = [0, 0, 1, 0]
GOAL_START_POSITION  = [ 1.0, 0.0, GOAL_Z]

# --- Supervisor initialization ---
robot = Supervisor()
robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

# --- Proximity sensors ---
proximity_sensors = []
PROXIMITY_SENSOR_NAMES = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
for i in range(8):
    sensor = robot.getDevice(PROXIMITY_SENSOR_NAMES[i])
    sensor.enable(TIME_STEP)
    proximity_sensors.append(sensor)

# --- Camera ---
camera = robot.getDevice('camera')
camera.enable(TIME_STEP)
CAMERA_WIDTH = camera.getWidth()
CAMERA_HEIGHT = camera.getHeight()
TOTAL_PIXELS = CAMERA_WIDTH * CAMERA_HEIGHT

# --- Motors ---
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# --- Supervisor references for position reset ---
robot_node = robot.getFromDef('ROBOT')
robot_translation_field = robot_node.getField('translation')
robot_rotation_field = robot_node.getField('rotation')

goal_node = robot.getFromDef('GOAL')
# Phase 2.5 also moves the goal between episodes, which requires a writable
# handle to its translation field.
goal_translation_field = goal_node.getField('translation')

# --- Robot dimensions ---
WHEEL_DISTANCE = 0.052  # [meters]


def convert_velocities_to_motor_speeds(linear_velocity, angular_velocity):
    """Differential-drive kinematics -> (left_wheel, right_wheel) speeds."""
    left_speed = linear_velocity - angular_velocity * WHEEL_DISTANCE / 2.0
    right_speed = linear_velocity + angular_velocity * WHEEL_DISTANCE / 2.0
    left_speed = np.clip(left_speed, -MAX_SPEED, MAX_SPEED)
    right_speed = np.clip(right_speed, -MAX_SPEED, MAX_SPEED)
    return left_speed, right_speed


def analyze_camera_for_red_ball():
    """Return (red_pixel_ratio, goal_horizontal_position) from the camera image."""
    try:
        raw_image = camera.getImage()
    except ValueError:
        return 0.0, 0.0
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


def get_front_proximity():
    """Mean of ps0 + ps7, normalized to [0.0, 1.0]."""
    front_value = (proximity_sensors[0].getValue() + proximity_sensors[7].getValue()) / 2.0
    return np.clip(front_value / 4095.0, 0.0, 1.0)


def get_all_proximities():
    """Return all 8 proximity sensors normalized to [0.0, 1.0]."""
    values = np.array([s.getValue() for s in proximity_sensors], dtype=np.float32)
    return np.clip(values / 4095.0, 0.0, 1.0)


def _is_far_enough_from_obstacles(x, y, extra_margin=0.0):
    """True if (x, y) is at least `obstacle_radius + extra_margin` from every static obstacle."""
    for ox, oy, oradius in STATIC_OBSTACLES:
        if (x - ox) ** 2 + (y - oy) ** 2 < (oradius + extra_margin) ** 2:
            return False
    return True


def sample_random_episode_layout(np_random):
    """
    Sample a fresh (robot_position, robot_rotation, goal_position) for an episode.

    Both poses are drawn uniformly inside the safe arena bounds. We re-sample
    until the robot and the goal are at least MIN_ROBOT_GOAL_DISTANCE apart and
    neither overlaps any static obstacle. After MAX_RANDOM_TRIES we fall back to
    the constant start poses so the episode can still proceed.

    The robot heading is drawn uniformly in [-pi, pi] around the world up axis
    (Z, matching the world's ENU orientation), so the policy must learn to find
    the ball regardless of initial heading.
    """
    # 1) Goal position.
    goal_x, goal_y = ROBOT_START_POSITION[0] + 1.5, 0.0  # placeholder, overwritten below
    for _ in range(MAX_RANDOM_TRIES):
        gx = float(np_random.uniform(*ARENA_X_RANGE))
        gy = float(np_random.uniform(*ARENA_Y_RANGE))
        if _is_far_enough_from_obstacles(gx, gy, extra_margin=0.10):
            goal_x, goal_y = gx, gy
            break
    else:
        goal_x, goal_y = GOAL_START_POSITION[0], GOAL_START_POSITION[1]

    # 2) Robot position — must be far from goal and from obstacles.
    robot_x, robot_y = ROBOT_START_POSITION[0], ROBOT_START_POSITION[1]
    for _ in range(MAX_RANDOM_TRIES):
        rx = float(np_random.uniform(*ARENA_X_RANGE))
        ry = float(np_random.uniform(*ARENA_Y_RANGE))
        if (rx - goal_x) ** 2 + (ry - goal_y) ** 2 < MIN_ROBOT_GOAL_DISTANCE ** 2:
            continue
        if not _is_far_enough_from_obstacles(rx, ry, extra_margin=0.05):
            continue
        robot_x, robot_y = rx, ry
        break

    # 3) Heading — uniform over the full circle, around the world up axis (Z).
    yaw = float(np_random.uniform(-np.pi, np.pi))

    robot_position = [robot_x, robot_y, ROBOT_Z]
    robot_rotation = [0.0, 0.0, 1.0, yaw]
    goal_position  = [goal_x, goal_y, GOAL_Z]
    return robot_position, robot_rotation, goal_position


class RoombaRedBallEnv(gym.Env):
    """
    Same Gymnasium environment as phase 2, but with randomized robot and goal
    positions on every reset.
    """

    def __init__(self, render_mode=None):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([-MAX_SPEED, -MAX_SPEED], dtype=np.float32),
            high=np.array([MAX_SPEED, MAX_SPEED], dtype=np.float32),
        )

        # Same 12-D observation as phase 2 — keeps the loaded weights compatible.
        #   [0]    red_pixel_ratio          0.0 .. 1.0
        #   [1]    goal_horizontal_position -1.0 .. 1.0
        #   [2]    linear_velocity          0.0 .. MAX_SPEED
        #   [3]    angular_velocity         -MAX_SPEED .. MAX_SPEED
        #   [4-11] proximity ps0..ps7       0.0 .. 1.0
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, -1.0, -MAX_SPEED, -MAX_SPEED] + [0.0] * 8,
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, MAX_SPEED, MAX_SPEED] + [1.0] * 8,
                dtype=np.float32,
            ),
        )

        self.render_mode = render_mode

        # --- Environment state ---
        self.current_red_pixel_ratio = 0.0
        self.previous_red_pixel_ratio = 0.0
        self.current_goal_position = 0.0
        self.previous_goal_position = 0.0
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.current_front_proximity = 0.0
        self.current_proximities = np.zeros(8, dtype=np.float32)

        # --- Counters ---
        self.step_count = 0
        self.steps_without_red_ball = 0
        self.episode_count = 0
        self.episode_total_reward = 0.0
        self.steps_since_ball_in_front = 999
        self.steps_since_ball_visible = 999
        self.steps_near_ball = 0
        self.steps_touching_wall = 0

        # Last sampled spawn poses, kept so per-step logging can show the
        # current goal/robot setup without an extra supervisor call.
        self.current_spawn_robot = list(ROBOT_START_POSITION)
        self.current_spawn_goal  = list(GOAL_START_POSITION)

    def _build_observation(self):
        """Build the 12-D observation vector from current environment state."""
        return np.concatenate([
            np.array([
                self.current_red_pixel_ratio,
                self.current_goal_position,
                self.current_linear_velocity,
                self.current_angular_velocity,
            ], dtype=np.float32),
            self.current_proximities,
        ])

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Phase 2.5 difference: in addition to teleporting the robot, the goal
        (red ball) is also teleported, and both the robot's (x, y, yaw) and
        the goal's (x, y) are sampled uniformly inside the arena under a
        minimum-separation constraint.
        """
        super().reset(seed=seed)

        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        if self.step_count > 0:
            self.episode_count += 1
            print(
                f"\n=== EPISODE {self.episode_count} DONE"
                f" | steps: {self.step_count}"
                f" | total reward: {self.episode_total_reward:.1f} ===\n"
            )

        # Sample a fresh layout for this episode.
        robot_position, robot_rotation, goal_position = sample_random_episode_layout(self.np_random)
        self.current_spawn_robot = robot_position
        self.current_spawn_goal  = goal_position

        robot_translation_field.setSFVec3f(robot_position)
        robot_rotation_field.setSFRotation(robot_rotation)
        goal_translation_field.setSFVec3f(goal_position)

        # Zero out residual physics so neither object inherits velocity from
        # the previous episode.
        robot_node.resetPhysics()
        goal_node.resetPhysics()

        print(
            f"--- Episode {self.episode_count + 1} layout"
            f" | robot ({robot_position[0]:+.2f},{robot_position[1]:+.2f}) yaw={robot_rotation[3]:+.2f}"
            f" | goal ({goal_position[0]:+.2f},{goal_position[1]:+.2f}) ---"
        )

        # One simulation step so the new poses take effect before the policy
        # observes the scene.
        robot.step(TIME_STEP)

        self.current_red_pixel_ratio = 0.0
        self.previous_red_pixel_ratio = 0.0
        self.current_goal_position = 0.0
        self.previous_goal_position = 0.0
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.current_front_proximity = 0.0
        self.current_proximities = np.zeros(8, dtype=np.float32)
        self.step_count = 0
        self.steps_without_red_ball = 0
        self.episode_total_reward = 0.0
        self.steps_since_ball_in_front = 999
        self.steps_since_ball_visible = 999
        self.steps_near_ball = 0
        self.steps_touching_wall = 0

        return self._build_observation(), {}

    def step(self, action):
        linear_velocity = float(action[0])
        angular_velocity = float(action[1])

        left_speed, right_speed = convert_velocities_to_motor_speeds(linear_velocity, angular_velocity)
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        accumulated_reward = 0.0
        terminated = False

        for _ in range(FRAME_SKIP):
            robot.step(TIME_STEP)

            self.current_linear_velocity = linear_velocity
            self.current_angular_velocity = angular_velocity

            self.previous_red_pixel_ratio = self.current_red_pixel_ratio
            self.previous_goal_position = self.current_goal_position
            self.current_red_pixel_ratio, self.current_goal_position = analyze_camera_for_red_ball()
            self.current_front_proximity = get_front_proximity()
            self.current_proximities = get_all_proximities()
            self.step_count += 1

            if self.current_red_pixel_ratio == 0.0:
                self.steps_without_red_ball += 1
            else:
                self.steps_without_red_ball = 0

            reward = 0.0
            reward -= 0.05
            reward += self.current_red_pixel_ratio * 3.0

            pixel_ratio_change = self.current_red_pixel_ratio - self.previous_red_pixel_ratio
            reward += pixel_ratio_change * 10.0

            ball_is_in_front = (
                self.current_red_pixel_ratio > 0.1
                and abs(self.current_goal_position) < 0.3
            )
            if ball_is_in_front:
                self.steps_since_ball_in_front = 0
            else:
                self.steps_since_ball_in_front += 1

            if self.current_red_pixel_ratio > 0.01:
                self.steps_since_ball_visible = 0
            else:
                self.steps_since_ball_visible += 1

            ball_recently_visible = self.steps_since_ball_visible <= 30

            if self.current_red_pixel_ratio > 0.0:
                centering_improvement = abs(self.previous_goal_position) - abs(self.current_goal_position)
                reward += centering_improvement * 2.0
            elif not ball_recently_visible:
                reward -= 0.5

            making_progress = (
                pixel_ratio_change > 0.001
                or (abs(self.previous_goal_position) - abs(self.current_goal_position)) > 0.01
            )
            if self.current_red_pixel_ratio > 0.05 and not making_progress:
                self.steps_near_ball += 1
            elif making_progress:
                self.steps_near_ball = 0

            sensor_max = np.max(self.current_proximities)
            in_contact = sensor_max > 0.9
            touching_ball = (
                in_contact
                and self.current_red_pixel_ratio >= 0.10
                and ball_recently_visible
            )
            touching_wall = in_contact and not touching_ball

            if self.current_red_pixel_ratio < 0.05:
                reward -= sensor_max * 0.3

            if touching_wall:
                reward -= 2.0
                self.steps_touching_wall += 1
            else:
                self.steps_touching_wall = 0

            if touching_ball:
                reward += 1000.0
                print(f"*** GOAL REACHED at step {self.step_count}! proximity={self.current_front_proximity:.3f} ***")
                terminated = True

            accumulated_reward += reward

            timed_out   = self.step_count >= MAX_STEPS_PER_EPISODE
            lost_ball   = self.steps_without_red_ball >= STEPS_WITHOUT_BALL_LIMIT
            stuck_orbit = self.steps_near_ball >= 500
            stuck_wall  = self.steps_touching_wall >= 50
            if terminated or timed_out or lost_ball or stuck_orbit or stuck_wall:
                break

        timed_out   = self.step_count >= MAX_STEPS_PER_EPISODE
        lost_ball   = self.steps_without_red_ball >= STEPS_WITHOUT_BALL_LIMIT
        stuck_orbit = self.steps_near_ball >= 500
        stuck_wall  = self.steps_touching_wall >= 50
        truncated   = timed_out or lost_ball or stuck_orbit or stuck_wall

        self.episode_total_reward += accumulated_reward
        observation = self._build_observation()

        if self.step_count % 50 == 0:
            rp = robot_translation_field.getSFVec3f()
            gp = goal_translation_field.getSFVec3f()
            # Floor distance is in the world's X-Y plane (Z is up in ENU).
            dist = np.sqrt((rp[0] - gp[0]) ** 2 + (rp[1] - gp[1]) ** 2)
            ps = self.current_proximities
            print(
                f"  step {self.step_count:4d}"
                f" | robot ({rp[0]:+.2f},{rp[1]:+.2f})"
                f" | ball ({gp[0]:+.2f},{gp[1]:+.2f})"
                f" | dist {dist:.3f}"
                f" | red {self.current_red_pixel_ratio:.3f}"
                f" | cam_x {self.current_goal_position:+.2f}"
                f" | prox_max {np.max(ps):.3f}"
                f" | ps [{ps[0]:.2f},{ps[1]:.2f},{ps[2]:.2f},{ps[3]:.2f},{ps[4]:.2f},{ps[5]:.2f},{ps[6]:.2f},{ps[7]:.2f}]"
                f" | near_ball {self.steps_near_ball}"
                f" | wall {self.steps_touching_wall}"
                f" | ep_rew {self.episode_total_reward:.1f}"
            )

        return observation, accumulated_reward, terminated, truncated, {}

    def close(self):
        """Clean up resources (OpenCV windows)."""
        cv2.destroyAllWindows()


# =============================================
# TRAINING
# =============================================
environment = RoombaRedBallEnv()

from stable_baselines3 import SAC


def run_training(
    resume_path=None,
    total_timesteps=300_000,
    output_path="randomized_following_red_ball_model",
    output_replay_buffer_path="randomized_following_red_ball_replay_buffer.pkl",
    phase2_preferred_path="obstacle_avoidance_following_red_ball_model.zip",
    fallback_model_path="following_red_ball_model.zip",
    obstacle_replay_buffer_path="obstacle_avoidance_following_red_ball_replay_buffer.pkl",
    load_replay_buffer=None,
    device=None,
):
    """
    Phase 2.5: load phase-2 checkpoint by default; fallback to phase 1 if missing.
    load_replay_buffer: if True load obstacle replay; if None, use env LOAD_REPLAY_BUFFER==1.
    """
    import os
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("SAC device:", device)

    def _resolve(path):
        if os.path.exists(path):
            return path
        base = path[:-4] if path.endswith(".zip") else path
        z = base + ".zip"
        if os.path.exists(z):
            return z
        return None

    if resume_path:
        chosen_model_path = _resolve(resume_path)
        if not chosen_model_path:
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    elif _resolve(phase2_preferred_path):
        chosen_model_path = _resolve(phase2_preferred_path)
    elif _resolve(fallback_model_path):
        print(
            f"WARNING: {phase2_preferred_path} not found, falling back to {fallback_model_path}. "
            "Phase 2.5 is meant to run on top of phase 2."
        )
        chosen_model_path = _resolve(fallback_model_path)
    else:
        raise FileNotFoundError(
            "No pretrained model found (neither phase-2 nor phase-1 checkpoint). "
            "Pass --checkpoint or train earlier phases."
        )

    print(f"Loading pretrained model from {chosen_model_path}")
    model = SAC.load(
        chosen_model_path,
        env=environment,
        verbose=1,
        device=device,
        learning_starts=1000,
        custom_objects={
            "learning_rate": 5e-5,
            "batch_size": 256,
            "gamma": 0.99,
        },
    )

    if load_replay_buffer is None:
        load_replay_buffer = os.environ.get("LOAD_REPLAY_BUFFER", "0") == "1"
    if load_replay_buffer and os.path.exists(obstacle_replay_buffer_path):
        print(f"Loading replay buffer from {obstacle_replay_buffer_path}")
        model.load_replay_buffer(obstacle_replay_buffer_path)

    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    model.save(output_path)
    model.save_replay_buffer(output_replay_buffer_path)
    print(f"Model saved to {output_path}.zip")
    return model


if __name__ == "__main__":
    run_training()
