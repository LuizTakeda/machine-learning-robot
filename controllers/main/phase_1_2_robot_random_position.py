"""
Webots controller — Phase 1.2: fixed goal (red ball) in the world as in phase 1,
but every episode the robot is placed at a random (x, y) inside the arena with a
random yaw. The policy must search, approach, and touch the ball from arbitrary
poses. Training continues from the phase 1.1 checkpoint (`following_red_ball_model_phase_1_1`).
Includes wall shaping, linear/prox penalties, blind-wall truncation (steps_blind_near_wall),
two-tier wedged spin bonus, blind clearance reward when max IR drops, and orbit/stuck-wall truncation.

Goal world (x, y) is read once from DEF GOAL in the .wbt at startup for spawn
distance checks; the ball is not moved between episodes.

Observation / action / reward: same vector as phase_1_initial_training.py plus the above rewards.
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
MAX_STEPS_PER_EPISODE = 40000     # max steps per episode
# Much higher than phase 1: the ball is often out of frame until the robot turns.
STEPS_WITHOUT_BALL_LIMIT = 40000   # steps without seeing the ball before truncation
FRAME_SKIP = 5

# --- Wall / corner escape (aligned with phase_2_training_with_obstacle.py) ---
WALL_CONTACT_SENSOR_MAX = 0.9
STUCK_WALL_STEP_LIMIT = 50
STEPS_NEAR_BALL_STALL_LIMIT = 500  # phase 2: truncate if orbiting / stalled near ball without progress
RED_SMALL_FOR_WALL_GRADIENT = 0.05
WALL_PROX_GRADIENT_SCALE = 0.3
WALL_CRASH_PENALTY = 2.0
TOUCHING_BALL_RED_MIN = 0.10
SPIN_WHEN_WEDGED_K = 0.12
WEDGED_SENSOR_MAX = 0.35
WEDGED_RED_MAX = 0.05
SPIN_BONUS_MAX_LINEAR_FRAC = 0.35
# Penalize driving forward hard while any side sees a wall and we are not on the ball.
LINEAR_WALL_PROX_MIN = 0.25
LINEAR_WALL_COEFF = 0.55
# Tiered wall gradient: small red no longer turns off all wall signal near corners.
WALL_GRADIENT_RED_EXTENDED = 0.12
WALL_EXTENDED_PROX_MIN = 0.35
WALL_EXTENDED_GRADIENT_FRACTION = 0.5
# When wedged (high prox, low red), penalize high linear speed (stop ramming).
WEDGED_LINEAR_THRESHOLD = 0.2
WEDGED_LINEAR_PENALTY_K = 0.35
# Penalize rapid increase in front proximity when we do not clearly see the ball.
FRONT_RISE_RED_MAX = 0.08
FRONT_RISE_COEFF = 8.0
# Blind + mid-range wall (never reaches sensor_max>0.9): recovery / truncation
BLIND_RED_TH = 0.03
MID_PROX_LO = 0.28
BLIND_WALL_STUCK_LIMIT = 300
# Second spin tier when linear is high (fraction of SPIN_WHEN_WEDGED_K)
SPIN_WHEN_WEDGED_HIGH_LINEAR_FRAC = 0.45
# Reward lowering max IR while blind (clamped delta)
BLIND_CLEARANCE_BONUS = 5.0
BLIND_CLEARANCE_MAX_DELTA = 0.08

# --- Arena & random robot pose (world Z-up; floor is X–Y). Goal stays fixed in .wbt. ---
# RectangleArena-style bounds with margin (see phase_2_5_randomized_goal_and_robot.py).
ARENA_X_RANGE = (-1.05, 1.05)
ARENA_Y_RANGE = (-0.55, 0.55)
ROBOT_Z = 0.0
# Keep spawns away from the ball so the task does not start as a trivial close-up.
MIN_ROBOT_GOAL_DISTANCE = 0.7
STATIC_OBSTACLES = []  # extend for worlds with boxes inside the arena
MAX_RANDOM_TRIES = 50
ROBOT_YAW_AXIS = [0.0, 0.0, 1.0]
# If rejection sampling fails, use a phase-1-like corner start (still valid with random yaw).
ROBOT_FALLBACK_POSITION = [-1.1, 0.019, ROBOT_Z]

# --- Supervisor initialization ---
# Supervisor inherits from Robot but can also modify scene objects
# (needed to reset the robot position between episodes)
robot = Supervisor()

# Fast simulation mode - skips rendering,
# runs as fast as the CPU allows (typically 5-20x faster)
robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

# --- Proximity sensors ---
# e-puck has 8 infrared sensors around its body
# ps0-ps7: 0 = nothing detected, 4095 = max reflection
# ps0 and ps7 point forward-right and forward-left respectively,
# ps0 alone is a reasonable "front sensor" for detecting the ball ahead
proximity_sensors = []
PROXIMITY_SENSOR_NAMES = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
for i in range(8):
    sensor = robot.getDevice(PROXIMITY_SENSOR_NAMES[i])
    sensor.enable(TIME_STEP)
    proximity_sensors.append(sensor)

# --- Camera ---
# Camera resolution is set via camera_width and camera_height fields in Webots
camera = robot.getDevice('camera')
camera.enable(TIME_STEP)
CAMERA_WIDTH = camera.getWidth()
CAMERA_HEIGHT = camera.getHeight()
TOTAL_PIXELS = CAMERA_WIDTH * CAMERA_HEIGHT  # total pixel count for normalization

# --- Motors ---
# setPosition(inf) = continuous rotation (not angle positioning)
# setVelocity(0) = start stationary
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# --- Supervisor references for position reset ---
# getFromDef('ROBOT') finds the node with DEF ROBOT in the .wbt file
# getField() gets a reference to a specific field that we can modify
robot_node = robot.getFromDef('ROBOT')
robot_translation_field = robot_node.getField('translation')
robot_rotation_field = robot_node.getField('rotation')

goal_node = robot.getFromDef('GOAL')
goal_translation_field = goal_node.getField('translation')
_g0 = goal_translation_field.getSFVec3f()
GOAL_PLANE_X = float(_g0[0])
GOAL_PLANE_Y = float(_g0[1])

# --- Robot dimensions ---
# Distance between e-puck wheels - needed for converting
# linear/angular velocity to individual wheel speeds
WHEEL_DISTANCE = 0.052  # [meters]


def _is_far_enough_from_obstacles(x, y, extra_margin=0.0):
    """True if (x, y) clears every static obstacle disk by extra_margin."""
    for ox, oy, oradius in STATIC_OBSTACLES:
        if (x - ox) ** 2 + (y - oy) ** 2 < (oradius + extra_margin) ** 2:
            return False
    return True


def sample_random_robot_pose(np_random):
    """
    Uniform position in the arena, uniform yaw on Z, at least MIN_ROBOT_GOAL_DISTANCE
    from the fixed goal. Falls back to ROBOT_FALLBACK_POSITION if sampling fails.
    """
    for _ in range(MAX_RANDOM_TRIES):
        rx = float(np_random.uniform(*ARENA_X_RANGE))
        ry = float(np_random.uniform(*ARENA_Y_RANGE))
        if (rx - GOAL_PLANE_X) ** 2 + (ry - GOAL_PLANE_Y) ** 2 < MIN_ROBOT_GOAL_DISTANCE ** 2:
            continue
        if not _is_far_enough_from_obstacles(rx, ry, extra_margin=0.05):
            continue
        yaw = float(np_random.uniform(-np.pi, np.pi))
        return [rx, ry, ROBOT_Z], ROBOT_YAW_AXIS + [yaw]
    yaw_fb = float(np_random.uniform(-np.pi, np.pi))
    return list(ROBOT_FALLBACK_POSITION), ROBOT_YAW_AXIS + [yaw_fb]


def convert_velocities_to_motor_speeds(linear_velocity, angular_velocity):
    """
    Convert (linear_velocity, angular_velocity) to (left_wheel, right_wheel).

    Based on differential drive kinematics:
        v_left  = v_linear - v_angular * (d / 2)
        v_right = v_linear + v_angular * (d / 2)

    where:
        v_linear  = forward speed of the robot center [motor rad/s]
        v_angular = turning speed (positive = left) [rad/s]
        d         = distance between wheels [m]

    Derivation:
        Robot center velocity:  v_linear  = (v_left + v_right) / 2
        Robot turning rate:     v_angular = (v_right - v_left) / d
        Solving these two equations for v_left and v_right gives the formulas above.
    """
    left_speed = linear_velocity - angular_velocity * WHEEL_DISTANCE / 2.0
    right_speed = linear_velocity + angular_velocity * WHEEL_DISTANCE / 2.0

    # Clamp to max motor speed (no reverse)
    left_speed = np.clip(left_speed, 0, MAX_SPEED)
    right_speed = np.clip(right_speed, 0, MAX_SPEED)
    return left_speed, right_speed


def analyze_camera_for_red_ball():
    """
    Analyze the camera image and detect the red ball.

    Steps:
        1. Read raw image from camera (BGRA format, 4 channels)
        2. Create red pixel mask: R > 130 AND G < 80 AND B < 80
           (red channel must be bright, green and blue must be dark)
        3. Compute red pixel ratio:
             ratio = red_pixel_count / total_pixel_count
           Result is in range 0.0 (none) to 1.0 (entire image is red)
        4. Compute horizontal position of the ball:
             avg_x = mean(x-coordinates of red pixels)
             position = (avg_x / (width - 1)) * 2 - 1
           This normalizes to range -1 (left edge) to +1 (right edge)

    Returns:
        red_pixel_ratio: ratio of red pixels (0.0 - 1.0)
        goal_horizontal_position: horizontal ball position (-1.0 to +1.0)
    """
    try:
        raw_image = camera.getImage()
    except ValueError:
        return 0.0, 0.0
    if not raw_image:
        return 0.0, 0.0

    # Webots returns image as BGRA (4 bytes per pixel)
    image = np.frombuffer(raw_image, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    # Mask: True for pixels that are "red enough"
    red_pixel_mask = (red_channel > 130) & (green_channel < 80) & (blue_channel < 80)
    red_pixel_count = np.count_nonzero(red_pixel_mask)

    # Ratio of red pixels to total image
    red_pixel_ratio = red_pixel_count / TOTAL_PIXELS

    # Horizontal goal position - where in the image the ball is located
    if red_pixel_count > 0:
        # np.where returns (rows, columns) - we want columns = x coordinates
        red_pixel_columns = np.where(red_pixel_mask)[1]
        average_x = np.mean(red_pixel_columns)
        # Normalize to [-1, +1]:
        #   pixel 0 (left edge)      -> (0 / (W-1)) * 2 - 1 = -1
        #   pixel (W-1)/2 (center)   -> (0.5) * 2 - 1       =  0
        #   pixel W-1 (right edge)   -> (1) * 2 - 1          = +1
        goal_horizontal_position = (average_x / (CAMERA_WIDTH - 1)) * 2.0 - 1.0
    else:
        goal_horizontal_position = 0.0

    return red_pixel_ratio, goal_horizontal_position


def get_front_proximity():
    """
    Read the front proximity sensor and normalize to [0.0, 1.0].

    ps0 points forward-right. Using the mean of ps0 and ps7 (forward-left)
    gives a more balanced 'front' reading.

    Returns:
        float: 0.0 = nothing in front, 1.0 = object touching the sensor
    """
    front_value = (proximity_sensors[0].getValue() + proximity_sensors[7].getValue()) / 2.0
    return np.clip(front_value / 4095.0, 0.0, 1.0)


def get_all_proximities():
    """
    Read all 8 proximity sensors (ps0-ps7) and normalize each to [0.0, 1.0].

    Layout on e-puck:
        ps7  ps0   -> forward
        ps6        ps1   -> forward-sides
        ps5        ps2   -> sides
        ps4        ps3   -> rear

    Returns:
        np.ndarray of shape (8,) with values in [0.0, 1.0].
    """
    values = np.array([s.getValue() for s in proximity_sensors], dtype=np.float32)
    return np.clip(values / 4095.0, 0.0, 1.0)


class RoombaRedBallEnv(gym.Env):
    """
    Phase 1.2: random robot position + yaw each reset; fixed goal in the scene.
    See module docstring.
    """

    def __init__(self, render_mode=None):
        super().__init__()

        # Action space: agent controls linear and angular velocity
        # linear_velocity: [0, MAX_SPEED] (no reverse), angular_velocity: [-MAX_SPEED, +MAX_SPEED]
        self.action_space = spaces.Box(
            low=np.array([0, -MAX_SPEED], dtype=np.float32),
            high=np.array([MAX_SPEED, MAX_SPEED], dtype=np.float32),
        )

        # Observation space: 12 values the agent "sees"
        #   [0]  red pixel ratio:          0.0 (none) to 1.0 (full image)
        #   [1]  goal horizontal position: -1.0 (left) to +1.0 (right)
        #   [2]  current linear velocity:  0.0 to +MAX_SPEED
        #   [3]  current angular velocity: -MAX_SPEED to +MAX_SPEED
        #   [4-11] 8 proximity sensors ps0..ps7: 0.0 (nothing) to 1.0 (touching)
        #       Full 360° physical distance signal - in obstacle phase the model
        #       can learn to use side/rear sensors to avoid walls and obstacles.
        # Defines the valid range for each value in the observation vector.
        # The RL algorithm uses these bounds to normalize inputs — they must match reality
        # exactly, otherwise the algorithm expects values that never occur.
        #   low/high[0]  red_pixel_ratio:           0.0 (no red) .. 1.0 (full image red)
        #   low/high[1]  goal_horizontal_position: -1.0 (ball left edge) .. +1.0 (ball right edge)
        #   low/high[2]  linear_velocity:           0.0 (stopped) .. MAX_SPEED (full forward)
        #   low/high[3]  angular_velocity:         -MAX_SPEED (full right) .. +MAX_SPEED (full left)
        #   low/high[4-11] proximity sensors ps0..ps7: 0.0 (nothing) .. 1.0 (touching)
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, -1.0, 0.0, -MAX_SPEED] + [0.0] * 8,
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, MAX_SPEED, MAX_SPEED] + [1.0] * 8,
                dtype=np.float32,
            ),
        )

        self.render_mode = render_mode

        # --- Environment state ---
        self.current_red_pixel_ratio = 0.0     # current red pixel ratio
        self.previous_red_pixel_ratio = 0.0    # previous step ratio (for computing delta)
        self.current_goal_position = 0.0       # horizontal ball position in the image
        self.previous_goal_position = 0.0      # previous step position (for computing centering delta)
        self.current_linear_velocity = 0.0     # current linear velocity
        self.current_angular_velocity = 0.0    # current angular velocity
        self.current_front_proximity = 0.0     # normalized front proximity (ps0+ps7 mean, used for reward)
        self.previous_front_proximity = 0.0    # for delta-based "ramming" penalty
        self.current_proximities = np.zeros(8, dtype=np.float32)  # all 8 normalized proximity sensors

        # --- Counters ---
        self.step_count = 0                    # steps in current episode
        self.steps_without_red_ball = 0        # consecutive steps without seeing the ball
        self.episode_count = 0                 # episode number
        self.episode_total_reward = 0.0        # cumulative reward in current episode
        self.steps_since_ball_in_front = 999   # steps since ball was visible and centered (999 = never)
        self.steps_since_ball_visible = 999    # steps since ball was visible at all (999 = never)
        self.steps_touching_wall = 0             # consecutive physics steps pressed against a wall (not ball)
        self.steps_near_ball = 0                 # linger near ball without progress (phase 2 orbit truncation)
        self.steps_blind_near_wall = 0           # blind + mid proximity wall hover (physics substeps)
        self.previous_sensor_max = 0.0           # for blind clearance shaping

    def _build_observation(self):
        """Build the observation vector from current environment state.
        Output is 1D array of shape (12,) with values in the order:
        """
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

        1. Stop motors
        2. Teleport robot to a random arena position and random yaw (Z axis)
        3. Reset physics (zero out inertia and velocities)
        4. Zero out all state variables
        """
        super().reset(seed=seed)

        # Stop motors
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        # Print stats of the completed episode
        if self.step_count > 0:
            self.episode_count += 1
            print(
                f"\n=== EPISODE {self.episode_count} DONE"
                f" | steps: {self.step_count}"
                f" | total reward: {self.episode_total_reward:.1f} ===\n"
            )

        pos, rot = sample_random_robot_pose(self.np_random)
        robot_translation_field.setSFVec3f(pos)
        robot_rotation_field.setSFRotation(rot)
        # resetPhysics() zeros out velocity and inertia,
        # otherwise the robot would keep moving from the previous episode
        robot_node.resetPhysics()

        # One simulation step so the new position takes effect
        robot.step(TIME_STEP)

        # Zero out state
        self.current_red_pixel_ratio = 0.0
        self.previous_red_pixel_ratio = 0.0
        self.current_goal_position = 0.0
        self.previous_goal_position = 0.0
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.current_front_proximity = 0.0
        self.previous_front_proximity = 0.0
        self.current_proximities = np.zeros(8, dtype=np.float32)
        self.step_count = 0
        self.steps_without_red_ball = 0
        self.episode_total_reward = 0.0
        self.steps_since_ball_in_front = 999
        self.steps_since_ball_visible = 999
        self.steps_touching_wall = 0
        self.steps_near_ball = 0
        self.steps_blind_near_wall = 0
        self.previous_sensor_max = 0.0

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
            robot.step(TIME_STEP) # 5 times per action to simulate 320ms of time per step

            self.current_linear_velocity = linear_velocity
            self.current_angular_velocity = angular_velocity

            self.previous_red_pixel_ratio = self.current_red_pixel_ratio
            self.previous_goal_position = self.current_goal_position
            self.current_red_pixel_ratio, self.current_goal_position = analyze_camera_for_red_ball()
            self.current_front_proximity = get_front_proximity()
            self.current_proximities = get_all_proximities()
            front_rise = max(0.0, float(self.current_front_proximity - self.previous_front_proximity))
            self.step_count += 1

            if self.current_red_pixel_ratio == 0.0:
                self.steps_without_red_ball += 1
            else:
                self.steps_without_red_ball = 0

            reward = 0.0

            # --- Step penalty ---
            reward -= 0.05

            # --- Visibility reward ---
            # Small absolute reward for seeing the ball — gives gradient for exploration.
            # At start position red_px ~ 0.004, so reward ~ 0.012, well below step penalty.
            reward += self.current_red_pixel_ratio * 3.0

            # --- Delta reward ---
            # Rewards getting CLOSER to the ball, penalizes moving away.
            pixel_ratio_change = self.current_red_pixel_ratio - self.previous_red_pixel_ratio
            reward += pixel_ratio_change * 10.0

            # Track ball visibility (two levels):
            #   ball_is_in_front  — visible AND centered (for centering delta)
            #   ball_recently_visible — seen at all recently (for goal condition)
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

            likely_near_ball = self.steps_since_ball_in_front <= 5
            ball_recently_visible = self.steps_since_ball_visible <= 30

            # --- Centering bonus ---
            # Delta-based: rewards improvement in centering, not absolute position.
            if self.current_red_pixel_ratio > 0.0:
                centering_improvement = abs(self.previous_goal_position) - abs(self.current_goal_position)
                reward += centering_improvement * 2.0
            elif not ball_recently_visible:
                reward -= 0.5

            # --- Linger near ball without progress (phase 2 — breaks orbit / stall at goal) ---
            making_progress = (
                pixel_ratio_change > 0.001
                or (abs(self.previous_goal_position) - abs(self.current_goal_position)) > 0.01
            )
            if self.current_red_pixel_ratio > 0.05 and not making_progress:
                self.steps_near_ball += 1
            elif making_progress:
                self.steps_near_ball = 0

            sensor_max = float(np.max(self.current_proximities))
            in_contact = sensor_max > WALL_CONTACT_SENSOR_MAX
            touching_ball = (
                in_contact
                and self.current_red_pixel_ratio >= TOUCHING_BALL_RED_MIN
                and ball_recently_visible
            )
            touching_wall = in_contact and not touching_ball

            lin_norm = float(np.clip(linear_velocity / MAX_SPEED, 0.0, 1.0))

            # --- Rising front proximity while not clearly seeing the ball ---
            if not touching_ball and self.current_red_pixel_ratio < FRONT_RISE_RED_MAX:
                reward -= FRONT_RISE_COEFF * front_rise

            # --- Tiered wall avoidance gradient ---
            if not touching_ball:
                if self.current_red_pixel_ratio < RED_SMALL_FOR_WALL_GRADIENT:
                    reward -= sensor_max * WALL_PROX_GRADIENT_SCALE
                elif (
                    self.current_red_pixel_ratio < WALL_GRADIENT_RED_EXTENDED
                    and sensor_max > WALL_EXTENDED_PROX_MIN
                ):
                    reward -= (
                        sensor_max
                        * WALL_PROX_GRADIENT_SCALE
                        * WALL_EXTENDED_GRADIENT_FRACTION
                    )

            # --- Discourage high forward speed when walls are close ---
            if not touching_ball and sensor_max > LINEAR_WALL_PROX_MIN:
                reward -= LINEAR_WALL_COEFF * sensor_max * lin_norm

            # --- Wall crash + stuck counter ---
            if touching_wall:
                reward -= WALL_CRASH_PENALTY
                self.steps_touching_wall += 1
            else:
                self.steps_touching_wall = 0

            # --- Wedged & blind: two-tier spin bonus + penalize ramming forward ---
            wedged_blind = (
                sensor_max > WEDGED_SENSOR_MAX
                and self.current_red_pixel_ratio < WEDGED_RED_MAX
            )
            if wedged_blind:
                ang = abs(angular_velocity)
                if linear_velocity < SPIN_BONUS_MAX_LINEAR_FRAC * MAX_SPEED:
                    reward += SPIN_WHEN_WEDGED_K * ang
                else:
                    reward += (
                        SPIN_WHEN_WEDGED_K * SPIN_WHEN_WEDGED_HIGH_LINEAR_FRAC * ang
                    )
            if wedged_blind and lin_norm > WEDGED_LINEAR_THRESHOLD:
                reward -= WEDGED_LINEAR_PENALTY_K * (lin_norm - WEDGED_LINEAR_THRESHOLD)

            # --- Blind clearance: reward backing off from walls (max IR drops) ---
            if not touching_ball and self.current_red_pixel_ratio < BLIND_RED_TH:
                sdrop = max(0.0, self.previous_sensor_max - sensor_max)
                sdrop = min(sdrop, BLIND_CLEARANCE_MAX_DELTA)
                reward += BLIND_CLEARANCE_BONUS * sdrop

            # --- Blind + mid-prox wall hover: count toward truncation ---
            if (
                not touching_ball
                and self.current_red_pixel_ratio < BLIND_RED_TH
                and sensor_max >= MID_PROX_LO
                and sensor_max < WALL_CONTACT_SENSOR_MAX
            ):
                self.steps_blind_near_wall += 1
            else:
                self.steps_blind_near_wall = 0

            # --- Goal: physical contact with the ball (same termination as phase 2) ---
            if touching_ball:
                reward += 1000.0
                print(f"*** GOAL REACHED at step {self.step_count}! proximity={self.current_front_proximity:.3f} ***")
                left_motor.setVelocity(0.0)
                right_motor.setVelocity(0.0)
                terminated = True
            elif self.current_front_proximity > 0.5:
                print(
                    f"  [GOAL DEBUG] step={self.step_count}"
                    f" proximity={self.current_front_proximity:.3f}"
                    f" red_px={self.current_red_pixel_ratio:.4f}"
                    f" steps_since_visible={self.steps_since_ball_visible}"
                    f" ball_recently_visible={ball_recently_visible}"
                    f" touching_ball={touching_ball}"
                    f" prox_max={sensor_max:.2f}"
                    f" wall_streak={self.steps_touching_wall}"
                    f" blind_wall={self.steps_blind_near_wall}"
                )

            self.previous_front_proximity = float(self.current_front_proximity)
            self.previous_sensor_max = float(sensor_max)

            accumulated_reward += reward

            # Break the frame-skip loop early if the episode ended
            timed_out = self.step_count >= MAX_STEPS_PER_EPISODE
            lost_ball = self.steps_without_red_ball >= STEPS_WITHOUT_BALL_LIMIT
            stuck_orbit = self.steps_near_ball >= STEPS_NEAR_BALL_STALL_LIMIT
            stuck_wall = self.steps_touching_wall >= STUCK_WALL_STEP_LIMIT
            stuck_blind_wall = self.steps_blind_near_wall >= BLIND_WALL_STUCK_LIMIT
            if terminated or timed_out or lost_ball or stuck_orbit or stuck_wall or stuck_blind_wall:
                break

        timed_out = self.step_count >= MAX_STEPS_PER_EPISODE
        lost_ball = self.steps_without_red_ball >= STEPS_WITHOUT_BALL_LIMIT
        stuck_orbit = self.steps_near_ball >= STEPS_NEAR_BALL_STALL_LIMIT
        stuck_wall = self.steps_touching_wall >= STUCK_WALL_STEP_LIMIT
        stuck_blind_wall = self.steps_blind_near_wall >= BLIND_WALL_STUCK_LIMIT
        truncated = timed_out or lost_ball or stuck_orbit or stuck_wall or stuck_blind_wall

        self.episode_total_reward += accumulated_reward
        observation = self._build_observation()

        if self.step_count % 100 == 0:
            sm = float(np.max(self.current_proximities))
            print(
                f"red pixels visible:"
                f"  step {self.step_count}/{MAX_STEPS_PER_EPISODE}"
                f" | red px: {self.current_red_pixel_ratio:.3f}"
                f" | goal pos: {self.current_goal_position:+.2f}"
                f" | proximity: {self.current_front_proximity:.3f}"
                f" | prox_max: {sm:.2f}"
                f" | wall {self.steps_touching_wall}"
                f" | blind_wall {self.steps_blind_near_wall}"
                f" | near_ball {self.steps_near_ball}"
                f" | lin {self.current_linear_velocity:.2f}"
                f" | ang {self.current_angular_velocity:.2f}"
                f" | ep reward: {self.episode_total_reward:.1f}"
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
import os


def run_training(
    resume_path=None,
    total_timesteps=70_000,
    output_path="following_red_ball_model_phase_1_2",
    device=None,
):
    """
    Fine-tune on phase 1.2 env. Default checkpoint is phase 1.1 unless resume_path is set.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("SAC device:", device)

    default_ckpt = "following_red_ball_model_phase_1_1"
    ckpt = resume_path if resume_path else default_ckpt
    if os.path.exists(ckpt):
        resolved = ckpt
    elif os.path.exists(ckpt + ".zip"):
        resolved = ckpt + ".zip"
    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt} (set resume_path or place {default_ckpt}.zip in the working directory)"
        )

    print(f"Loading pretrained SAC from {resolved} …")
    model = SAC.load(resolved, env=environment, device=device, verbose=1)
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    model.save(output_path)
    print(f"Model saved to {output_path}.zip")
    return model


if __name__ == "__main__":
    run_training()
