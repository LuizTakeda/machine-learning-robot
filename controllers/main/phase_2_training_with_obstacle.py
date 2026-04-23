"""
Webots controller - e-puck robot learns to approach a red ball
using reinforcement learning (SAC).

Observation space:
    [red_pixel_ratio, goal_horizontal_position, linear_velocity, angular_velocity, front_proximity]

Action space:
    [linear_velocity, angular_velocity]

Reward:
    - Red pixel ratio (closer = more red pixels visible)
    - Change in red pixel ratio (approaching vs moving away)
    - Centering bonus (ball in the middle of the image)
    - Penalty for losing sight of the ball
    - Step penalty (forces the agent to move, not stand still)
    - Proximity bonus (physical closeness via front sensor)
    - Large bonus for reaching the goal (physical contact)
"""
import sys
DESKTOP_SETUP = True

if DESKTOP_SETUP == True:
    sys.path.append(r'D:\Webots\lib\controller\python')
else:
    sys.path.append(r'C:\Program Files\Webots\lib\controller\python')

from controller import Supervisor, Motor, Camera, DistanceSensor
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2

# --- Simulation constants ---
TIME_STEP = 64                   # simulation step in milliseconds
MAX_SPEED = 6.279                # max angular velocity of e-puck motors [rad/s]

# --- Episode constants ---
MAX_STEPS_PER_EPISODE = 10000     # max steps per episode
# Physical contact threshold: front proximity sensor value > 0.95 means the robot
# is essentially touching the ball. Much more realistic than a 95% pixel ratio.
PROXIMITY_GOAL_THRESHOLD = 0.95  # normalized proximity sensor value (0.0 - 1.0)
STEPS_WITHOUT_BALL_LIMIT = 6000   # steps without seeing the ball before truncation
FRAME_SKIP = 5

# --- Robot starting pose (values from .wbt file) ---
ROBOT_START_POSITION = [-1.1, 0.019, 0.0]   # [x, y, z] in meters
ROBOT_START_ROTATION = [0, 0, 1, 0]         # [axis_x, axis_y, axis_z, angle] facing the ball

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

# --- Robot dimensions ---
# Distance between e-puck wheels - needed for converting
# linear/angular velocity to individual wheel speeds
WHEEL_DISTANCE = 0.052  # [meters]


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

    # Clamp to max motor speed
    left_speed = np.clip(left_speed, -MAX_SPEED, MAX_SPEED)
    right_speed = np.clip(right_speed, -MAX_SPEED, MAX_SPEED)
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
    Gymnasium environment for an e-puck robot learning to find and approach
    a red ball in a Webots simulation.
    """

    def __init__(self, render_mode=None):
        super().__init__()

        # Action space: agent controls linear and angular velocity
        # Both in range [-MAX_SPEED, +MAX_SPEED]
        self.action_space = spaces.Box(
            low=np.array([-MAX_SPEED, -MAX_SPEED], dtype=np.float32),
            high=np.array([MAX_SPEED, MAX_SPEED], dtype=np.float32),
        )

        # Observation space: 12 values the agent "sees"
        #   [0]  red pixel ratio:          0.0 (none) to 1.0 (full image)
        #   [1]  goal horizontal position: -1.0 (left) to +1.0 (right)
        #   [2]  current linear velocity:  -MAX_SPEED to +MAX_SPEED
        #   [3]  current angular velocity: -MAX_SPEED to +MAX_SPEED
        #   [4-11] 8 proximity sensors ps0..ps7: 0.0 (nothing) to 1.0 (touching)
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
        self.current_red_pixel_ratio = 0.0     # current red pixel ratio
        self.previous_red_pixel_ratio = 0.0    # previous step ratio (for computing delta)
        self.current_goal_position = 0.0       # horizontal ball position in the image
        self.previous_goal_position = 0.0      # previous step position (for centering delta)
        self.current_linear_velocity = 0.0     # current linear velocity
        self.current_angular_velocity = 0.0    # current angular velocity
        self.current_front_proximity = 0.0     # normalized front proximity (ps0+ps7 mean, used for reward)
        self.current_proximities = np.zeros(8, dtype=np.float32)  # all 8 normalized proximity sensors

        # --- Counters ---
        self.step_count = 0                    # steps in current episode
        self.steps_without_red_ball = 0        # consecutive steps without seeing the ball
        self.episode_count = 0                 # episode number
        self.episode_total_reward = 0.0        # cumulative reward in current episode
        self.steps_since_ball_in_front = 999   # steps since ball was visible and centered (999 = never)
        self.steps_since_ball_visible = 999    # steps since ball was visible at all (999 = never)

    def _build_observation(self):
        """Build the observation vector from current environment state."""
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
        2. Teleport robot back to starting position (supervisor API)
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
            print(f"\n=== EPISODE {self.episode_count} DONE | steps: {self.step_count} | total reward: {self.episode_total_reward:.1f} ===\n")

        # Teleport robot to starting position using supervisor API
        robot_translation_field.setSFVec3f(ROBOT_START_POSITION)
        robot_rotation_field.setSFRotation(ROBOT_START_ROTATION)
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
        self.current_proximities = np.zeros(8, dtype=np.float32)
        self.step_count = 0
        self.steps_without_red_ball = 0
        self.episode_total_reward = 0.0
        self.steps_since_ball_in_front = 999
        self.steps_since_ball_visible = 999

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

            # --- Step penalty ---
            reward -= 0.05

            # --- Visibility reward ---
            reward += self.current_red_pixel_ratio * 3.0

            # --- Delta reward ---
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

            ball_recently_visible = self.steps_since_ball_visible <= 30

            # --- Centering bonus (delta-based) ---
            if self.current_red_pixel_ratio > 0.0:
                centering_improvement = abs(self.previous_goal_position) - abs(self.current_goal_position)
                reward += centering_improvement * 2.0
            elif not ball_recently_visible:
                reward -= 0.5

            # --- Goal: physical contact with the ball ---
            reached_goal = self.current_front_proximity >= PROXIMITY_GOAL_THRESHOLD and ball_recently_visible
            if reached_goal:
                reward += 100.0
                print(f"*** GOAL REACHED at step {self.step_count}! proximity={self.current_front_proximity:.3f} ***")
                terminated = True

            accumulated_reward += reward

            # Break the frame-skip loop early if the episode ended
            timed_out = self.step_count >= MAX_STEPS_PER_EPISODE
            lost_ball = self.steps_without_red_ball >= STEPS_WITHOUT_BALL_LIMIT
            if terminated or timed_out or lost_ball:
                break

        timed_out = self.step_count >= MAX_STEPS_PER_EPISODE
        lost_ball = self.steps_without_red_ball >= STEPS_WITHOUT_BALL_LIMIT
        truncated = timed_out or lost_ball

        self.episode_total_reward += accumulated_reward
        observation = self._build_observation()

        if self.step_count % 100 == 0:
            print(
                f"  step {self.step_count}/{MAX_STEPS_PER_EPISODE}"
                f" | red px: {self.current_red_pixel_ratio:.3f}"
                f" | goal pos: {self.current_goal_position:+.2f}"
                f" | proximity: {self.current_front_proximity:.3f}"
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

PRETRAINED_MODEL_PATH = "following_red_ball_model.zip"
REPLAY_BUFFER_PATH = "following_red_ball_replay_buffer.pkl"

import os

if os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}")
    model = SAC.load( #ignores kwargs, have to define them in custom_objects
        PRETRAINED_MODEL_PATH,
        env=environment,
        verbose=1,
        device="cuda",
        learning_starts=0,
        custom_objects={
            "learning_rate": 1e-4,
            "batch_size": 256,
            "gamma": 0.99,
        }
    )

    if os.path.exists(REPLAY_BUFFER_PATH):
        print(f"Loading replay buffer from {REPLAY_BUFFER_PATH}")
        model.load_replay_buffer(REPLAY_BUFFER_PATH)
else:
    print("No pretrained model found")

# reset_num_timesteps=False keeps the step counter / LR schedule continuous
# across training sessions when resuming from a checkpoint.
model.learn(total_timesteps=100_000, reset_num_timesteps=False)
model.save("obstacle_avoidance_following_red_ball_model")
model.save_replay_buffer("obstacle_avoidance_following_red_ball_replay_buffer.pkl")
print("Model saved to obstacle_avoidance_following_red_ball_model.zip")
