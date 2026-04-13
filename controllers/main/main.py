"""
Webots controller - e-puck robot learns to approach a red ball
using reinforcement learning (SAC).

Observation space:
    [red_pixel_ratio, goal_horizontal_position, linear_velocity, angular_velocity]

Action space:
    [linear_velocity, angular_velocity]

Reward:
    - Red pixel ratio (closer = more red pixels visible)
    - Change in red pixel ratio (approaching vs moving away)
    - Centering bonus (ball in the middle of the image)
    - Penalty for losing sight of the ball
    - Large bonus for reaching the goal
"""
import sys
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
MAX_STEPS_PER_EPISODE = 3000     # max steps per episode
RED_PIXEL_RATIO_GOAL = 0.95      # red pixel ratio threshold = robot is at the ball
STEPS_WITHOUT_BALL_LIMIT = 100   # steps without seeing the ball before truncation

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
    raw_image = camera.getImage()
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

        # Observation space: 4 values the agent "sees"
        #   [0] red pixel ratio:          0.0 (none) to 1.0 (full image)
        #   [1] goal horizontal position: -1.0 (left) to +1.0 (right)
        #   [2] current linear velocity:  -MAX_SPEED to +MAX_SPEED
        #   [3] current angular velocity: -MAX_SPEED to +MAX_SPEED
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -MAX_SPEED, -MAX_SPEED], dtype=np.float32),
            high=np.array([1.0, 1.0, MAX_SPEED, MAX_SPEED], dtype=np.float32),
        )

        self.render_mode = render_mode

        # --- Environment state ---
        self.current_red_pixel_ratio = 0.0     # current red pixel ratio
        self.previous_red_pixel_ratio = 0.0    # previous step ratio (for computing change)
        self.current_goal_position = 0.0       # horizontal ball position in the image
        self.current_linear_velocity = 0.0     # current linear velocity
        self.current_angular_velocity = 0.0    # current angular velocity

        # --- Counters ---
        self.step_count = 0                    # steps in current episode
        self.steps_without_red_ball = 0        # consecutive steps without seeing the ball
        self.episode_count = 0                 # episode number
        self.episode_total_reward = 0.0        # cumulative reward in current episode

    def _build_observation(self):
        """Build the observation vector from current environment state."""
        return np.array([
            self.current_red_pixel_ratio,
            self.current_goal_position,
            self.current_linear_velocity,
            self.current_angular_velocity,
        ], dtype=np.float32)

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
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.step_count = 0
        self.steps_without_red_ball = 0
        self.episode_total_reward = 0.0

        return self._build_observation(), {}

    def step(self, action):
        """
        Execute one environment step.

        1. Unpack action from the agent (linear and angular velocity)
        2. Convert to motor speeds and apply
        3. Advance simulation by one TIME_STEP
        4. Analyze camera - detect the red ball
        5. Compute reward
        6. Check episode termination conditions
        """
        # Unpack action from the agent
        linear_velocity = float(action[0])
        angular_velocity = float(action[1])

        # Convert linear/angular velocity to wheel speeds and set motors
        left_speed, right_speed = convert_velocities_to_motor_speeds(linear_velocity, angular_velocity)
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        # Advance simulation by one step
        robot.step(TIME_STEP)

        # Save current velocities to state (will be part of the observation)
        self.current_linear_velocity = linear_velocity
        self.current_angular_velocity = angular_velocity

        # Analyze camera - find how many red pixels we see and where the ball is
        self.previous_red_pixel_ratio = self.current_red_pixel_ratio
        self.current_red_pixel_ratio, self.current_goal_position = analyze_camera_for_red_ball()
        self.step_count += 1

        # Track consecutive steps without seeing the ball
        if self.current_red_pixel_ratio == 0.0:
            self.steps_without_red_ball += 1
        else:
            self.steps_without_red_ball = 0

        # =============================================
        # REWARD COMPUTATION
        # =============================================
        reward = 0.0

        # 1) Reward for ball size in the image
        #    More red pixels = closer to the ball
        #    Formula: reward += ratio * 2.0
        #    Example: ratio 0.1 (10% of image) -> +0.2, ratio 0.5 -> +1.0
        reward += self.current_red_pixel_ratio * 2.0

        # 2) Reward for approaching (change in ratio vs previous step)
        #    Positive change = getting closer, negative = moving away
        #    Formula: reward += (current_ratio - previous_ratio) * 5.0
        #    Example: ratio grew from 0.1 to 0.15 -> change +0.05 -> reward +0.25
        #             ratio fell from 0.15 to 0.1  -> change -0.05 -> reward -0.25
        pixel_ratio_change = self.current_red_pixel_ratio - self.previous_red_pixel_ratio
        reward += pixel_ratio_change * 5.0

        # 3) Reward for centering the ball in the middle of the image
        #    When we see the ball, we want it as close to center (position = 0) as possible.
        #    Formula: centering = 1.0 - |position|, then reward += centering * 1.0
        #    Example: position  0.0 (center)       -> centering 1.0 -> reward +1.0
        #             position -0.5 (slightly left) -> centering 0.5 -> reward +0.5
        #             position -1.0 (far left)      -> centering 0.0 -> reward +0.0
        #    When we don't see the ball -> penalty -0.5
        if self.current_red_pixel_ratio > 0.0:
            centering_reward = 1.0 - abs(self.current_goal_position)
            reward += centering_reward * 1.0
        else:
            reward -= 0.5

        # 4) Large bonus for reaching the goal (ball fills almost the entire image)
        #    If red pixel ratio >= RED_PIXEL_RATIO_GOAL -> bonus +100
        reached_goal = self.current_red_pixel_ratio >= RED_PIXEL_RATIO_GOAL
        if reached_goal:
            reward += 100.0
            print(f"*** GOAL REACHED at step {self.step_count}! ***")

        # =============================================
        # EPISODE TERMINATION CONDITIONS
        # =============================================
        # terminated (reached_goal) = agent reached the goal (success)
        # truncated = timeout or lost the ball for too long (failure)
        timed_out = self.step_count >= MAX_STEPS_PER_EPISODE
        lost_ball = self.steps_without_red_ball >= STEPS_WITHOUT_BALL_LIMIT
        truncated = timed_out or lost_ball

        # Accumulate reward for statistics
        self.episode_total_reward += reward
        observation = self._build_observation()

        # Log every 100 steps
        if self.step_count % 100 == 0:
            print(
                f"  step {self.step_count}/{MAX_STEPS_PER_EPISODE}"
                f" | red px: {self.current_red_pixel_ratio:.3f}"
                f" | goal pos: {self.current_goal_position:+.2f}"
                f" | ep reward: {self.episode_total_reward:.1f}"
            )

        return observation, reward, reached_goal, truncated, {}

    def close(self):
        """Clean up resources (OpenCV windows)."""
        cv2.destroyAllWindows()


# =============================================
# TRAINING
# =============================================
environment = RoombaRedBallEnv()

from stable_baselines3 import SAC

# SAC (Soft Actor-Critic) - off-policy algorithm suitable for continuous actions
# MlpPolicy = two-layer neural network (2x64 neurons) for our 4 input values
model = SAC("MlpPolicy", environment, verbose=1)
model.learn(total_timesteps=15000)
model.save("following_red_ball_model")
