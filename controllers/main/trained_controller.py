"""
Webots controller - e-puck robot uses a trained SAC model
to approach a red ball (inference only, no training).

Usage:
    1. Place this file as the robot controller in Webots
    2. Make sure 'following_red_ball_model.zip' is in the same folder
       (or set MODEL_PATH below to an absolute path)
    3. Run the simulation - the robot will use the trained policy
"""
import sys
sys.path.append(r'C:\Program Files\Webots\lib\controller\python')
from controller import Supervisor
import numpy as np

# --- Model path ---
MODEL_PATH = "following_red_ball_model"   # .zip extension is optional for SB3

# --- Simulation constants ---
TIME_STEP = 64
MAX_SPEED = 6.279

# --- Episode constants ---
MAX_STEPS_PER_EPISODE = 3000
RED_PIXEL_RATIO_GOAL = 0.95
STEPS_WITHOUT_BALL_LIMIT = 100
FRAME_SKIP = 5

# --- Robot starting pose ---
ROBOT_START_POSITION = [-1.1, 0.019, 0.0]
ROBOT_START_ROTATION = [0, 0, 1, 0]

# --- Wheel geometry ---
WHEEL_DISTANCE = 0.052  # [m]

# =============================================
# SUPERVISOR + DEVICES INIT
# =============================================
robot = Supervisor()

# Normal rendering mode so you can watch the trained agent
# Switch to SIMULATION_MODE_FAST if you still want speed
#robot.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)
robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

# Proximity sensors (not used in observations but kept enabled)
proximity_sensors = []
for name in ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']:
    s = robot.getDevice(name)
    s.enable(TIME_STEP)
    proximity_sensors.append(s)

# Camera
camera = robot.getDevice('camera')
camera.enable(TIME_STEP)
CAMERA_WIDTH  = camera.getWidth()
CAMERA_HEIGHT = camera.getHeight()
TOTAL_PIXELS  = CAMERA_WIDTH * CAMERA_HEIGHT

# Motors
left_motor  = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Supervisor references for position reset
robot_node              = robot.getFromDef('ROBOT')
robot_translation_field = robot_node.getField('translation')
robot_rotation_field    = robot_node.getField('rotation')


# =============================================
# HELPER FUNCTIONS
# =============================================
def convert_velocities_to_motor_speeds(linear_velocity, angular_velocity):
    left_speed  = linear_velocity - angular_velocity * WHEEL_DISTANCE / 2.0
    right_speed = linear_velocity + angular_velocity * WHEEL_DISTANCE / 2.0
    left_speed  = np.clip(left_speed,  -MAX_SPEED, MAX_SPEED)
    right_speed = np.clip(right_speed, -MAX_SPEED, MAX_SPEED)
    return left_speed, right_speed


def analyze_camera_for_red_ball():
    raw_image = camera.getImage()
    if not raw_image:
        return 0.0, 0.0

    image         = np.frombuffer(raw_image, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
    blue_channel  = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel   = image[:, :, 2]

    red_pixel_mask  = (red_channel > 130) & (green_channel < 80) & (blue_channel < 80)
    red_pixel_count = np.count_nonzero(red_pixel_mask)
    red_pixel_ratio = red_pixel_count / TOTAL_PIXELS

    if red_pixel_count > 0:
        red_pixel_columns      = np.where(red_pixel_mask)[1]
        average_x              = np.mean(red_pixel_columns)
        goal_horizontal_position = (average_x / (CAMERA_WIDTH - 1)) * 2.0 - 1.0
    else:
        goal_horizontal_position = 0.0

    return red_pixel_ratio, goal_horizontal_position


def reset_robot():
    """Teleport robot to starting position and reset physics."""
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    robot_translation_field.setSFVec3f(ROBOT_START_POSITION)
    robot_rotation_field.setSFRotation(ROBOT_START_ROTATION)
    robot_node.resetPhysics()
    robot.step(TIME_STEP)


def build_observation(red_pixel_ratio, goal_horizontal_position,
                       linear_velocity, angular_velocity):
    return np.array([
        red_pixel_ratio,
        goal_horizontal_position,
        linear_velocity,
        angular_velocity,
    ], dtype=np.float32)


# =============================================
# LOAD TRAINED MODEL
# =============================================
from stable_baselines3 import SAC

print(f"Loading model from: {MODEL_PATH}")
model = SAC.load(MODEL_PATH)
print("Model loaded successfully.")

# =============================================
# INFERENCE LOOP
# =============================================
episode        = 0
step_count     = 0
steps_no_ball  = 0
lin_vel        = 0.0
ang_vel        = 0.0
red_ratio      = 0.0
goal_pos       = 0.0
episode_reward = 0.0

reset_robot()

print("Starting inference. Press Ctrl+C in the terminal or stop Webots to quit.\n")

while robot.step(TIME_STEP) != -1:
    # --- Build current observation ---
    red_ratio, goal_pos = analyze_camera_for_red_ball()
    obs = build_observation(red_ratio, goal_pos, lin_vel, ang_vel)

    # --- Ask the policy for an action (deterministic=True = no exploration noise) ---
    action, _ = model.predict(obs, deterministic=True)

    lin_vel = float(action[0])
    ang_vel = float(action[1])

    left_speed, right_speed = convert_velocities_to_motor_speeds(lin_vel, ang_vel)
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # --- Frame skip: apply same action FRAME_SKIP times ---
    for _ in range(FRAME_SKIP - 1):   # -1 because we already did one step above
        if robot.step(TIME_STEP) == -1:
            break
        red_ratio, goal_pos = analyze_camera_for_red_ball()
        step_count += 1

        if red_ratio == 0.0:
            steps_no_ball += 1
        else:
            steps_no_ball = 0

        if red_ratio >= RED_PIXEL_RATIO_GOAL:
            print(f"*** GOAL REACHED at step {step_count} in episode {episode + 1}! ***")
            break

    step_count += 1

    # --- Logging every 100 steps ---
    if step_count % 100 == 0:
        print(
            f"  ep {episode + 1} | step {step_count}/{MAX_STEPS_PER_EPISODE}"
            f" | red px: {red_ratio:.3f}"
            f" | goal pos: {goal_pos:+.2f}"
            f" | lin: {lin_vel:+.2f}  ang: {ang_vel:+.2f}"
        )

    # --- Check episode end conditions ---
    reached_goal = red_ratio >= RED_PIXEL_RATIO_GOAL
    timed_out    = step_count >= MAX_STEPS_PER_EPISODE
    lost_ball    = steps_no_ball >= STEPS_WITHOUT_BALL_LIMIT

    if reached_goal or timed_out or lost_ball:
        reason = "GOAL" if reached_goal else ("TIMEOUT" if timed_out else "LOST BALL")
        print(f"\n=== EPISODE {episode + 1} END [{reason}] | steps: {step_count} ===\n")

        episode       += 1
        step_count     = 0
        steps_no_ball  = 0
        lin_vel        = 0.0
        ang_vel        = 0.0
        episode_reward = 0.0

        reset_robot()
