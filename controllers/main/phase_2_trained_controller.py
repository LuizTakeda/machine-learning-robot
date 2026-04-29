import sys
DESKTOP_SETUP = True

if DESKTOP_SETUP:
    sys.path.append(r'D:\Webots\lib\controller\python')
else:
    sys.path.append(r'C:\Program Files\Webots\lib\controller\python')

from controller import Supervisor
import numpy as np
from stable_baselines3 import SAC

MODEL_PATH = "obstacle_avoidance_following_red_ball_model"

TIME_STEP      = 64
MAX_SPEED      = 6.279
WHEEL_DISTANCE = 0.052
FRAME_SKIP     = 5

MAX_STEPS_PER_EPISODE  = 10000
STEPS_WITHOUT_BALL_LIMIT = 10000

ROBOT_START_POSITION = [-1.1, 0.019, 0.0]
ROBOT_START_ROTATION = [0, 0, 1, 0]

robot = Supervisor()
robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

proximity_sensors = []
for name in ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']:
    s = robot.getDevice(name)
    s.enable(TIME_STEP)
    proximity_sensors.append(s)

camera = robot.getDevice('camera')
camera.enable(TIME_STEP)
CAMERA_WIDTH  = camera.getWidth()
CAMERA_HEIGHT = camera.getHeight()
TOTAL_PIXELS  = CAMERA_WIDTH * CAMERA_HEIGHT

left_motor  = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

robot_node              = robot.getFromDef('ROBOT')
robot_translation_field = robot_node.getField('translation')
robot_rotation_field    = robot_node.getField('rotation')


def get_camera():
    try:
        raw = camera.getImage()
    except ValueError:
        return 0.0, 0.0
    if not raw:
        return 0.0, 0.0
    image = np.frombuffer(raw, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
    mask  = (image[:,:,2] > 130) & (image[:,:,1] < 80) & (image[:,:,0] < 80)
    count = np.count_nonzero(mask)
    ratio = count / TOTAL_PIXELS
    if count > 0:
        avg_x    = np.mean(np.where(mask)[1])
        goal_pos = (avg_x / (CAMERA_WIDTH - 1)) * 2.0 - 1.0
    else:
        goal_pos = 0.0
    return ratio, goal_pos


def get_proximities():
    vals = np.array([s.getValue() for s in proximity_sensors], dtype=np.float32)
    return np.clip(vals / 4095.0, 0.0, 1.0)


def get_front_proximity():
    return float(np.clip(
        (proximity_sensors[0].getValue() + proximity_sensors[7].getValue()) / 2.0 / 4095.0,
        0.0, 1.0
    ))


def build_observation(red_ratio, goal_pos, lin_vel, ang_vel, proximities):
    return np.concatenate([
        np.array([red_ratio, goal_pos, lin_vel, ang_vel], dtype=np.float32),
        proximities,
    ])


def motor_speeds(lin_vel, ang_vel):
    l = np.clip(lin_vel - ang_vel * WHEEL_DISTANCE / 2.0, -MAX_SPEED, MAX_SPEED)
    r = np.clip(lin_vel + ang_vel * WHEEL_DISTANCE / 2.0, -MAX_SPEED, MAX_SPEED)
    return l, r


def reset_robot():
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    robot_translation_field.setSFVec3f(ROBOT_START_POSITION)
    robot_rotation_field.setSFRotation(ROBOT_START_ROTATION)
    robot_node.resetPhysics()
    robot.step(TIME_STEP)


print(f"Loading model: {MODEL_PATH}")
model = SAC.load(MODEL_PATH)
print("Model loaded.\n")

episode      = 0
step_count   = 0
steps_no_ball = 0
lin_vel      = 0.0
ang_vel      = 0.0

reset_robot()

while robot.step(TIME_STEP) != -1:
    red_ratio, goal_pos = get_camera()
    proximities         = get_proximities()
    front_prox          = get_front_proximity()

    obs    = build_observation(red_ratio, goal_pos, lin_vel, ang_vel, proximities)
    action, _ = model.predict(obs, deterministic=True)

    lin_vel = float(action[0])
    ang_vel = float(action[1])
    l, r    = motor_speeds(lin_vel, ang_vel)
    left_motor.setVelocity(l)
    right_motor.setVelocity(r)

    for _ in range(FRAME_SKIP - 1):
        if robot.step(TIME_STEP) == -1:
            break
        red_ratio, goal_pos = get_camera()
        step_count += 1
        if red_ratio == 0.0:
            steps_no_ball += 1
        else:
            steps_no_ball = 0

    step_count += 1

    if step_count % 100 == 0:
        print(
            f"  ep {episode+1} | step {step_count}"
            f" | red px: {red_ratio:.3f}"
            f" | goal pos: {goal_pos:+.2f}"
            f" | front prox: {front_prox:.3f}"
        )

    reached_goal = front_prox >= 0.25 and red_ratio >= 0.5
    timed_out    = step_count >= MAX_STEPS_PER_EPISODE
    lost_ball    = steps_no_ball >= STEPS_WITHOUT_BALL_LIMIT

    if reached_goal or timed_out or lost_ball:
        reason = "GOAL" if reached_goal else ("TIMEOUT" if timed_out else "LOST BALL")
        print(f"\n=== EPISODE {episode+1} END [{reason}] | steps: {step_count} ===\n")
        episode      += 1
        step_count    = 0
        steps_no_ball = 0
        lin_vel       = 0.0
        ang_vel       = 0.0
        reset_robot()
