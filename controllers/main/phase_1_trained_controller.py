import sys

DESKTOP_SETUP = True
if DESKTOP_SETUP:
    sys.path.append(r'D:\Webots\lib\controller\python')
else:
    sys.path.append(r'C:\Program Files\Webots\lib\controller\python')

from controller import Supervisor
import numpy as np
from stable_baselines3 import SAC

MODEL_PATH = "following_red_ball_model"

TIME_STEP = 64
MAX_SPEED = 6.279
WHEEL_DISTANCE = 0.052

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

while robot.step(TIME_STEP) != -1:
    red_pixel_ratio, goal_horizontal_position = analyze_camera_for_red_ball()
    proximity_values = read_proximity_sensors()
    observation = np.concatenate([
        np.array(
            [red_pixel_ratio, goal_horizontal_position, linear_velocity, angular_velocity],
            dtype=np.float32,
        ),
        proximity_values,
    ])

    action, _ = model.predict(observation, deterministic=True)
    linear_velocity = float(action[0])
    angular_velocity = float(action[1])

    left_wheel_speed = np.clip(
        linear_velocity - angular_velocity * WHEEL_DISTANCE / 2.0,
        -MAX_SPEED,
        MAX_SPEED,
    )
    right_wheel_speed = np.clip(
        linear_velocity + angular_velocity * WHEEL_DISTANCE / 2.0,
        -MAX_SPEED,
        MAX_SPEED,
    )
    left_motor.setVelocity(left_wheel_speed)
    right_motor.setVelocity(right_wheel_speed)
