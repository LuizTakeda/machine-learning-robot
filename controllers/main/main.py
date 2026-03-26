"""
distance senosors: 0 -> no light is detected, 4095 ->o maximum light is detected
"""
import sys
#sys.path.append(r'C:\Program Files\Webots\lib\controller\python') # add Webots controller library to the path
from controller import Robot, Motor, Camera, DistanceSensor
import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import cv2

TIME_STEP = 64
MAX_SPEED = 6.279

robot = Robot()

proximity_sensors = []
PROXIMITY_SENSORS_NAMES = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']

for index in range(8):
    proximity_sensors.append(robot.getDevice(PROXIMITY_SENSORS_NAMES[index]))
    proximity_sensors[index].enable(TIME_STEP)

camera = robot.getDevice('camera')
camera.enable(TIME_STEP)
#camera resolution can be set in camera_width and camera_height fields in webots

#initialization of motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

leftMotor.setPosition(float('inf')) #setting destinatoin posotion - infinity - continuous
rightMotor.setPosition(float('inf'))

# set up the motor speeds at 10% of the MAX_SPEED.
leftMotor.setVelocity(0.15 * MAX_SPEED)
rightMotor.setVelocity(0.15 * MAX_SPEED)

"""

# init of pen - pen has to be attached for this to work
pen = robot.getDevice('pen')

# color + intensity of the color
pen.setInkColor(0xFF0000, 1.0)  # red, full intensity
# Turn on the writing
pen.write(True)
"""



class MyEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        # 2 - left motor and right motor continuous speed control
        self.action_space = spaces.Box(low=-MAX_SPEED, high=MAX_SPEED, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.state = {"previous_red_pixels": 0.0, "current_red_pixels": 0.0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        robot.step(TIME_STEP)

        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)

        self.state = {"previous_red_pixels": 0.0, "current_red_pixels": 0.0}
        return np.array([0.0], dtype=np.float32), {}

    def red_pixels_observed(self):
        raw_image = camera.getImage()
        if raw_image:
            width = camera.getWidth()
            height = camera.getHeight()
            image = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))

            BGR = image[:, :, :3]
            # NumPy maska - faster than loop
            mask = (BGR[:, :, 2] > 130) & (BGR[:, :, 1] < 80) & (BGR[:, :, 0] < 80)
            count = float(np.count_nonzero(mask))

            print(f"SEEING {count} RED pixels")
            cv2.imshow("Robot view red", mask.astype(np.uint8) * 255)
            cv2.waitKey(1)

            # správný přístup ke slovníku
            self.state["current_red_pixels"] = count


    def step(self, action):
        robot.step(TIME_STEP)
        leftMotor.setVelocity(float(action[0]))
        rightMotor.setVelocity(float(action[1]))

        self.red_pixels_observed()

        # odměna podle změny červených pixelů
        if self.state["current_red_pixels"] > self.state["previous_red_pixels"]:
            reward = 1   # robot got closer
        else:
            reward = -1  # robot got further

        self.state["previous_red_pixels"] = self.state["current_red_pixels"]
        observation = np.array([self.state["current_red_pixels"]], dtype=np.float32)

        return observation, reward, False, False, {}

    def close(self):
        cv2.destroyAllWindows()




environment = MyEnv()
model = PPO("MlpPolicy", environment, verbose=1)
model.learn(total_timesteps=50000)
model.save("following_red_ball_model")




"""
while robot.step(TIME_STEP) != -1:
    environment.step()

    pass
"""


"""
 # read sensor outputs
    proximity_sensor_values = []
    for index in range(8):
        proximity_sensor_values.append(proximity_sensors[index].getValue())

    right_obstacle: bool = (proximity_sensor_values[0] > 80.0
                            or proximity_sensor_values[1] > 80.0
                            or proximity_sensor_values[2] > 80.0)
    left_obstacle: bool = (proximity_sensor_values[5] > 80.0
                            or proximity_sensor_values[6] > 80.0
                            or proximity_sensor_values[7] > 80.0)

    left_speed = 0.5 * MAX_SPEED
    right_speed = 0.5 * MAX_SPEED

    if left_obstacle:
        print("Left obstacle detected, turning right")
        #left_speed = 0.5 * MAX_SPEED
        #right_speed = -0.5 * MAX_SPEED

    if right_obstacle:
        print("Right obstacle detected, turning left")
        #left_speed = -0.5 * MAX_SPEED
        #right_speed = 0.5 * MAX_SPEED
"""