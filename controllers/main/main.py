"""
distance senosors: 0 -> no light is detected, 4095 ->o maximum light is detected
"""
import sys

from networkx.classes import number_of_edges

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
rightMotor.setVelocity(0.1 * MAX_SPEED)

"""

# init of pen - pen has to be attached for this to work
pen = robot.getDevice('pen')

# color + intensity of the color
pen.setInkColor(0xFF0000, 1.0)  # red, full intensity
# Turn on the writing
pen.write(True)
"""


while robot.step(TIME_STEP) != -1:
    #print(f"[{robot.getTime():.2f}s] left={leftMotor.getVelocity():.2f} right={rightMotor.getVelocity():.2f}")

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
        left_speed = 0.5 * MAX_SPEED
        right_speed = -0.5 * MAX_SPEED

    if right_obstacle:
        print("Right obstacle detected, turning left")
        left_speed = -0.5 * MAX_SPEED
        right_speed = 0.5 * MAX_SPEED

    leftMotor.setVelocity(left_speed)
    rightMotor.setVelocity(right_speed)

    raw_image = camera.getImage()
    if raw_image:
        # Webots sends format BGRA (Blue, Green, Red, Alpha)
        width = camera.getWidth()
        height = camera.getHeight()

        # image construction
        image = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))


        BGR_image = image[:, :, :3]
        imRed = BGR_image[:, :, 2]
        imGreen = BGR_image[:, :, 1]
        imBlue = BGR_image[:, :, 0]

        [rows, columns, num_of_chanes] = BGR_image.shape[:3]

        output_image = np.zeros((rows, columns), dtype=np.uint8)  # Create an empty output image

        for row in range(rows):  # Loop over each row index
            for column in range(columns):  # Loop over each column index
                if imRed[row, column] > 130 and imGreen[row, column] < 80 and imBlue[row, column] < 80:  # Check if the pixel is red
                    output_image[row, column] = 255  # Set the output pixel to white (255) if the condition is met, otherwise it remains black (0)

        number_of_red_pixels = np.count_nonzero(output_image)
        print(f"SEEING {number_of_red_pixels} RED pixels")


        # view of robot
        """
        cv2.imshow("Robot view", image)
        cv2.waitKey(1)  # wait 1 ms for window render
        """
        cv2.imshow("Robot view red", output_image)
        cv2.waitKey(1)  # wait 1 ms for window render


    pass