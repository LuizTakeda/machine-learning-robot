"""epuck_go_forward controller."""
"""
distance senosors: 0 -> no light is detected, 4095 ->o maximum light is detected
"""
from controller import Robot, Motor, Camera, DistanceSensor

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

#constants
TIME_STEP = 64
MAX_SPEED = 6.24 * 4.0

#init
robot = Robot()

proximity_sensors = []
PROXIMITY_SENSORS_NAMES = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']

for index in range(8):
    proximity_sensors.append(robot.getDevice(PROXIMITY_SENSORS_NAMES[index]))
    proximity_sensors[index].enable(TIME_STEP)

camera = robot.getDevice('camera')
camera.enable(TIME_STEP)

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
    print(f"[{robot.getTime():.2f}s] left={leftMotor.getVelocity():.2f} right={rightMotor.getVelocity():.2f}")

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

    MAX_SPEED = 6.28
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
    pass