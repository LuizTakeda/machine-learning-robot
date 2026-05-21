import gymnasium as gym
import numpy as np

from controller import Supervisor, Motor, Camera, DistanceSensor
from typing import cast
from gymnasium.spaces import Box
from abc import ABC, abstractmethod


class BaseEnv(gym.Env, ABC):

    # =========================
    # CONSTANTS
    # =========================

    _TIME_STEP = 64
    _FRAME_SKIP = 4
    _DISTANCE_SENSOR_NAMES = [
        'ps0', 'ps1', 'ps2', 'ps3',
        'ps4', 'ps5', 'ps6', 'ps7'
    ]

    # =========================
    # INITIALIZATION
    # =========================

    def __init__(self):
        super().__init__()

        self._robot = Supervisor()

        self._robot_node = self._robot.getFromDef('ROBOT')
        self._robot_translation_field = self._robot_node.getField('translation')
        self._robot_rotation_field = self._robot_node.getField('rotation')

        # -------- Sensors --------

        self._distance_sensor = {}

        for name in self._DISTANCE_SENSOR_NAMES:
            sensor = cast(DistanceSensor, self._robot.getDevice(name))
            sensor.enable(self._TIME_STEP)
            self._distance_sensor[name] = sensor

        # -------- Camera --------

        self._camera = cast(Camera, self._robot.getDevice('camera'))
        self._camera.enable(self._TIME_STEP)

        # -------- Motors --------

        self._leftMotor = cast(Motor, self._robot.getDevice('left wheel motor'))
        self._leftMotor.setPosition(float('inf')) # Continuous
        self._MAX_VELOCITY = self._leftMotor.getMaxVelocity()
        self._leftMotor.setVelocity(0.0)

        self._rightMotor = cast(Motor, self._robot.getDevice('right wheel motor'))
        self._rightMotor.setPosition(float('inf'))
        self._rightMotor.setVelocity(0.0)


        # =========================
        # ACTION SPACE
        # =========================

        # linear_velocity, angular_velocity
        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # =========================
        # OBSERVATION SPACE
        # =========================

        # red centrality, red proportion, 8 distance sensors, left_speed, right_speed
        self.observation_space = Box(
            low=np.array(
                [-1.0, 0.0] + ([0.0] * 8) + [-1.0, -1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0] + ([1.0] * 8) + [1.0, 1.0],
                dtype=np.float32,
            ),
        )

        self._episode_steps = 0
        self._max_episode_steps = 1000

    # =========================
    # RESET
    # =========================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._memory_reset()

        self._episode_steps = 0

        self._leftMotor.setVelocity(0.0)
        self._rightMotor.setVelocity(0.0)

        position, rotation = self._get_robot_position_and_rotate()

        self._robot_translation_field.setSFVec3f(position)
        self._robot_rotation_field.setSFRotation(rotation)
        self._robot_node.resetPhysics()

        for _ in range(2):
            self._robot.step(self._TIME_STEP)

        observation = self._get_observation()

        info = {
            "red_centrality": observation[0],
            "red_proportion": observation[1],
        }

        return observation, info

    # =========================
    # STEP
    # =========================

    def step(self, action):
        self._episode_steps += 1

        left_velocity, right_velocity = self._calculate_motor_velocity(action[0], action[1])

        self._leftMotor.setVelocity(left_velocity)
        self._rightMotor.setVelocity(right_velocity)

        for _ in range(self._FRAME_SKIP):
            self._robot.step(self._TIME_STEP)

        observation = self._get_observation()

        reward = self._compute_reward(action, observation)

        terminated = self._is_terminated(observation)

        truncated = self._episode_steps >= self._max_episode_steps

        info = {
            "red_centrality": observation[0],
            "red_proportion": observation[1],
            "left_velocity": left_velocity,
            "right_velocity": right_velocity,
        }

        return observation, reward, terminated, truncated, info

    # =========================
    # RENDER
    # =========================

    def render(self):
        pass

    # =========================
    # CLOSE
    # =========================

    def close(self):
        pass

    # =========================
    # OBSERVATIONS
    # =========================

    def _get_observation(self):
        sensor_values = []
        for name in self._DISTANCE_SENSOR_NAMES:
            sensor_value = self._distance_sensor[name].getValue()
            sensor_value = sensor_value / 4095.0 # normalize
            sensor_values.append(sensor_value)

        red_centrality, red_proportion = self._get_camera_observation()

        left_velocity = self._leftMotor.getVelocity() / self._MAX_VELOCITY
        right_velocity = self._rightMotor.getVelocity() / self._MAX_VELOCITY

        observations = [red_centrality, red_proportion] + sensor_values + [left_velocity, right_velocity]

        return np.array(observations, dtype=np.float32)

    # =========================
    # REWARD
    # =========================

    @abstractmethod
    def _compute_reward(self, action, observation):
        pass

    # =========================
    # TERMINATE
    # =========================

    @abstractmethod
    def _is_terminated(self, observation) -> bool:
        pass

    # =========================
    # POSITION AND ROTATE
    # =========================

    @abstractmethod
    def _get_robot_position_and_rotate(self):
        pass

    # =========================
    # MEMORY RESET
    # =========================

    def _memory_reset(self):
        pass

    # =========================
    # CAMERA PROCESSING
    # =========================

    def _get_camera_observation(self):
        height, width = (self._camera.getHeight(), self._camera.getWidth())

        total_pixels = height * width

        img = np.frombuffer(self._camera.getImage(), np.uint8).reshape((height, width, 4))
        mask = (
                (img[:, :, 2] > 130) &
                (img[:, :, 1] < 80) &
                (img[:, :, 0] < 80)
        )
        total_red_pixels = np.sum(mask)

        normal_total_red_pixels = total_red_pixels / total_pixels

        # red_pixels_left = np.sum(mask[:, :width // 2])
        # red_pixels_right = np.sum(mask[:, width // 2:])

        if total_red_pixels > 0:
            x_coords = np.where(mask)[1]
            centroid_x = np.mean(x_coords)
            half_width = width / 2
            centrality = (centroid_x - half_width) / half_width

        else:
            centrality = 0.0

        return centrality, normal_total_red_pixels

    def _calculate_motor_velocity(self, linear_velocity, angular_velocity):
        EPSILON = 1e-3
        max_velocity = self._MAX_VELOCITY - EPSILON

        linear_velocity *= 3.0
        angular_velocity *= 1.5

        left_velocity = np.clip(
            linear_velocity - angular_velocity,
            -max_velocity,
            max_velocity
        )

        right_velocity = np.clip(
            linear_velocity + angular_velocity,
            -max_velocity,
            max_velocity
        )

        return left_velocity, right_velocity

