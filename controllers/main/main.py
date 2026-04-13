from controller import Supervisor
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import cv2

TIME_STEP = 64
MAX_SPEED = 6.279
REACH_THRESHOLD = 0.15
ACTION_REPEAT = 8
MAX_STEPS_PER_EPISODE = 500

BALL_POS = np.array([1.03, 0.0])
ARENA_X = (-1.1, 0.8)
ARENA_Y = (-0.6, 0.6)
MIN_DIST_FROM_BALL = 0.3

robot = Supervisor()

robot_node = robot.getFromDef("ROBOT")
robot_translation = robot_node.getField("translation")
robot_rotation = robot_node.getField("rotation")

camera = robot.getDevice('camera')
camera.enable(TIME_STEP)

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)


class RedBallEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        self.action_space = spaces.Box(
            low=-MAX_SPEED, high=MAX_SPEED, shape=(2,), dtype=np.float32
        )

        # ball_x (-1..1), ball_size (0..1), left_speed (-1..1), right_speed (-1..1)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        self.render_mode = render_mode
        self.current_speeds = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_ball_size = 0.0
        self.steps = 0

    def _random_spawn(self):
        """Pick a random position inside the arena, facing roughly toward the ball."""
        while True:
            x = self.np_random.uniform(*ARENA_X)
            y = self.np_random.uniform(*ARENA_Y)
            if np.linalg.norm(np.array([x, y]) - BALL_POS) > MIN_DIST_FROM_BALL:
                break
        angle_to_ball = np.arctan2(BALL_POS[1] - y, BALL_POS[0] - x)
        angle = angle_to_ball + self.np_random.uniform(-np.pi / 3, np.pi / 3)
        return [x, y, 0.0], [0, 0, 1, angle]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)

        pos, rot = self._random_spawn()
        robot_translation.setSFVec3f(pos)
        robot_rotation.setSFRotation(rot)
        robot_node.resetPhysics()

        robot.step(TIME_STEP)

        self.current_speeds = np.array([0.0, 0.0], dtype=np.float32)
        self.steps = 0

        ball_x, ball_size = self._detect_ball()
        self.prev_ball_size = ball_size
        observation = np.array(
            [ball_x, ball_size, 0.0, 0.0], dtype=np.float32
        )
        return observation, {}

    def _detect_ball(self):
        """Return (ball_x, ball_size) from the camera image."""
        raw_image = camera.getImage()
        if not raw_image:
            return 0.0, 0.0

        width = camera.getWidth()
        height = camera.getHeight()
        image = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))

        bgr = image[:, :, :3]
        mask = (bgr[:, :, 2] > 130) & (bgr[:, :, 1] < 80) & (bgr[:, :, 0] < 80)

        total_pixels = width * height
        count = int(np.count_nonzero(mask))
        ball_size = count / total_pixels

        if count == 0:
            ball_x = 0.0
        else:
            cols = np.where(mask)[1]
            centroid_x = np.mean(cols)
            ball_x = (centroid_x / width) * 2.0 - 1.0  # normalize to -1..1

        cv2.imshow("Robot view red", mask.astype(np.uint8) * 255)
        cv2.waitKey(1)

        return float(ball_x), float(ball_size)

    def step(self, action):
        left_vel = float(np.clip(action[0], -MAX_SPEED, MAX_SPEED))
        right_vel = float(np.clip(action[1], -MAX_SPEED, MAX_SPEED))
        leftMotor.setVelocity(left_vel)
        rightMotor.setVelocity(right_vel)

        for _ in range(ACTION_REPEAT):
            robot.step(TIME_STEP)

        self.steps += 1
        self.current_speeds = np.array(
            [left_vel / MAX_SPEED, right_vel / MAX_SPEED], dtype=np.float32
        )

        ball_x, ball_size = self._detect_ball()

        approach = (ball_size - self.prev_ball_size) * 500.0
        self.prev_ball_size = ball_size

        turn_rate = abs(right_vel - left_vel) / (2.0 * MAX_SPEED)

        if ball_size > 0:
            centering = 1.0 - abs(ball_x)

            signed_turn = (right_vel - left_vel) / (2.0 * MAX_SPEED)
            steering = -ball_x * signed_turn * 2.0

            reward = approach + centering + steering
        else:
            reward = -1.0 + turn_rate * 0.5

        terminated = ball_size > REACH_THRESHOLD
        if terminated:
            reward += 10.0

        truncated = self.steps >= MAX_STEPS_PER_EPISODE

        observation = np.array(
            [ball_x, ball_size, self.current_speeds[0], self.current_speeds[1]],
            dtype=np.float32,
        )

        return observation, reward, terminated, truncated, {}

    def close(self):
        cv2.destroyAllWindows()


environment = RedBallEnv()

import os
MODEL_PATH = "following_red_ball_model"

if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH, env=environment)
    print(f"Loaded existing model from {MODEL_PATH}.zip — continuing training")
else:
    model = PPO("MlpPolicy", environment, verbose=1)
    print("No saved model found — starting fresh")

model.learn(total_timesteps=50000)
model.save(MODEL_PATH)
