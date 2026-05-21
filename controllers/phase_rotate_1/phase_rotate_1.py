import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch

from core.base_env import BaseEnv

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

class PhaseRotateEnv(BaseEnv):

    # =========================
    # INIT
    # =========================

    def __init__(self):
        super().__init__()

        self._goal_found = False

    # =========================
    # MEMORY RESET
    # =========================

    def _memory_reset(self):
        self._goal_found = False

    # =========================
    # GET POSITION AND ROTATE
    # =========================

    def _get_robot_position_and_rotate(self):
        position = [-1, 0, 0]
        rotation = [0, 1, 0, 0]

        return position, rotation

    # =========================
    # REWARD
    # =========================

    def _compute_reward(self, action, observation):
        red_centrality = observation[0]
        red_proportion = observation[1]

        reward = 0

        reward -= (action[1] ** 2) * 0.05

        if red_proportion <= 0.05:
            reward -= 0.5
            if self._goal_found:
                self._goal_found = False
                reward -= 5.0
        else:
            self._goal_found = True
            reward += red_proportion * 5.0
            reward -= (red_centrality ** 2) * 3.0

        if observation[10] < 0 and observation[11] < 0:
            reward -= 0.25

        if self._is_success(observation):
            reward += 10.0

        return float(reward)

    # =========================
    # TERMINATE
    # =========================

    def _is_terminated(self, observation):
        return self._is_success(observation)

    # =========================
    # AUXILIARY
    # =========================

    def _is_success(self, observation):
        # red_centrality = observation[0]
        red_proportion = observation[1]

        front_distance = max(
            observation[2],
            observation[9]
        )

        return red_proportion > 0.5 and front_distance < 0.15


if __name__ == '__main__':
    CUDA_AVAILABLE = torch.cuda.is_available()

    print(f"CUDA available: {CUDA_AVAILABLE}")

    environment = Monitor(
        PhaseRotateEnv(),
        info_keywords=(
            "red_centrality",
            "red_proportion",
            "left_velocity",
            "right_velocity",
        ),
    )

    device = "cuda" if CUDA_AVAILABLE else "cpu"

    model = SAC(
        "MlpPolicy",
        environment,
        device=device,
        verbose=1,
        tensorboard_log="./tensorboard/",
        learning_starts=1000,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=200_000, tb_log_name="phase_rotate")
    model.save("phase-rotate-1")

    print("Model saved to phase-rotate-1.zip")