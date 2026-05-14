"""
Submission for the e-puck red-ball competition.

The underlying SAC policy was trained against a slightly different observation /
action layout than the one the evaluator uses, so this Agent acts as a thin
translation layer:

    competition obs (11,)          -> our 12-D obs
        [ps0..ps7,                     [red_pixel_ratio,
         red_detected,                  centroid_x_norm,
         centroid_x_norm,               last_linear_velocity,
         red_pixel_ratio]               last_angular_velocity,
                                        ps0..ps7]

    SAC action (lin, ang) in       -> competition action
    [-MAX_SPEED, MAX_SPEED]^2          (left_frac, right_frac) in [-1, 1]^2
                                       via differential-drive kinematics
"""

import pathlib

import numpy as np
from stable_baselines3 import SAC

from epuck_agent import EpuckAgent


MAX_SPEED = 6.279        # e-puck max wheel angular velocity (rad/s)
WHEEL_DISTANCE = 0.052   # distance between e-puck wheels (m)


class Agent(EpuckAgent):
    """SAC policy with an obs/action adapter for the competition interface."""

    def load(self, model_dir: str) -> None:
        model_path = pathlib.Path(model_dir) / "policy"
        self.model = SAC.load(str(model_path), device="cpu")
        self._last_linear_velocity = 0.0
        self._last_angular_velocity = 0.0

    def act(self, obs: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        proximity = obs[0:8]
        centroid_x_norm = float(obs[9])
        red_pixel_ratio = float(obs[10])

        # Reassemble the 12-D observation the SAC policy was trained on.
        # Velocity slots are filled with the previous commanded values (one-step
        # lag is harmless — those slots were never strong drivers of the policy).
        internal_obs = np.empty(12, dtype=np.float32)
        internal_obs[0] = red_pixel_ratio
        internal_obs[1] = centroid_x_norm
        internal_obs[2] = self._last_linear_velocity
        internal_obs[3] = self._last_angular_velocity
        internal_obs[4:12] = proximity

        action, _ = self.model.predict(internal_obs, deterministic=True)
        linear_velocity = float(action[0])
        angular_velocity = float(action[1])

        self._last_linear_velocity = linear_velocity
        self._last_angular_velocity = angular_velocity

        # Differential drive -> per-wheel speeds, then normalise to [-1, 1].
        left_speed = linear_velocity - angular_velocity * WHEEL_DISTANCE / 2.0
        right_speed = linear_velocity + angular_velocity * WHEEL_DISTANCE / 2.0
        left_frac = np.clip(left_speed / MAX_SPEED, -1.0, 1.0)
        right_frac = np.clip(right_speed / MAX_SPEED, -1.0, 1.0)

        return np.array([left_frac, right_frac], dtype=np.float32)
