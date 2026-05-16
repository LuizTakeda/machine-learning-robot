# TensorBoard

Each training phase (1, 1.1, 1.2, 1.3, 2, 2.5) logs episodic reward, episode
length, and SAC training metrics to `controllers/main/tb_logs/`. Subdirectories
are auto-indexed per run (`phase_1_2_1/`, `phase_1_2_2/`, ...), so re-training
does not overwrite previous logs.

## Run

```
pip install tensorboard
tensorboard --logdir tb_logs
```

Then open the URL printed by TensorBoard (typically `http://localhost:6006`).

## What you get

- `rollout/ep_rew_mean` — mean episodic reward over the last 100 episodes.
- `rollout/ep_len_mean` — mean episode length over the last 100 episodes.
- `train/actor_loss`, `train/critic_loss`, `train/ent_coef`, `train/learning_rate`
  — SAC internals, useful for diagnosing instability.

The episode metrics require the env to be wrapped in `stable_baselines3.common.monitor.Monitor`,
which is already done in every `phase_*_initial_training.py` / `phase_*_random_position.py`
/ `phase_*_corner_focused_training.py` / `phase_2*.py` script.
