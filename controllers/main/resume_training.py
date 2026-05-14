#!/usr/bin/env python3
"""
Continue Stable-Baselines3 SAC training for a chosen curriculum phase.

Run from the controller working directory Webots uses (where your .zip checkpoints live),
or pass absolute paths to --checkpoint.

Examples:
  python resume_training.py --phase 1.2 --checkpoint ./following_red_ball_model_phase_1_1.zip
  python resume_training.py --phase 2 --timesteps 100000
  python resume_training.py --phase 2.5 --checkpoint /path/to/obstacle_avoidance_following_red_ball_model.zip
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_MAIN_DIR = Path(__file__).resolve().parent
if str(_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_MAIN_DIR))

PHASE_SPECS: dict[str, dict] = {
    "1": {
        "module": "phase_1_initial_training",
        "default_timesteps": 70_000,
        "default_output": "following_red_ball_model",
    },
    "1.1": {
        "module": "phase_1_1_initial_training",
        "default_timesteps": 70_000,
        "default_output": "following_red_ball_model_phase_1_1",
    },
    "1.2": {
        "module": "phase_1_2_initial_training",
        "default_timesteps": 70_000,
        "default_output": "following_red_ball_model_phase_1_2",
    },
    "1.3": {
        "module": "phase_1_3_corner_focused_training",
        "default_timesteps": 70_000,
        "default_output": "following_red_ball_model_phase_1_3",
    },
    "2": {
        "module": "phase_2_training_with_obstacle",
        "default_timesteps": 300_000,
        "default_output": "obstacle_avoidance_following_red_ball_model",
    },
    "2.5": {
        "module": "phase_2_5_randomized_goal_and_robot",
        "default_timesteps": 300_000,
        "default_output": "randomized_following_red_ball_model",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume SAC training by phase (uses loaded phase module's env + hyperparameters).",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=list(PHASE_SPECS.keys()),
        help="Curriculum phase to train (must match an *_initial_training / phase_* script).",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=None,
        help="Path to policy checkpoint (.zip optional). Defaults: phase 1–2 use built-in paths; "
        "1.2 defaults to following_red_ball_model_phase_1_1 if omitted.",
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=None,
        help="Total env steps for this run (defaults per phase).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Basename for model.save (no .zip suffix).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Torch device for SAC (default: auto).",
    )
    parser.add_argument(
        "--no-replay-buffer",
        action="store_true",
        help="Phase 2 / 1.3: do not load any replay buffer.",
    )
    parser.add_argument(
        "--replay-buffer",
        default=None,
        help="Phase 2 / 1.3: explicit replay buffer .pkl to load first.",
    )
    parser.add_argument(
        "--no-save-replay-buffer",
        action="store_true",
        help="Phase 1.3: skip saving the replay buffer at the end of training.",
    )
    parser.add_argument(
        "--load-obstacle-replay",
        action="store_true",
        help="Phase 2.5 only: load obstacle_avoidance_following_red_ball_replay_buffer.pkl if present.",
    )
    args = parser.parse_args()

    spec = PHASE_SPECS[args.phase]
    mod = importlib.import_module(spec["module"])
    run_training = getattr(mod, "run_training", None)
    if run_training is None:
        raise RuntimeError(f"Module {spec['module']} has no run_training()")

    timesteps = args.timesteps if args.timesteps is not None else spec["default_timesteps"]
    output_path = args.output if args.output is not None else spec["default_output"]
    device = None if args.device == "auto" else args.device

    kwargs: dict = {
        "total_timesteps": timesteps,
        "output_path": output_path,
        "device": device,
    }

    if args.checkpoint is not None:
        kwargs["resume_path"] = args.checkpoint

    if args.phase == "2":
        kwargs["load_replay_buffer"] = not args.no_replay_buffer
        if args.replay_buffer:
            kwargs["replay_buffer_load_path"] = args.replay_buffer

    if args.phase == "2.5" and args.load_obstacle_replay:
        kwargs["load_replay_buffer"] = True

    if args.phase == "1.3":
        if args.no_replay_buffer:
            # signal to phase 1.3: pass an obviously-missing path so the loader skips it
            kwargs["replay_buffer_load_path"] = "__none__"
        elif args.replay_buffer:
            kwargs["replay_buffer_load_path"] = args.replay_buffer
        kwargs["save_replay_buffer"] = not args.no_save_replay_buffer

    print(f"Phase {args.phase} → {spec['module']}.run_training(**{kwargs})")
    run_training(**kwargs)


if __name__ == "__main__":
    main()
