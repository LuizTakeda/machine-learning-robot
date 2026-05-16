# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Webots simulation that trains an e-puck robot to find and touch a red ball using Stable-Baselines3 SAC. Final artifact is a submission package (`submission/marcio_and_luiz_and_martin/`) that the competition evaluator loads as `Agent`.

## How to run things

Scripts under `controllers/main/` are **Webots controllers**, not standalone CLI programs. They `import controller` from the Webots Python API and only work when launched by Webots (or as `<extern>`). The bootstrap in each script (`_prepend_webots_controller_python_path`) tries Windows paths `D:\Webots\lib\controller\python` and `C:\Program Files\Webots\lib\controller\python`; set `WEBOTS_HOME` if Webots is elsewhere.

Training a phase: set the world's `DEF ROBOT E-puck` controller to the relevant `phase_*_initial_training.py` (or `phase_1_2_robot_random_position.py`, etc.) and press Play in Webots. The world used is `worlds/sample_lab.wbt`. Output models save as `*.zip` in the controller's working directory (which is `controllers/main/`).

Resuming/continuing training across phases:
```
python controllers/main/resume_training.py --phase 1.2 --checkpoint ./following_red_ball_model_phase_1_1.zip
python controllers/main/resume_training.py --phase 2 --timesteps 100000
```
`resume_training.py` is a thin dispatcher — it imports the phase module and calls its `run_training()`. Phase 2 / 1.3 / 2.5 accept replay-buffer flags; see `PHASE_SPECS` in that file.

Testing the submission end-to-end before zipping: point the .wbt's robot `controller` to `submission_test` and run. `controllers/submission_test/submission_test.py` drives `submission/marcio_and_luiz_and_martin/agent.py` through the **competition** 11-D obs / `[-1, 1]^2` action contract — NOT the internal 12-D layout. It expects the evaluator kit (`epuck_agent.py`, `camera_utils.py`) at `~/Downloads/Arquivo/`. NB: the script's `SUBMISSION_DIR` is hardcoded to `submission/marcio_gabriel` — update it to `marcio_and_luiz_and_martin` if you actually run it.

## Curriculum architecture (the part that needs reading multiple files)

The project is a **curriculum-learning pipeline** — each phase is a separate `RoombaRedBallEnv` (same class name, different module, different reset/reward logic) that loads the previous phase's SAC checkpoint and fine-tunes. Phase numbering matches `long_strategy.txt`:

- **Phase 1 / 1.1 / 1.2 / 1.3**: empty arena, progressively randomized robot pose (1 = fixed; 1.1 = random yaw; 1.2 = random position + yaw; 1.3 = corner-focused). Fixed red ball.
- **Phase 2 / 2.5**: static obstacles. 2.5 also randomizes goal position.
- **Phase 3 / 4**: dense randomized environments.

Default checkpoint chain (each phase loads the previous phase's `.zip`):
```
phase 1 -> following_red_ball_model
phase 1.1 -> following_red_ball_model_phase_1_1
phase 1.2 -> following_red_ball_model_phase_1_2
phase 2  -> obstacle_avoidance_following_red_ball_model
phase 2.5 -> randomized_following_red_ball_model
```
Checkpoint paths are resolved with both bare basename and `.zip` suffix (see e.g. `phase_1_2_robot_random_position.py:730`).

### Observation / action contract — DO NOT BREAK

The internal training env uses a **12-D observation**:
```
[0]   red_pixel_ratio          0..1
[1]   goal_horizontal_position -1..+1   (red-pixel centroid x, normalized)
[2]   linear_velocity          -MAX_SPEED..MAX_SPEED
[3]   angular_velocity         -MAX_SPEED..MAX_SPEED
[4-11] ps0..ps7 proximities    0..1     (each / 4095)
```
**Action is `(linear, angular)` in `[-MAX_SPEED, MAX_SPEED]^2`** (MAX_SPEED = 6.279, WHEEL_DISTANCE = 0.052). Conversion to wheel speeds uses standard differential-drive kinematics: `v_l = lin - ang*d/2`, `v_r = lin + ang*d/2`.

The competition evaluator's contract is **different**: 11-D obs `[ps0..ps7, red_detected, centroid_x_norm, red_pixel_ratio]` and action `(left_frac, right_frac) in [-1,1]^2`. `submission/.../agent.py` is the adapter that translates between them — velocity slots in the internal obs are filled with the last commanded values (a one-step lag, deliberate). The red-detection mask thresholds in `submission_test.py` (`r > 100, (r-g) > 50, (r-b) > 50`) differ from training (`r > 130, g < 80, b < 80`), because the evaluator's `camera_utils.is_red` differs from our training-time mask — keep both definitions intact when editing either side.

**Curriculum gotcha** (see `gameplan.txt`): the observation vector must stay constant across phases or `SAC.load` shape-mismatches. All 8 proximity sensors are already in phase 1 specifically so phase 2 can use them without a manual weight transfer.

### Reward shaping

Reward design is documented in `long_strategy.txt` and implemented in each phase's `step()`. Core terms (phase 1.2 has the most-developed version):

- step penalty `-0.05`, visibility bonus `red_ratio * 3`, delta-pixel reward `Δratio * 10`, centering bonus on `|prev_x| - |curr_x|`.
- Wall-avoidance gradient gated by `red_pixel_ratio` thresholds (tiered: stronger when "blind", reduced when ball is in view).
- Wedged-and-blind spin bonus + forward-ramming penalty to break corner traps.
- Truncation conditions: `STUCK_WALL_STEP_LIMIT`, `STEPS_NEAR_BALL_STALL_LIMIT` (anti-orbit), `BLIND_WALL_STUCK_LIMIT`, `STEPS_WITHOUT_BALL_LIMIT`, `MAX_STEPS_PER_EPISODE`.
- Goal = front-prox > contact threshold **AND** recent visibility (avoids "touch wall, get reward" false positives).

When editing reward terms, the dominant warning from `gameplan.txt` is **catastrophic forgetting**: keep approach-to-ball as the main signal; wall-avoidance must stay a penalty, never the dominant objective.

### Supervisor reset pattern

Each phase uses `Supervisor` (not just `Robot`) and grabs `DEF ROBOT` + `DEF GOAL` nodes by DEF name. `reset()` teleports via `translation_field.setSFVec3f` + `rotation_field.setSFRotation` then calls `robot_node.resetPhysics()` to zero inertia. `simulationSetMode(SIMULATION_MODE_FAST)` skips rendering for ~5-20x training speedup.

## Files worth knowing about

- `long_strategy.txt` — overall RL strategy, reward shaping rationale, source papers (EN + CS).
- `gameplan.txt` — curriculum-transfer warnings (obs consistency, replay-buffer policy, LR).
- `LSI project - Autonomous Robotics.pdf` — competition spec.
- `phase_*_trained_controller.py` — inference-only runtime controllers that load a saved `.zip` and drive the robot in Webots (no training, no Supervisor reset).
- `controllers/main/*.zip` and `*.pkl` are gitignored — checkpoints and replay buffers do not live in git.
