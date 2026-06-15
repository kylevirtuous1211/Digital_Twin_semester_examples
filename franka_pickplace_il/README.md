# Franka Pick-Place — Imitation Learning Pipeline

End-to-end Isaac Lab project traced from `examples/docs/IL_hand_on.pdf`. A
scripted state-machine teacher generates demonstrations in parallel envs,
serializes them to a Robomimic-formatted HDF5 dataset, and trains a
Behavioral Cloning student that can be replayed in Isaac Sim.

```
Configure Env → Build Data Generator → Save Dataset → Train BC → Validate
```

## Layout

```
franka_pickplace_il/
├── scripts/
│   └── pickplace_policy.py            # state machine + HDF5 collector
└── source/franka_pickplace/
    ├── pyproject.toml
    └── franka_pickplace/
        └── tasks/manager_based/franka_pickplace/
            ├── __init__.py            # gym.register("Template-Franka-Pickplace-v0")
            ├── franka_pickplace_env_cfg.py
            ├── mdp/
            │   ├── constants.py       # success thresholds
            │   ├── observations.py    # eef/cube/goal pose terms
            │   └── terminations.py    # task_success (cube_at_goal & gripper_open)
            └── agents/robomimic/
                └── bc.json            # BC training config
```

## Prerequisites

Isaac Lab installed on the host (the Docker container in this repo is Isaac
Sim only). Robomimic for training:

```bash
sudo apt install cmake build-essential
./isaaclab.sh -i robomimic
```

## Install

From the host, with the Isaac Lab conda env active:

```bash
cd franka_pickplace_il
python -m pip install -e source/franka_pickplace
```

Verify the task registered:

```bash
python -c "import franka_pickplace, gymnasium as gym; \
    print('Template-Franka-Pickplace-v0' in gym.envs.registry)"
```

## 1. Collect demos

```bash
python scripts/pickplace_policy.py \
    --num_envs 16 \
    --num_demos 200 \
    --dataset ./datasets/pickplace_demos.hdf5
```

Inspect:

```bash
pip install hdf5view pyqt5
hdf5view -f ./datasets/pickplace_demos.hdf5
```

Expected structure:

```
data/
  env_args (attr) {"env_name": "Template-Franka-Pickplace-v0", "type": 2}
  demo_0/
    actions       [T, 7]
    obs/{eef_pos, eef_quat, cube_pos, cube_quat, goal_pos}
    num_samples   (attr) = T
  demo_1/ ...
mask/
  train [...]
  valid [...]
```

## 2. Train BC

```bash
python /PATH/TO/ISAACLAB/scripts/imitation_learning/robomimic/train.py \
    --task Template-Franka-Pickplace-v0 \
    --algo bc \
    --dataset ./datasets/pickplace_demos.hdf5
```

Monitor:

```bash
python -m tensorboard.main --logdir logs
```

~7000 epochs ≈ 1 hour on an RTX 4090 (<8 GB VRAM).

## 3. Roll out the trained policy

```bash
python /PATH/TO/ISAACLAB/scripts/imitation_learning/robomimic/play.py \
    --device cpu \
    --task Template-Franka-Pickplace-v0 \
    --num_rollouts 50 \
    --checkpoint /PATH/TO/CHECKPOINT.pth
```

## Notes / deviations from the slides

- File is `franka_pickplace_env_cfg.py`, not `env_cfg.py` — matches the
  `entry_point` string from the slide's `gym.register` snippet.
- A `LIFT` waypoint is inserted between `GRASP` and `TRANSLATE`; without it
  the EE drags the cube through the goal pad's edge on most resets.
- `task_success` is gated on `terminated[i]` only — `truncated` (timeout)
  episodes are discarded. The slides treat both as "done" but only flush
  successful ones.
- `num_envs` defaults to 8 (slides show 10); raise as VRAM permits — episodes
  run independently and the dataset is filled out-of-order.
