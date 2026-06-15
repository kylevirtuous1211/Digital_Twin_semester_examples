"""Custom termination terms."""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from . import constants


def task_success(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_threshold: float = constants.TASK_SUCCESS_POS_THRESHOLD,
    finger_open_threshold: float = constants.TASK_SUCCESS_FINGER_OPEN_THRESHOLD,
    min_hold_steps: int = constants.TASK_SUCCESS_HOLD_STEPS,
) -> torch.Tensor:
    """Episode succeeds when cube-at-goal AND gripper-open hold for N consecutive steps.

    Without the hold requirement the env terminates the instant the gripper opens
    over the pad, so demos only contain a few "gripper-open at goal" frames and
    BC underweights the release. Holding for N steps forces ~N more positive-
    gripper frames into every demo.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    cube_pos = cube.data.root_pos_w
    goal_pos = goal.data.root_pos_w
    cube_at_goal = torch.norm(cube_pos - goal_pos, dim=-1) < pos_threshold

    finger_ids = robot_cfg.joint_ids
    if finger_ids is None:
        finger_ids = [i for i, n in enumerate(robot.data.joint_names) if "finger" in n]
    finger_pos = robot.data.joint_pos[:, finger_ids].mean(dim=-1)
    gripper_open = finger_pos > finger_open_threshold

    instant_ok = cube_at_goal & gripper_open

    # Per-env "consecutive successful steps" counter stored on the env.
    if not hasattr(env, "_task_success_counter") or env._task_success_counter.shape[0] != env.num_envs:
        env._task_success_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # Reset counter for envs that just got reset (episode_length_buf == 0).
    just_reset = env.episode_length_buf == 0
    env._task_success_counter = torch.where(
        just_reset,
        torch.zeros_like(env._task_success_counter),
        env._task_success_counter,
    )

    # Increment on success, reset on failure.
    env._task_success_counter = torch.where(
        instant_ok,
        env._task_success_counter + 1,
        torch.zeros_like(env._task_success_counter),
    )

    return env._task_success_counter >= min_hold_steps
