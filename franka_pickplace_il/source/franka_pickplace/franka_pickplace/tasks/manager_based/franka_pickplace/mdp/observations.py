"""Custom observation terms exposed to the policy."""
from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def ee_frame_pos(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position in env frame (world pos minus env origin). Shape [N, 3]."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    return ee_pos_w - env.scene.env_origins


def ee_frame_quat(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector orientation quaternion (w, x, y, z) in world frame. Shape [N, 4]."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


def object_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Object position in env frame. Shape [N, 3]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :3] - env.scene.env_origins


def object_quat_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Object orientation quaternion (w, x, y, z) in world frame. Shape [N, 4]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_quat_w
