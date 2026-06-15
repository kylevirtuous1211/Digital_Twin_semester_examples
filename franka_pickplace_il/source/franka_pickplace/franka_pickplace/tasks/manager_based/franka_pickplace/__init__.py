"""Gymnasium registration for the Franka pick-place IL task."""
import gymnasium as gym

from . import agents
from .franka_pickplace_env_cfg import FrankaPickplaceEnvCfg

gym.register(
    id="Template-Franka-Pickplace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_pickplace_env_cfg:FrankaPickplaceEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc.json",
    },
)
