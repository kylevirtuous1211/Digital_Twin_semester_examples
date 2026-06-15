"""Manager-based RL env config for the Franka pick-and-place IL task.

Mirrors the seven-step build from the NVIDIA IL hands-on slides:
  scene, actions, observations, events, rewards, terminations.

The env is shaped for IL data collection — the reward is a placeholder
required by ManagerBasedRLEnvCfg; only termination flags drive demos.
"""
from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from . import mdp


# ───────────────────────── Scene ─────────────────────────
@configclass
class FrankaPickplaceSceneCfg(InteractiveSceneCfg):
    """Franka + green cube (pick) + red region (place) on a ground plane."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # 0.107 m offset matches the IK controller's body_offset below.
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.107]),
            ),
        ],
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.05, 0.02)),
    )

    # Kinematic so the place pad doesn't drift when nudged.
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.05, 0.001)),
    )


# ───────────────────────── Actions ─────────────────────────
@configclass
class ActionsCfg:
    """Arm = relative-pose IK delta; gripper = binary open/close."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.107]
        ),
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


# ───────────────────────── Observations ─────────────────────────
@configclass
class ObservationsCfg:
    """Low-dim state shared by the scripted policy and the BC student."""

    @configclass
    class PolicyCfg(ObsGroup):
        eef_pos = ObsTerm(
            func=mdp.ee_frame_pos,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        eef_quat = ObsTerm(
            func=mdp.ee_frame_quat,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        cube_pos = ObsTerm(
            func=mdp.object_pos_in_env_frame,
            params={"asset_cfg": SceneEntityCfg("cube")},
        )
        cube_quat = ObsTerm(
            func=mdp.object_quat_w,
            params={"asset_cfg": SceneEntityCfg("cube")},
        )
        goal_pos = ObsTerm(
            func=mdp.object_pos_in_env_frame,
            params={"asset_cfg": SceneEntityCfg("target")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ───────────────────────── Events (reset + domain randomization) ─────────────────────────
@configclass
class EventCfg:
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    # Small joint noise so demos don't all start from identical state.
    reset_robot_joints_noise = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "position_range": (-0.025, 0.025),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_cube_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.4, -0.2),
                "yaw": (-math.pi / 4, math.pi / 4),
            },
            "velocity_range": {},
        },
    )

    reset_target_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (0.2, 0.4),
            },
            "velocity_range": {},
        },
    )


# ───────────────────────── Rewards ─────────────────────────
@configclass
class RewardsCfg:
    """Minimal reward placeholder required by ManagerBasedRLEnvCfg."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)


# ───────────────────────── Terminations ─────────────────────────
@configclass
class TerminationsCfg:
    """`success` is the only non-timeout term — robomimic uses it as the
    rollout success signal in play.py."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.task_success,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
            "pos_threshold": mdp.constants.TASK_SUCCESS_POS_THRESHOLD,
            "finger_open_threshold": mdp.constants.TASK_SUCCESS_FINGER_OPEN_THRESHOLD,
        },
    )


# ───────────────────────── Top-level env ─────────────────────────
@configclass
class FrankaPickplaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: FrankaPickplaceSceneCfg = FrankaPickplaceSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
