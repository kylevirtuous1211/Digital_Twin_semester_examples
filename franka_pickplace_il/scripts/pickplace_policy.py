"""Scripted pick-place controller + Robomimic-format HDF5 dataset collector.

Drives the registered ``Template-Franka-Pickplace-v0`` env with a deterministic
state machine (APPROACH → GRASP → TRANSLATE → PLACE → RELEASE → DONE) and
streams successful episodes to disk in the layout expected by Isaac Lab's
``imitation_learning/robomimic/train.py``:

    data/
      env_args   (attr) {"env_name": "<task>", "type": 2}
      demo_N/
        actions  [T, 7]
        obs/
          eef_pos  [T, 3]
          eef_quat [T, 4]
          cube_pos [T, 3]
          cube_quat[T, 4]
          goal_pos [T, 3]
        num_samples (attr) = T
    mask/
      train ["demo_0", ...]
      valid ["demo_K", ...]

Run:
    python scripts/pickplace_policy.py --num_envs 16 --num_demos 200 \
        --dataset ./datasets/pickplace_demos.hdf5
"""
from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

# ───── CLI must be parsed BEFORE Isaac Sim imports ─────
parser = argparse.ArgumentParser(description="Collect Franka pick-place demos.")
parser.add_argument("--num_envs", type=int, default=8, help="Parallel envs.")
parser.add_argument("--num_demos", type=int, default=200, help="Successful demos to collect.")
parser.add_argument(
    "--dataset",
    type=str,
    default="./datasets/pickplace_demos.hdf5",
    help="HDF5 output path.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Template-Franka-Pickplace-v0",
    help="Registered task name (must match the bc.json env_args).",
)
parser.add_argument(
    "--valid_ratio",
    type=float,
    default=0.1,
    help="Fraction of demos reserved for the validation mask.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=400,
    help="Safety cap on per-episode buffer length to avoid runaway memory.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ───── Imports that need Isaac Sim running ─────
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import franka_pickplace  # noqa: F401,E402  registers the gym env


# ───── State machine ─────
# Phases match the canonical NVIDIA PickPlaceController: descend with gripper OPEN,
# settle, then close, then lift. Closing during descent makes fingers shut before
# they reach the cube and the grasp misses.
(_APPROACH, _DESCEND, _CLOSE, _LIFT, _TRANSLATE, _PLACE, _RELEASE, _DONE) = range(8)

APPROACH_OFFSET_Z = 0.10   # hover height above cube before grasping (m)
PLACE_OFFSET_Z = 0.10      # hover height above goal before lowering (m)
PLACE_DROP_Z = 0.04        # height to release cube above pad (m)
GRASP_DESCEND_Z = -0.005   # how far below cube center to descend
SETTLE_AND_CLOSE_STEPS = 25  # ticks holding closed (with gripper closing) before lifting
LIFT_STEPS = 6             # pure +Z lift steps before translating to goal — ~30 cm of upward intent, IK reaches ~10-15 cm before TRANSLATE
RELEASE_HOLD_STEPS = 80    # ticks holding open before declaring done — long, so BC sees many "gripper-open at goal" frames
POS_TOL = 0.025            # transition tolerance for waypoint arrival (m)
DELTA_CLAMP = 0.05         # max single-step linear delta (m); larger = faster motion


class PickPlacePolicy:
    """Per-environment state-machine controller producing IK delta poses."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = num_envs
        self.device = device
        self.states = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.hold_counter = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset_idx(self, env_ids) -> None:
        self.states[env_ids] = _APPROACH
        self.hold_counter[env_ids] = 0

    def reset(self) -> None:
        self.states[:] = _APPROACH
        self.hold_counter[:] = 0

    def compute(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        eef_pos = obs_dict["eef_pos"]    # [N, 3]
        cube_pos = obs_dict["cube_pos"]  # [N, 3]
        goal_pos = obs_dict["goal_pos"]  # [N, 3]

        N = eef_pos.shape[0]
        # actions: 3 pos delta + 3 axis-angle rotation + 1 gripper (-1 close, +1 open)
        actions = torch.zeros(N, 7, device=self.device)
        # rotation stays at zero — the scripted policy doesn't reorient the EE.

        for env_id in range(N):
            state = int(self.states[env_id].item())
            ee_p = eef_pos[env_id]
            cu_p = cube_pos[env_id]
            go_p = goal_pos[env_id]

            if state == _APPROACH:
                target = cu_p.clone()
                target[2] = cu_p[2] + APPROACH_OFFSET_Z
                rel = target - ee_p
                actions[env_id, 0:3] = rel
                actions[env_id, 6] = 1.0  # open
                if torch.norm(rel) < POS_TOL:
                    self.states[env_id] = _DESCEND

            elif state == _DESCEND:
                # Descend to cube WITH GRIPPER OPEN so fingers can engulf it.
                target = cu_p.clone()
                target[2] = cu_p[2] + GRASP_DESCEND_Z
                rel = target - ee_p
                actions[env_id, 0:3] = rel
                actions[env_id, 6] = 1.0  # still open
                if torch.norm(rel) < POS_TOL:
                    self.states[env_id] = _CLOSE

            elif state == _CLOSE:
                # Hold position; close gripper for SETTLE_AND_CLOSE_STEPS ticks.
                target = cu_p.clone()
                target[2] = cu_p[2] + GRASP_DESCEND_Z
                rel = target - ee_p
                actions[env_id, 0:3] = rel
                actions[env_id, 6] = -1.0  # close
                self.hold_counter[env_id] += 1
                if self.hold_counter[env_id] >= SETTLE_AND_CLOSE_STEPS:
                    self.hold_counter[env_id] = 0
                    self.states[env_id] = _LIFT

            elif state == _LIFT:
                # Position-based lift to TRANSLATE's start z (goal_z + PLACE_OFFSET_Z).
                # cube_pos tracks the gripper while held, so cube_xy ≈ ee_xy — using
                # it as the xy target makes rel_xy ≈ 0 (no lateral motion during lift).
                # The z component shrinks as EE rises → natural deceleration, no
                # velocity carry-over into TRANSLATE.
                target = cu_p.clone()
                target[2] = go_p[2] + PLACE_OFFSET_Z
                rel = target - ee_p
                actions[env_id, 0:3] = rel
                actions[env_id, 6] = -1.0
                if torch.norm(rel) < POS_TOL:
                    self.states[env_id] = _TRANSLATE

            elif state == _TRANSLATE:
                target = go_p.clone()
                target[2] = go_p[2] + PLACE_OFFSET_Z
                rel = target - ee_p
                actions[env_id, 0:3] = rel
                actions[env_id, 6] = -1.0
                if torch.norm(rel) < POS_TOL:
                    self.states[env_id] = _PLACE

            elif state == _PLACE:
                target = go_p.clone()
                target[2] = go_p[2] + PLACE_DROP_Z
                rel = target - ee_p
                actions[env_id, 0:3] = rel
                actions[env_id, 6] = -1.0
                if torch.norm(rel) < POS_TOL:
                    self.states[env_id] = _RELEASE

            elif state == _RELEASE:
                actions[env_id, 6] = 1.0
                self.hold_counter[env_id] += 1
                if self.hold_counter[env_id] >= RELEASE_HOLD_STEPS:
                    self.hold_counter[env_id] = 0
                    self.states[env_id] = _DONE

            else:  # _DONE — hold open, wait for env to terminate
                actions[env_id, 6] = 1.0

        # Norm-based clamp: scale the position delta so its L2 magnitude is at
        # most DELTA_CLAMP. Preserves direction (per-axis clamping caused the
        # arm to overshoot the short axis and bounce back). When far from the
        # target this commands a full-magnitude step in the correct direction;
        # when within DELTA_CLAMP it naturally tapers to zero.
        pos_delta = actions[:, 0:3]
        norms = torch.linalg.vector_norm(pos_delta, dim=-1, keepdim=True)
        scale = torch.where(
            norms > DELTA_CLAMP,
            DELTA_CLAMP / norms.clamp(min=1e-9),
            torch.ones_like(norms),
        )
        actions[:, 0:3] = pos_delta * scale
        return actions


def _flush_episode(
    data_grp: h5py.Group,
    saved_idx: int,
    steps: list[dict],
) -> None:
    ep = data_grp.create_group(f"demo_{saved_idx}")
    ep.create_dataset(
        "actions",
        data=np.stack([s["action"] for s in steps], axis=0),
    )
    obs_grp = ep.create_group("obs")
    for key in steps[0]["obs"].keys():
        obs_grp.create_dataset(
            key,
            data=np.stack([s["obs"][key] for s in steps], axis=0),
        )
    ep.attrs["num_samples"] = len(steps)


def run(env, policy: PickPlacePolicy, dataset_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)

    buffers: list[list[dict]] = [[] for _ in range(env.num_envs)]
    saved = 0
    attempted = 0

    with h5py.File(dataset_path, "w") as f:
        data = f.create_group("data")
        # type=2 tells robomimic this is a gymnasium env, not MuJoCo (the default).
        data.attrs["env_args"] = json.dumps({"env_name": args_cli.task, "type": 2})

        obs_dict, _ = env.reset()
        policy.reset()

        step_counter = 0
        phase_names = ["APP", "DSC", "CLS", "LFT", "TRN", "PLC", "REL", "DON"]
        while saved < args_cli.num_demos:
            with torch.inference_mode():
                actions = policy.compute(obs_dict["policy"])
                next_obs_dict, _, terminated, truncated, _ = env.step(actions)

            step_counter += 1
            if step_counter % 60 == 0:
                hist = torch.bincount(policy.states, minlength=8).tolist()
                tagged = " ".join(f"{n}={c}" for n, c in zip(phase_names, hist))
                # Diagnostic: cube-goal distance for envs in DONE so we can see if placement is close.
                done_envs = (policy.states == 7).nonzero(as_tuple=False).flatten().tolist()
                diag = ""
                if done_envs:
                    cube = obs_dict["policy"]["cube_pos"][done_envs]
                    goal = obs_dict["policy"]["goal_pos"][done_envs]
                    dists = torch.norm(cube - goal, dim=-1).cpu().tolist()
                    diag = "  cube-goal-dist=" + ",".join(f"{d:.3f}" for d in dists[:5])
                print(f"[step {step_counter}] saved={saved} attempts={attempted}  {tagged}{diag}", flush=True)

            actions_np = actions.cpu().numpy()
            obs_np = {k: v.cpu().numpy() for k, v in obs_dict["policy"].items()}

            for i in range(env.num_envs):
                if len(buffers[i]) < args_cli.max_steps:
                    buffers[i].append({
                        "obs": {k: obs_np[k][i] for k in obs_np},
                        "action": actions_np[i],
                    })

            done_mask = terminated | truncated
            done_ids = done_mask.nonzero(as_tuple=False).flatten().tolist()
            for i in done_ids:
                attempted += 1
                success = bool(terminated[i].item())
                if success and saved < args_cli.num_demos:
                    _flush_episode(data, saved, buffers[i])
                    saved += 1
                    print(
                        f"[saved {saved}/{args_cli.num_demos}] env={i} "
                        f"len={len(buffers[i])}  attempts={attempted}"
                    )
                buffers[i] = []
                policy.reset_idx([i])

            obs_dict = next_obs_dict

        # train / valid mask required by robomimic
        all_demos = [f"demo_{i}" for i in range(saved)]
        n_valid = max(1, int(saved * args_cli.valid_ratio))
        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.array(all_demos[:-n_valid], dtype=object))
        mask.create_dataset("valid", data=np.array(all_demos[-n_valid:], dtype=object))

    print(
        f"\nDone. {saved} demos saved to {dataset_path}  "
        f"(train={saved - n_valid}, valid={n_valid}, attempts={attempted})"
    )


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # Keep obs as a named dict so we can index by key (eef_pos, cube_pos, ...).
    env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    policy = PickPlacePolicy(num_envs=env.num_envs, device=env.device)

    try:
        run(env, policy, args_cli.dataset)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
