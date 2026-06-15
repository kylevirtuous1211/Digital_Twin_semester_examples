"""Self-contained BC rollout for Template-Franka-Pickplace-v0.

Replaces Lab's `scripts/imitation_learning/robomimic/play.py`. Differences:

* Imports `franka_pickplace` so the gym env is registered.
* Reads per-dimension action min/max from the dataset's `/data` attrs (written
  by `scripts/perdim_normalize_hdf5.py`) and un-normalizes each action axis
  independently. Lab's play.py only supports a scalar min/max which doesn't
  fit our [pos × 3, axis-angle × 3, gripper × 1] action layout.

Usage:
    python scripts/play_bc.py \
        --task Template-Franka-Pickplace-v0 \
        --num_rollouts 50 \
        --checkpoint <path/to/best.pth> \
        --norm_dataset ./datasets/pickplace_demos_perdim_norm.hdf5
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# ───── CLI must be parsed BEFORE Isaac Sim imports ─────
parser = argparse.ArgumentParser(description="Play BC policy on franka_pickplace.")
parser.add_argument("--task", type=str, default="Template-Franka-Pickplace-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_rollouts", type=int, default=50)
parser.add_argument("--horizon", type=int, default=1200)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument(
    "--norm_dataset",
    type=str,
    default=None,
    help="Dataset HDF5 with per-dim min/max attrs on /data. If unset, no unnormalization.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ───── Imports that need Kit booted ─────
import copy  # noqa: E402

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
import torch  # noqa: E402

import franka_pickplace  # noqa: F401, E402  registers Template-Franka-Pickplace-v0
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def load_perdim_params(dataset_path: str | None):
    """Read per-dimension min/max written by perdim_normalize_hdf5.py."""
    if dataset_path is None:
        return None, None
    with h5py.File(dataset_path, "r") as f:
        if "perdim_action_min" not in f["data"].attrs:
            return None, None
        a_min = np.asarray(f["data"].attrs["perdim_action_min"], dtype=np.float32)
        a_max = np.asarray(f["data"].attrs["perdim_action_max"], dtype=np.float32)
    return a_min, a_max


def unnormalize(actions: np.ndarray, a_min: np.ndarray, a_max: np.ndarray) -> np.ndarray:
    span = np.where(a_max - a_min > 1e-12, a_max - a_min, 1.0)
    return (actions + 1.0) * span / 2.0 + a_min


RELEASE_RADIUS_XY = 0.08  # m — how close cube xy must be to goal xy to trigger override
RELEASE_LIFT_MIN = 0.02   # m — cube must be at least this high above goal z (i.e. lifted)


def rollout(policy, env, success_term, horizon, device, a_min, a_max):
    policy.start_episode()
    obs_dict, _ = env.reset()
    for _ in range(horizon):
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
        actions = policy(obs)
        if a_min is not None:
            actions = unnormalize(actions, a_min, a_max)

        # Inference-time release heuristic: BC underweights the rare gripper-open
        # frames, so the policy parks the cube over the pad without releasing.
        # If the cube is held above and near the goal pad, force gripper open.
        cube_p = obs["cube_pos"].detach().cpu().numpy()
        goal_p = obs["goal_pos"].detach().cpu().numpy()
        xy_dist = float(np.linalg.norm(cube_p[:2] - goal_p[:2]))
        z_lift = float(cube_p[2] - goal_p[2])
        if xy_dist < RELEASE_RADIUS_XY and z_lift > RELEASE_LIFT_MIN:
            actions[..., 6] = 1.0  # open gripper

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        if bool(success_term.func(env, **success_term.params)[0]):
            return True
        if terminated or truncated:
            return False
    return False


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    device = TorchUtils.get_torch_device(try_to_use_cuda=args_cli.device != "cpu")
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args_cli.checkpoint, device=device, verbose=True
    )

    a_min, a_max = load_perdim_params(args_cli.norm_dataset)
    if a_min is not None:
        print(f"[INFO] Using per-dim un-norm: min={a_min.tolist()}  max={a_max.tolist()}")
    else:
        print("[INFO] No per-dim un-norm applied (raw action output).")

    results: list[bool] = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}", flush=True)
        ok = rollout(policy, env, success_term, args_cli.horizon, device, a_min, a_max)
        results.append(ok)
        print(f"[INFO] Trial {trial}: {ok}\n", flush=True)

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
