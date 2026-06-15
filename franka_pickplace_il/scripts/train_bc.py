"""Wrapper around Isaac Lab's robomimic train.py that also registers our task.

Lab's `scripts/imitation_learning/robomimic/train.py` imports a fixed set of
`isaaclab_tasks` modules but doesn't know about our external `franka_pickplace`
package, so `gym.spec("Template-Franka-Pickplace-v0")` fails. This wrapper:

  1. Boots Kit (so `pxr` is available and our task package can import).
  2. Imports `franka_pickplace`, which triggers gym.register(...).
  3. Replaces `isaaclab.app.AppLauncher` with a no-op so train.py won't try to
     boot Kit a second time.
  4. Executes train.py as __main__, forwarding all CLI args.

Run exactly the same args you'd pass to train.py, e.g.:

    python scripts/train_bc.py \
        --task Template-Franka-Pickplace-v0 \
        --algo bc \
        --dataset ./datasets/pickplace_demos.hdf5
"""
from __future__ import annotations

import runpy
import sys

from isaaclab.app import AppLauncher

LAB_TRAIN_PY = (
    "/home/kyle/Desktop/IsaacLab/scripts/imitation_learning/robomimic/train.py"
)


def main() -> None:
    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app

    import franka_pickplace  # noqa: F401  registers Template-Franka-Pickplace-v0

    import isaaclab.app as app_mod

    class _AlreadyLaunched:
        """Stand-in so Lab's train.py doesn't relaunch Kit."""

        def __init__(self, *args, **kwargs) -> None:
            self._app = app

        @property
        def app(self):
            return self._app

    app_mod.AppLauncher = _AlreadyLaunched

    sys.argv[0] = LAB_TRAIN_PY
    runpy.run_path(LAB_TRAIN_PY, run_name="__main__")


if __name__ == "__main__":
    main()
