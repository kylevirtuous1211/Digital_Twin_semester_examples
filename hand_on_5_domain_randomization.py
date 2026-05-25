# ============================================================
# Hand-on 5: Replicator Domain Randomization
# Scatters 6 YCB props around a Franka, runs pick-and-place on
# a blue cube, and captures RGB + semantic + instance
# segmentation to /workspace/output (host: ./output/).
# Run via: ./run_in_isaac.py examples/hand_on_5_domain_randomization.py --wait
# ============================================================
import os
import random
import numpy as np

import carb
import omni.kit.app
import omni.timeline
import omni.replicator.core as rep

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.semantics import add_labels
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from omni.physx.scripts.utils import setRigidBody
from pxr import Gf, UsdGeom

# Re-runs in the same Kit instance must start from a clean World
if World.instance() is not None:
    World.clear_instance()

# ── Config ──────────────────────────────────────────────────────
SCATTER_ROOT     = "/World/ScatterProps"
DATA_SAVE_PATH   = "/workspace/output"
STAGE_FPS        = 30
CAPTURE_EVERY_N  = 100            # capture every Nth outer tick (~12 Hz at 60 FPS)
CUBE_SIZE        = 0.0515
CUBE_HALF        = CUBE_SIZE / 2.0
PLACE_TARGET     = np.array([0.4, -0.3, CUBE_HALF])
RESOLUTION       = (512, 512)
SEED             = 42           # fixed seed for reproducible scatter
CAPTURE_ENABLED  = True         # set False to disable orchestrator.step (debug pick-place)
SCATTER_RADIUS_MIN = 0.45       # min XY distance from Franka (m)
SCATTER_RADIUS_MAX = 1.0        # max XY distance from Franka (m)
SCATTER_MIN_SEP   = 0.18        # min XY distance between any two props (m)
SCATTER_Z         = 0.5         # spawn height — physics drops props onto ground

# YCB props (Axis_Aligned_Physics variant ships with rigid-body + convex-hull
# colliders baked in — no manual physics-API setup needed).
YCB = [
    ("002_master_chef_can",  "master_chef_can"),
    ("003_cracker_box",      "cracker_box"),
    ("004_sugar_box",        "sugar_box"),
    ("005_tomato_soup_can",  "tomato_soup_can"),
    ("006_mustard_bottle",   "mustard_bottle"),
    ("011_banana",           "banana"),
]


# ── BaseSample lifecycle ────────────────────────────────────────
class DomainRandomizationDemo(BaseSample):
    def setup_scene(self):
        """Build the scene programmatically: ground, Franka, 6 YCB props
        under /World/ScatterProps/, plus the blue pick-target cube outside
        that group so it isn't scattered."""
        world = self.get_world()
        world.scene.add_default_ground_plane()
        world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, SCATTER_ROOT)
        root = get_assets_root_path()

        # Sample non-overlapping spawn positions in an annulus around the robot.
        # Reproducible via SEED; replaces Replicator's scatter_2d (which silently
        # failed to fire in our running-Kit context).
        rng = random.Random(SEED)
        positions = []
        for _ in YCB:
            for _attempt in range(200):
                angle = rng.uniform(-np.pi, np.pi)
                radius = rng.uniform(SCATTER_RADIUS_MIN, SCATTER_RADIUS_MAX)
                x, y = radius * np.cos(angle), radius * np.sin(angle)
                if all((x - px)**2 + (y - py)**2 >= SCATTER_MIN_SEP**2
                       for px, py in positions):
                    positions.append((x, y))
                    break
            else:
                # Fallback: fan out along +x if rejection sampling fails
                positions.append((SCATTER_RADIUS_MAX, 0.2 * len(positions)))

        for (usd_name, label), (x, y) in zip(YCB, positions):
            prim_path = f"{SCATTER_ROOT}/_{usd_name}"
            add_reference_to_stage(
                usd_path=f"{root}/Isaac/Props/YCB/Axis_Aligned/{usd_name}.usd",
                prim_path=prim_path,
            )
            prim = stage.GetPrimAtPath(prim_path)

            # Set initial position via xformOp:translate (random, non-overlapping).
            xformable = UsdGeom.Xformable(prim)
            translate_op = next(
                (op for op in xformable.GetOrderedXformOps()
                 if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
                None,
            )
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(float(x), float(y), SCATTER_Z))

            # Canonical Isaac Sim physics setup. setRigidBody applies:
            #   - UsdPhysics.RigidBodyAPI on the parent
            #   - UsdPhysics.CollisionAPI on the parent
            #   - UsdPhysics.MeshCollisionAPI(approximation="convexHull") on
            #     each descendant Mesh, BEFORE PhysX parses the stage.
            # This is what eliminates the "triangle mesh ... cannot be a part
            # of a dynamic body" warning (which was invalidating the physics
            # tensor view and breaking get_world_pose() in physics_step).
            setRigidBody(prim, approximationShape="convexHull", kinematic=False)

            add_labels(prim, labels=[label], instance_name="class")

        add_labels(
            stage.GetPrimAtPath("/World/Franka"),
            labels=["franka"],
            instance_name="class",
        )

        cube = world.scene.add(DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=np.array([0.3, 0.3, 0.3]),
            scale=np.array([CUBE_SIZE] * 3),
            color=np.array([0.0, 0.0, 1.0]),
        ))
        add_labels(cube.prim, labels=["cube"], instance_name="class")

    async def setup_post_load(self):
        """Configure Replicator (camera + render product + BasicWriter) and
        the PickPlaceController. YCB props are already scattered in
        setup_scene via Python random — no Replicator scatter randomizer."""
        self._world  = self.get_world()
        self._franka = self._world.scene.get_object("franka")
        self._cube   = self._world.scene.get_object("cube")

        # Replicator + timeline settings (PDF p.42-43).
        # Skip timeline.play()/commit() — World.play_async() owns playback.
        # captureOnPlay=True: writer auto-captures on every play tick. The
        # PDF's manual rep.orchestrator.step() approach silently no-ops in our
        # running-Kit (TCP-socket) context — framesWritten stayed at -1.
        s = carb.settings.get_settings()
        s.set("/rtx/post/dlss/execMode", 2)               # DLSS Quality
        s.set("/omni/replicator/captureOnPlay", True)     # auto-capture on each play tick
        s.set("/app/player/useFixedTimeStepping", True)
        tl = omni.timeline.get_timeline_interface()
        tl.set_looping(False)
        tl.set_time_codes_per_second(STAGE_FPS)

        # ── Camera + render product + BasicWriter (PDF p.45-49) ──
        cam = rep.create.camera(
            name="SDGCam",
            position=(2, 2, 2),
            look_at="/World/Franka",
        )
        self._rp = rep.create.render_product(cam, RESOLUTION, name="SDGView")
        self._rp.hydra_texture.set_updates_enabled(False)

        out_dir = (DATA_SAVE_PATH if os.path.isdir("/workspace")
                   else "/tmp/replicator_output")
        os.makedirs(out_dir, exist_ok=True)
        if out_dir != DATA_SAVE_PATH:
            print(f"[WARN] {DATA_SAVE_PATH} not mounted; falling back to {out_dir}")
        print(f"Outputting data to {out_dir}")

        # BasicWriter writes one subdir per capture type:
        #   rgb/                        rgb_<frame>.png
        #   semantic_segmentation/      semantic_segmentation_<frame>.png
        #                               + semantic_segmentation_labels_<frame>.json
        #   instance_segmentation/      instance_segmentation_<frame>.png
        #                               + instance_segmentation_mapping_<frame>.json
        # colorize_* turns the raw ID buffers into viewable color PNGs.
        self._writer = rep.WriterRegistry.get("BasicWriter")
        self._writer.initialize(
            output_dir=out_dir,
            rgb=True,
            semantic_segmentation=True,
            colorize_semantic_segmentation=True,
            instance_segmentation=True,
            colorize_instance_segmentation=True,
        )
        self._writer.attach(self._rp)
        print(f"BasicWriter attached: rgb + semantic_segmentation + instance_segmentation → {out_dir}")

        # CRITICAL: rep.create.render_product / writer.attach invalidates the
        # physics tensor view that backs Franka and the cube wrappers. Without
        # this reset, the FIRST physics_step call to cube.get_world_pose()
        # raises "Failed to get rigid body transforms from backend" and the
        # controller never advances. Resetting rebuilds the physics view and
        # re-initializes all scene-registered wrappers with valid handles.
        await self._world.reset_async()
        self._franka = self._world.scene.get_object("franka")
        self._cube   = self._world.scene.get_object("cube")
        print("[setup_post_load] world reset_async done — physics view rebuilt")

        # ── Pick-place controller ────────────────────────────────
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        self._franka.gripper.set_joint_positions(
            self._franka.gripper.joint_opened_positions)

        self._frame_count    = 0
        self._capture_armed  = False
        self._task_completed = False
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)

        # Debug snapshot before play
        franka_pose = self._franka.get_world_pose()
        cube_pose   = self._cube.get_world_pose()
        joint_pos   = self._franka.get_joint_positions()
        gripper_pos = self._franka.gripper.get_joint_positions()
        print(f"[setup_post_load] Franka pose: pos={franka_pose[0]}, ori={franka_pose[1]}")
        print(f"[setup_post_load] Cube pose:   pos={cube_pose[0]}, ori={cube_pose[1]}")
        print(f"[setup_post_load] Franka joints: {joint_pos}")
        print(f"[setup_post_load] Gripper joints: {gripper_pos}")
        print(f"[setup_post_load] PLACE_TARGET: {PLACE_TARGET}")
        print(f"[setup_post_load] About to call play_async()")
        await self._world.play_async()
        print(f"[setup_post_load] play_async returned. timeline.is_playing={omni.timeline.get_timeline_interface().is_playing()}")

    def physics_step(self, step_size):
        """Per-tick: arm the data writer on frame 1, then drive PickPlaceController."""
        if self._task_completed:
            return

        self._frame_count += 1
        if self._frame_count == 1:
            # First physics frame: enable rendering + arm the data writer.
            self._rp.hydra_texture.set_updates_enabled(True)
            self._capture_armed = True
            cube_pos, _ = self._cube.get_world_pose()
            joints = self._franka.get_joint_positions()
            print(f"[physics_step] frame 1 fired. cube={cube_pos}, target={PLACE_TARGET}")
            print(f"[physics_step] frame 1 joints={joints}")

        cube_pos, _ = self._cube.get_world_pose()
        joint_positions = self._franka.get_joint_positions()
        actions = self._controller.forward(
            picking_position=cube_pos,
            placing_position=PLACE_TARGET,
            current_joint_positions=joint_positions,
        )
        self._franka.apply_action(actions)

        # Heartbeat every 50 ticks with controller phase + joint snapshot.
        if self._frame_count % 50 == 0:
            phase = getattr(self._controller, "_event", None)
            if phase is None:
                # Fallback: look up internal state machine attribute name
                phase = getattr(self._controller, "event", "?")
            tgt = getattr(actions, "joint_positions", None)
            tgt_str = (f"[{','.join(f'{v:.2f}' for v in tgt[:3])}...]"
                       if tgt is not None else "None")
            print(f"[physics_step] f={self._frame_count} phase={phase} "
                  f"cube_xy=({cube_pos[0]:.2f},{cube_pos[1]:.2f},{cube_pos[2]:.2f}) "
                  f"j0={joint_positions[0]:.2f} action_tgt={tgt_str} "
                  f"done={self._controller.is_done()}")

        if self._controller.is_done():
            print(f"[physics_step] controller done at frame {self._frame_count}")
            self._task_completed = True

    async def setup_post_reset(self):
        self._controller.reset()
        self._franka.gripper.set_joint_positions(
            self._franka.gripper.joint_opened_positions)
        self._frame_count    = 0
        self._capture_armed  = False
        self._task_completed = False
        await self._world.play_async()


# ── Run ─────────────────────────────────────────────────────────
sample = DomainRandomizationDemo()
await sample.load_world_async()
print("Domain randomization started — picking blue cube while YCB props scatter.")

try:
    # captureOnPlay=True drives the writer automatically — no explicit
    # rep.orchestrator.step() needed. Just pump Kit ticks until the
    # pick-and-place finishes.
    for _ in range(60000):
        await omni.kit.app.get_app().next_update_async()
        if sample._task_completed:
            break
finally:
    if hasattr(sample, "_rp"):
        sample._rp.hydra_texture.set_updates_enabled(False)
    rep.orchestrator.wait_until_complete()
    if hasattr(sample, "_writer"):
        sample._writer.detach()
    if hasattr(sample, "_rp"):
        sample._rp.destroy()
    print("Domain randomization complete. Output saved.")
