# ============================================================
# CosmosWriter Hands-On: record a Franka Pick-and-Place task
#
# Consolidated, self-contained version of the NVIDIA
# "2_CosmosWriter_HandsOn" slide deck. The slides show fragments
# to paste into the interactive sample file; this is all 7 steps
# stitched into ONE script you can paste straight into the Isaac
# Sim Script Editor (no UI button needed).
#
# Run via:  ./run_in_isaac.py examples/cosmos_writer_pick_place.py
#   or:     paste the whole file into Window > Script Editor and Run
#
# Built against Isaac Sim 5.1.0 (FrankaPickPlace + base_sample_experimental).
#
# IMPORTANT — Docker output path:
#   Isaac Sim runs as root inside the container, so "~" = /root and
#   /root/Documents is NOT mounted (invisible from the host). We write
#   to /workspace/output, which docker-compose maps to ./output/ on the
#   host. After a run, look on the HOST at:
#       ./output/cosmos_pick_place/_out_cosmos_pick_place/clip_0000/
#   The CosmosWriter API here is the VERIFIED Isaac Sim 5.1.0 pattern
#   (matches standalone_examples/.../cosmos_writer_simple.py), not the
#   slide-deck "guide" snippets.
# ============================================================

# ── Step 1 — CosmosWriter imports ─────────────────────────────
import asyncio
import os
import shutil
import time

import carb.settings
import omni.kit.app
import omni.replicator.core as rep
from isaacsim.examples.interactive.base_sample.base_sample_experimental import BaseSample
from isaacsim.robot.manipulators.examples.franka import FrankaPickPlace
from omni.replicator.core.backends import io_queue

# ── Step 1 — CosmosWriter hands-on settings ───────────────────
# Write to the host-mounted /workspace/output (./output/ on the host).
# Falls back to the cwd if that mount isn't present (e.g. native install).
#
# Each run goes in its OWN timestamped folder. The container writes as
# root, so a fixed dir becomes root-owned and can't be deleted/rerun
# from the host (EACCES). Unique dirs avoid that entirely — nothing is
# ever deleted. Old runs are cleaned in-container; see the note at the
# bottom of this file.
_MOUNTED_OUTPUT = "/workspace/output"
_OUTPUT_ROOT = os.path.join(
    _MOUNTED_OUTPUT if os.path.isdir(_MOUNTED_OUTPUT) else os.getcwd(),
    "cosmos_pick_place",
)
COSMOS_OUTPUT_DIR = os.path.join(
    _OUTPUT_ROOT, "run_" + time.strftime("%Y%m%d_%H%M%S")
)
COSMOS_RESOLUTION = (960, 540)
COSMOS_MAX_FRAMES = 120
COSMOS_WARMUP_UPDATES = 30
COSMOS_RESET_OUTPUT = False  # unique dir per run => never delete anything
COSMOS_ENABLE_MP4_ENCODING = False

# CosmosWriter writes one folder per modality under clip_0000/.
COSMOS_MODALITIES = ["rgb", "depth", "segmentation", "shaded_seg", "edges"]


class FrankaPickPlaceInteractive(BaseSample):
    """Interactive pick-and-place sample with CosmosWriter recording.

    Same class the hands-on builds on — we only add the recording
    plumbing around the existing FrankaPickPlace controller.
    """

    # ── Step 2 — recording state on the class ─────────────────
    def __init__(self) -> None:
        super().__init__()
        self.controller: FrankaPickPlace = None
        self._is_executing = False
        # CosmosWriter runtime state.
        self._cosmos_writer = None
        self._cosmos_render_product = None
        self._cosmos_capture_task = None
        self._cosmos_capture_enabled = False
        self._cosmos_frame_count = 0
        self._cosmos_max_frames = COSMOS_MAX_FRAMES
        self._cosmos_output_dir = COSMOS_OUTPUT_DIR

    def get_simulation_context(self):
        """Simulation context used for physics callbacks / play."""
        return self._simulation_context

    def setup_scene(self) -> None:
        """Build the Franka + cube scene via the stock controller."""
        self.controller = FrankaPickPlace()
        self.controller.setup_scene()
        print("Scene setup complete with FrankaPickPlace controller")

    async def setup_post_load(self):
        print("Pick-place scene loaded successfully")

    # ── Step 3 — create the Replicator camera ─────────────────
    async def _setup_cosmos_writer_async(self):
        """Create the recording camera + render product, then attach
        CosmosWriter to it (Steps 3 & 4). Uses the verified Isaac Sim
        5.1.0 API (cosmos_writer_simple.py), not the slide snippets."""
        if COSMOS_RESET_OUTPUT and os.path.isdir(self._cosmos_output_dir):
            shutil.rmtree(self._cosmos_output_dir)
        os.makedirs(self._cosmos_output_dir, exist_ok=True)

        settings = carb.settings.get_settings()
        # CosmosWriter is built on OmniGraph script nodes — without this
        # opt-in the writer produces NO data. (Required; PDF omitted it.)
        settings.set_bool("/app/omni.graph.scriptnode/opt_in", True)
        # DLSS Quality (2) gives the best synthetic-data image quality.
        settings.set("rtx/post/dlss/execMode", 2)

        # Capture is driven explicitly by the capture loop, not by play.
        rep.orchestrator.set_capture_on_play(False)

        camera = rep.functional.create.camera(
            position=(1.45, -1.35, 0.6),   # adjust if the robot isn't centered
            look_at=(0.0, 0.0, 0.18),      # aim at the table / cube
        )
        self._cosmos_render_product = rep.create.render_product(
            camera, COSMOS_RESOLUTION
        )

        # ── Step 4 — attach the writer ────────────────────────
        # CosmosWriter builds its own DiskBackend from output_dir and
        # exports the Cosmos modalities (RGB, Depth, Segmentation,
        # Shaded Segmentation, Edges). use_instance_id=True => no manual
        # semantic labels needed.
        self._cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
        self._cosmos_writer.initialize(
            output_dir=self._cosmos_output_dir,
            use_instance_id=True,
        )

        if not COSMOS_ENABLE_MP4_ENCODING:
            # on_final_frame is the real method that does the (sometimes
            # unstable) MP4 encoding — shadow it with a no-op so only the
            # PNG modality folders are produced. PNGs are written
            # incrementally during capture regardless.
            def _skip_mp4_finalization():
                print("Skipping MP4 finalization (PNG modalities are enough)")

            self._cosmos_writer.on_final_frame = _skip_mp4_finalization

        self._cosmos_writer.attach(self._cosmos_render_product)
        print(f"CosmosWriter attached -> {self._cosmos_output_dir}")

    # ── Step 5 — capture while the robot moves ────────────────
    async def _cosmos_capture_loop_async(self):
        """Record the current sim state each step. delta_time=0.0 because
        the pick-place task already drives physics — Replicator should
        only snapshot the current state, not advance time itself."""
        self._cosmos_frame_count = 0
        self._cosmos_capture_enabled = True

        # Let the renderer / scene settle before recording frame 0.
        for _ in range(COSMOS_WARMUP_UPDATES):
            await omni.kit.app.get_app().next_update_async()

        while (
            self._cosmos_capture_enabled
            and self._cosmos_frame_count < self._cosmos_max_frames
        ):
            await omni.kit.app.get_app().next_update_async()
            await rep.orchestrator.step_async(
                pause_timeline=False,
                delta_time=0.0,
            )
            self._cosmos_frame_count += 1

        print(f"Capture loop finished at {self._cosmos_frame_count} frames")

    def _pick_place_physics_callback(self, dt):
        """One pick-place step per physics tick (stock sample logic)."""
        if not self._is_executing or self.controller is None:
            return
        if self.controller.is_done():
            print("Pick-and-place completed successfully!")
            self._is_executing = False
            if self.get_simulation_context():
                self.get_simulation_context().remove_physics_callback("sim_step")
            return
        try:
            if not self.controller.forward():
                self._is_executing = False
                if self.get_simulation_context():
                    self.get_simulation_context().remove_physics_callback("sim_step")
        except Exception as e:
            print(f"Error during pick-and-place step: {e}")
            self._is_executing = False
            if self.get_simulation_context():
                self.get_simulation_context().remove_physics_callback("sim_step")

    # ── Step 6 — start recording when the task starts ─────────
    async def execute_pick_place_async(self):
        """Reset, set up the writer, launch the capture loop, then run
        the pick-and-place task — capture and control run together."""
        if self.controller is None:
            print("ERROR: Controller not initialized")
            return False
        print("Starting pick-and-place with CosmosWriter...")
        self._is_executing = True
        self.controller.reset()

        await self._setup_cosmos_writer_async()
        self._cosmos_capture_task = asyncio.ensure_future(
            self._cosmos_capture_loop_async()
        )

        world = self.get_simulation_context()
        world.add_physics_callback("sim_step", self._pick_place_physics_callback)
        await world.play_async()
        return True

    # ── Step 7 — stop and verify the dataset ──────────────────
    async def _stop_cosmos_capture_async(self):
        """Stop capture, flush the writer queue, and count PNGs per
        modality. A healthy run shows the same count in every folder."""
        self._cosmos_capture_enabled = False
        if self._cosmos_capture_task is not None:
            try:
                await self._cosmos_capture_task
            except asyncio.CancelledError:
                pass

        await rep.orchestrator.wait_until_complete_async()
        io_queue.wait_until_done()

        # Verified teardown (matches cosmos_writer_simple.py). Best-effort
        # so an unstable MP4 finalization can't lose the PNGs already on disk.
        try:
            if self._cosmos_writer is not None:
                self._cosmos_writer.detach()
            if self._cosmos_render_product is not None:
                self._cosmos_render_product.destroy()
        except Exception as e:
            print(f"(non-fatal) writer teardown raised: {e}")

        clip_dir = os.path.join(self._cosmos_output_dir, "clip_0000")
        print(f"\n=== Dataset verification: {clip_dir} ===")
        total = 0
        for modality in COSMOS_MODALITIES:
            modality_dir = os.path.join(clip_dir, modality)
            if os.path.isdir(modality_dir):
                count = len(
                    [n for n in os.listdir(modality_dir) if n.endswith(".png")]
                )
            else:
                count = 0  # guarded: stock snippet would crash if missing
            total += count
            print(f"  {modality}: {count} PNG frames")
        if total == 0:
            print("  WARNING: no frames written — check the Script Editor "
                  "console above for errors.")
        # The container path maps to the host repo dir:
        host_hint = self._cosmos_output_dir.replace(_MOUNTED_OUTPUT, "./output", 1)
        print(f"\nOn the HOST, find the dataset at: {host_hint}")


# ── Driver — runs the whole hands-on without the UI button ────
# The Isaac Sim GUI Script Editor compiles this file as a plain module,
# so top-level `await` is a SyntaxError. Wrap the flow in a coroutine
# and schedule it on Kit's event loop with asyncio.ensure_future — this
# works in BOTH the GUI Script Editor and run_in_isaac.py.
async def _run_handson():
    sample = FrankaPickPlaceInteractive()
    await sample.load_world_async()
    await sample.execute_pick_place_async()

    # Wait for the capture loop to record COSMOS_MAX_FRAMES (with a
    # safety cap so a stalled run can't hang the editor forever).
    max_wait_updates = COSMOS_WARMUP_UPDATES + COSMOS_MAX_FRAMES * 20
    for _ in range(max_wait_updates):
        await omni.kit.app.get_app().next_update_async()
        task = sample._cosmos_capture_task
        if task is not None and task.done():
            break

    await sample._stop_cosmos_capture_async()
    print("\nCosmosWriter hands-on complete.")


# Keep a reference so the task isn't garbage-collected mid-run.
_cosmos_handson_task = asyncio.ensure_future(_run_handson())

# ─────────────────────────────────────────────────────────────
# Cleaning up old runs (files are root-owned — no host `sudo` needed):
#   docker exec isaac-sim-quickstart-isaac-sim-1 \
#       rm -rf /workspace/output/cosmos_pick_place
# ─────────────────────────────────────────────────────────────
