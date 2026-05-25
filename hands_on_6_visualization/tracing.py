from pxr import UsdGeom, Gf
import omni.usd
import omni.kit.app

# User settings
TARGET_PATH = "/World/Franka/panda_rightfinger"
SEGMENTS_ROOT = "/World/trajectory_segments_time"

MAX_SEGMENTS = 100 # maximum number of segments
CURVE_WIDTH = 0.01 # segment width
SAMPLE_INTERVAL = 0.05  # seconds between samples

# velocity color range
MIN_SPEED = 0.0
MAX_SPEED = 0.3

# stage
stage = omni.usd.get_context().get_stage()
if "trajectory_sub_time" in globals() and trajectory_sub_time is not None:
    try:
        trajectory_sub_time.unsubscribe()
    except Exception:
        pass
    trajectory_sub_time = None

# Clean previous segments root
old_root = stage.GetPrimAtPath(SEGMENTS_ROOT)
if old_root.IsValid():
    stage.RemovePrim(SEGMENTS_ROOT)

UsdGeom.Xform.Define(stage, SEGMENTS_ROOT)

# Global storage
last_point = None
segment_paths = []
segment_count = 0
time_acc = 0.0

# Get current target position
def get_target_position():
    target = stage.GetPrimAtPath(TARGET_PATH)
    if not target.IsValid():
        return None
    xform = UsdGeom.Xformable(target)
    world_matrix = xform.ComputeLocalToWorldTransform(0)
    pos = world_matrix.ExtractTranslation()
    return Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2]))

# Create one colored segment
def create_segment(p0, p1, color, seg_idx):
    seg_path = f"{SEGMENTS_ROOT}/seg_{seg_idx}"

    curve = UsdGeom.BasisCurves.Define(stage, seg_path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateBasisAttr(UsdGeom.Tokens.bspline)
    curve.GetPointsAttr().Set([p0, p1])
    curve.GetCurveVertexCountsAttr().Set([2])
    curve.GetWidthsAttr().Set([CURVE_WIDTH, CURVE_WIDTH])
    curve.GetDisplayColorAttr().Set([color])

    return seg_path

# Init first point
p_init = get_target_position()
if p_init is None:
    raise RuntimeError(f"Target prim not found: {TARGET_PATH}")

last_point = p_init
dt = 1.0 / 50.0

# Update callback
def update_trajectory_curve_time(event):
    global last_point, segment_paths, segment_count, time_acc

    # Conotrol sampling rate by accumulating time and only processing when it exceeds the sample interval
    time_acc += dt
    if time_acc < SAMPLE_INTERVAL:
        return
    sample_dt = time_acc
    time_acc = 0.0

    # Gate new position and calculate displacement
    new_point = get_target_position()
    dist = (new_point - last_point).GetLength()

    if dist > 1e-4:

        #! Do not modify the code outside of the TODO scetions.
        #! Calculate speed and map to color, create new segment, maintain a fixed maximum number of segments, and update last point
        # TODO3 ###########################################################

        # calculate speed
        speed = dist / sample_dt

        # map to color
        color = speed_to_color(speed)

        # create new segment
        seg_path = create_segment(last_point, new_point, color, segment_count)
        segment_paths.append(seg_path)
        segment_count += 1

        # Maintain a fixed maximum number of segments
        if len(segment_paths) > MAX_SEGMENTS:
            stage.RemovePrim(segment_paths.pop(0))

        # update last point
        last_point = new_point

        # TODO3 ###########################################################

#! Do not modify the code outside of the TODO scetions.
#! Implementation of the color mapping function for 5 color heatmap (blue, cyan, green, yellow, red)
# TODO4 #####################################
def speed_to_color(speed, vmin=MIN_SPEED, vmax=MAX_SPEED):
    t = 0.0 if vmax <= vmin else (speed - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))

    colors = [
        (0.0, 0.0, 1.0),  # blue
        (0.0, 1.0, 1.0),  # cyan
        (0.0, 1.0, 0.0),  # green
        (1.0, 1.0, 0.0),  # yellow
        (1.0, 0.0, 0.0),  # red
    ]

    n = len(colors) - 1
    scaled = t * n
    i = min(int(scaled), n - 1)
    f = scaled - i

    c0 = colors[i]
    c1 = colors[i + 1]
    r = c0[0] + (c1[0] - c0[0]) * f
    g = c0[1] + (c1[1] - c0[1]) * f
    b = c0[2] + (c1[2] - c0[2]) * f
    return (r, g, b)
# TODO4 #######################################

# Subscribe to update event
trajectory_sub_time = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
    update_trajectory_curve_time
)

print("Time-sampled colored trajectory started.")