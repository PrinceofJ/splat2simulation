"""Isaac Sim 5.x standalone: load scene.usd, spawn Nova Carter, follow the planned waypoints with a
proportional differential-drive controller, and record the main view + onboard camera to MP4.

Run with Isaac Sim's Python (NOT run on the laptop):
    python cloud/drive.py --scene data/room_sample --headless
    python cloud/drive.py --scene data/room_sample --main overview --show-proxy

Writes <scene>/out/video/{main.mp4, onboard.mp4, demo_inset.mp4, trajectory.csv}.
Needs: pip install imageio[ffmpeg] (inside the Isaac Sim venv).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scene", type=Path, required=True, help="scene folder (uses <scene>/out)")
ap.add_argument("--usd", type=Path, help="default <scene>/out/scene.usd")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--livestream", action="store_true", help="enable WebRTC streaming (view from the laptop)")
ap.add_argument("--robot-usd", help="override the Nova Carter asset path/URL")
ap.add_argument("--onboard-camera", help="prim path of the robot camera to record (default: auto-detect)")
ap.add_argument("--main", choices=["chase", "overview"], default="chase", help="main view camera")
ap.add_argument("--show-proxy", action="store_true", help="show proxy walls/obstacles even with a splat")
ap.add_argument("--hide-path", action="store_true", help="hide the path line + waypoint markers")
ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--max-speed", type=float, default=0.6, help="m/s")
ap.add_argument("--max-turn", type=float, default=1.2, help="rad/s")
ap.add_argument("--k-lin", type=float, default=0.8)
ap.add_argument("--k-ang", type=float, default=2.0)
ap.add_argument("--tol", type=float, default=0.15, help="waypoint reached radius (m)")
ap.add_argument("--warmup", type=int, default=90, help="frames rendered before recording (splat load)")
ap.add_argument("--max-seconds", type=float, default=120)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402  (must come before any omni import)

app = SimulationApp({"headless": args.headless, "width": args.res[0], "height": args.res[1]})

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim  # noqa: E402
from isaacsim.core.utils.nucleus import get_assets_root_path  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

if args.livestream:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.kit.livestream.webrtc")

OUT = args.scene / "out"
VIDEO = OUT / "video"
VIDEO.mkdir(parents=True, exist_ok=True)
WAYPOINTS = json.loads((OUT / "waypoints.json").read_text())
ROBOT = "/World/NovaCarter"
WHEEL_RADIUS, WHEEL_BASE = 0.14, 0.4132  # Nova Carter (Isaac Sim robot asset docs)
CARTER_CANDIDATES = [  # asset layout moved between Isaac Sim releases; first one that exists wins
    "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
    "/Isaac/Robots/NVIDIA/Carter/nova_carter/nova_carter.usd",
    "/Isaac/Robots/Carter/nova_carter_sensors.usd",
    "/Isaac/Robots/Carter/nova_carter.usd",
]


def log(*a):
    print("[drive]", *a, flush=True)


def look_at(eye, target, up=(0, 0, 1)):
    """USD camera transform (camera looks down its -Z, +Y up), row-vector matrix."""
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return Gf.Matrix4d(r[0], r[1], r[2], 0, u[0], u[1], u[2], 0, -f[0], -f[1], -f[2], 0, eye[0], eye[1], eye[2], 1)


def set_matrix(prim, m):
    x = UsdGeom.Xformable(prim)
    ops = x.GetOrderedXformOps()
    if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
        ops[0].Set(m)
    else:
        x.ClearXformOpOrder()
        x.AddTransformOp().Set(m)


def resolve_carter():
    import omni.client

    if args.robot_usd:
        return args.robot_usd
    root = get_assets_root_path()
    if root is None:
        sys.exit("Could not reach the Isaac Sim asset root; pass --robot-usd")
    for c in CARTER_CANDIDATES:
        if omni.client.stat(root + c)[0] == omni.client.Result.OK:
            return root + c
    sys.exit(f"Nova Carter not found under {root}; tried {CARTER_CANDIDATES}. Pass --robot-usd.")


def find_onboard_camera(stage):
    if args.onboard_camera:
        return args.onboard_camera
    cams = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera) and str(p.GetPath()).startswith(ROBOT)]
    log("robot cameras:", [str(c.GetPath()) for c in cams])
    for pref in (("front", "left"), ("front",), ("left",)):
        for c in cams:
            p = str(c.GetPath()).lower()
            if all(k in p for k in pref) and "depth" not in p:
                return str(c.GetPath())
    # none found: mount our own forward-looking camera on the chassis
    parent = next((str(p.GetPath()) for p in stage.Traverse()
                   if str(p.GetPath()).startswith(ROBOT) and p.GetName() in ("chassis_link", "base_link")), ROBOT)
    cam = UsdGeom.Camera.Define(stage, parent + "/splat2sim_cam")
    cam.CreateFocalLengthAttr(14.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100))
    # camera -Z = robot +X (forward), camera +Y = robot +Z (up)
    set_matrix(cam.GetPrim(), Gf.Matrix4d(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0.3, 0, 0.45, 1))
    return str(cam.GetPath())


def yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    usd = (args.usd or OUT / "scene.usd").resolve()
    log("opening", usd)
    omni.usd.get_context().open_stage(str(usd))
    while is_stage_loading():
        app.update()
    stage = omni.usd.get_context().get_stage()

    if args.show_proxy:
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Proxy")).MakeVisible()
    if args.hide_path:
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Plan")).MakeInvisible()
    if not any(p.HasAPI(UsdLux.LightAPI) for p in stage.Traverse()):  # splats are self-lit; proxies are not
        UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000)

    splat = stage.GetPrimAtPath("/World/Splat")
    refs = splat.GetMetadata("references") if splat else None
    log("splat reference:", refs)

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / args.fps,
                  physics_prim_path="/World/PhysicsScene")

    carter = resolve_carter()
    log("robot asset:", carter)
    add_reference_to_stage(usd_path=carter, prim_path=ROBOT)
    sx, sy = WAYPOINTS["start"]["xy"]
    syaw = WAYPOINTS["start"]["yaw"]
    q = [math.cos(syaw / 2), 0, 0, math.sin(syaw / 2)]
    SingleXFormPrim(ROBOT, position=np.array([sx, sy, 0.05]), orientation=np.array(q))
    root = next((str(p.GetPath()) for p in stage.Traverse()
                 if str(p.GetPath()).startswith(ROBOT) and p.HasAPI(UsdPhysics.ArticulationRootAPI)), ROBOT)
    robot = world.scene.add(SingleArticulation(prim_path=root, name="carter"))

    # cameras -------------------------------------------------------------------------------
    onboard = find_onboard_camera(stage)
    main_cam = UsdGeom.Camera.Define(stage, "/World/Splat2SimCams/Main")
    main_cam.CreateFocalLengthAttr(18.0)
    main_cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200))
    wps = np.array([[w["x"], w["y"]] for w in WAYPOINTS["waypoints"]])
    center = wps.mean(0)
    span = max(np.ptp(wps, 0).max(), 3.0)
    if args.main == "overview":
        set_matrix(main_cam.GetPrim(), look_at((center[0] - 0.6 * span, center[1] - 0.6 * span, 0.9 * span),
                                               (center[0], center[1], 0)))
    log("cameras: main", main_cam.GetPath(), "onboard", onboard)

    world.reset()
    dofs = list(robot.dof_names)
    log("dofs:", dofs)

    def pick(side):
        exact = f"joint_wheel_{side}"
        if exact in dofs:
            return dofs.index(exact)
        for i, n in enumerate(dofs):
            if "wheel" in n.lower() and side in n.lower() and "caster" not in n.lower():
                return i
        sys.exit(f"No {side} wheel joint among {dofs}")

    wheels = np.array([pick("left"), pick("right")])

    rp_main = rep.create.render_product(str(main_cam.GetPath()), tuple(args.res))
    rp_board = rep.create.render_product(onboard, tuple(args.res))
    an_main = rep.AnnotatorRegistry.get_annotator("rgb")
    an_board = rep.AnnotatorRegistry.get_annotator("rgb")
    an_main.attach([rp_main])
    an_board.attach([rp_board])

    def chase_cam(x, y, yaw):
        back = np.array([math.cos(yaw), math.sin(yaw)])
        eye = (x - 2.2 * back[0], y - 2.2 * back[1], 1.6)
        set_matrix(main_cam.GetPrim(), look_at(eye, (x + 0.8 * back[0], y + 0.8 * back[1], 0.3)))

    def drive(v, w):
        wl = (v - w * WHEEL_BASE / 2) / WHEEL_RADIUS
        wr = (v + w * WHEEL_BASE / 2) / WHEEL_RADIUS
        robot.apply_action(ArticulationAction(joint_velocities=np.array([wl, wr]), joint_indices=wheels))

    # warm up: let physics settle and the splat stream in before recording ----------------
    smooth = None
    for _ in range(args.warmup):
        drive(0, 0)
        pos, rot = robot.get_world_pose()
        if args.main == "chase":
            chase_cam(pos[0], pos[1], yaw_of(rot))
        world.step(render=True)

    writers = {k: imageio.get_writer(VIDEO / f"{k}.mp4", fps=args.fps, codec="libx264", quality=8,
                                     macro_block_size=8)
               for k in ("main", "onboard", "demo_inset")}
    traj = open(VIDEO / "trajectory.csv", "w", newline="")
    tw = csv.writer(traj)
    tw.writerow(["t", "x", "y", "yaw", "v", "w", "target"])

    target, t, frames = 1, 0.0, 0
    log(f"following {len(wps)} waypoints, {WAYPOINTS['length_m']:.1f} m")
    while app.is_running() and t < args.max_seconds:
        pos, rot = robot.get_world_pose()
        x, y, yaw = float(pos[0]), float(pos[1]), yaw_of(rot)
        gx, gy = wps[target]
        dist = math.hypot(gx - x, gy - y)
        last = target == len(wps) - 1
        if dist < args.tol:
            if last:
                log(f"goal reached at t={t:.1f}s")
                break
            target += 1
            continue
        err = wrap(math.atan2(gy - y, gx - x) - yaw)
        # proportional control; slow down while badly misaligned, and near the final goal
        v = min(args.max_speed, args.k_lin * (dist if last else max(dist, 0.5))) * max(0.0, math.cos(err)) ** 2
        w = float(np.clip(args.k_ang * err, -args.max_turn, args.max_turn))
        drive(v, w)
        tw.writerow([f"{t:.3f}", f"{x:.3f}", f"{y:.3f}", f"{yaw:.3f}", f"{v:.3f}", f"{w:.3f}", target])

        if args.main == "chase":  # smoothed chase camera
            smooth = np.array([x, y, yaw]) if smooth is None else smooth + 0.1 * (
                np.array([x, y, smooth[2] + wrap(yaw - smooth[2])]) - smooth)
            chase_cam(*smooth)
        world.step(render=True)
        t += 1 / args.fps

        a, b = an_main.get_data(), an_board.get_data()
        if a is None or b is None or getattr(a, "size", 0) == 0 or getattr(b, "size", 0) == 0:
            continue
        a, b = np.asarray(a)[..., :3], np.asarray(b)[..., :3]
        writers["main"].append_data(a)
        writers["onboard"].append_data(b)
        inset = np.asarray(Image.fromarray(b).resize((a.shape[1] // 3, a.shape[0] // 3)))
        comp = a.copy()
        h, w_ = inset.shape[:2]
        comp[-h - 16:-16, -w_ - 16:-16] = inset
        comp[-h - 18:-h - 16, -w_ - 18:-14] = 255  # thin frame
        writers["demo_inset"].append_data(comp)
        frames += 1
    else:
        log(f"stopped after {t:.1f}s without reaching the goal (target waypoint {target})")

    drive(0, 0)
    for wr in writers.values():
        wr.close()
    traj.close()
    log(f"wrote {frames} frames to {VIDEO}")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
