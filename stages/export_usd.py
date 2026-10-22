"""Stage 5 - export_usd: write scene.usd (Z up, meters) for Isaac Sim.

/World
  PhysicsScene
  Collision/            invisible static colliders
    Ground              thin box at Z=0
    Walls/wall_XX       one box per detected wall segment
    Obstacles/obst_XX   extruded obstacle footprints (triangle meshes)
  Proxy/                visible stand-ins (no collision): gray floor, walls, obstacles.
                        Hidden automatically once a splat is attached.
  Splat                 reference to the 3DGRUT USDZ (NuRec), COLMAP frame -> world transform.
                        Points at an empty splat/placeholder.usda until --splat is given.
  Plan                  path curve + waypoint markers; waypoints also stored as attribute
                        splat2sim:waypoints and Start pose xform.

Examples:
    python stages/export_usd.py --scene data/room_sample
    python stages/export_usd.py --scene data/room_sample --splat splat/room.usdz   # path relative to scene.usd
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from common import load_json

SPLAT_PLACEHOLDER = "splat/placeholder.usda"


def set_xform(prim, translate=(0, 0, 0), yaw_deg=0.0, scale=(1, 1, 1)):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, translate)))
    x.AddRotateZOp().Set(float(yaw_deg))
    x.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def box(stage, path, center_xy, size_xy, height, yaw, color=None, collision=False, z0=0.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    set_xform(cube.GetPrim(), (center_xy[0], center_xy[1], z0 + height / 2), np.degrees(yaw),
              (size_xy[0], size_xy[1], height))
    if color is not None:
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.MakeInvisible()
    return cube


def triangulate(poly):
    """Ear clipping for a simple polygon (Nx2, any winding). Returns triangle index triples."""
    pts = np.asarray(poly, float)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    idx = list(range(len(pts))) if area > 0 else list(range(len(pts)))[::-1]
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    tris, guard = [], 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 1e-12:
                continue  # reflex vertex
            if any(cross(a, b, pts[j]) > 0 and cross(b, c, pts[j]) > 0 and cross(c, a, pts[j]) > 0
                   for j in idx if j not in (i0, i1, i2)):
                continue  # another vertex inside the ear
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            return None  # not simple; caller falls back to the convex hull
    tris.append(tuple(idx))
    return tris


def prism(stage, path, poly, height, color=None, collision=False):
    """Vertical extrusion of a 2D footprint polygon from Z=0 to Z=height."""
    poly = np.asarray(poly, float)
    tris = triangulate(poly)
    if tris is None:
        poly = cv2.convexHull(poly.astype(np.float32))[:, 0].astype(float)
        tris = triangulate(poly)
    n = len(poly)
    pts = [Gf.Vec3f(x, y, 0) for x, y in poly] + [Gf.Vec3f(x, y, height) for x, y in poly]
    counts, ids = [], []
    for a, b, c in tris:
        counts += [3, 3]
        ids += [a, c, b, n + a, n + b, n + c]  # bottom faces down, top faces up
    ccw = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) > 0
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        ids += [i, j, n + j, n + i] if ccw else [j, i, n + i, n + j]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(ids)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([Gf.Vec3f(*poly.min(0), 0), Gf.Vec3f(*poly.max(0), height)])
    if color is not None:
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")  # exact, static
        mesh.MakeInvisible()
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, help="scene folder (reads <scene>/out)")
    ap.add_argument("--out", type=Path, help="folder with walls.json + waypoints.json (default <scene>/out)")
    ap.add_argument("--usd", type=Path, help="output file (default <out>/scene.usd)")
    ap.add_argument("--splat", help="USDZ from 3DGRUT, path relative to scene.usd (activates the splat prim)")
    ap.add_argument("--proxy", choices=["auto", "on", "off"], default="auto",
                    help="visible proxy geometry: auto = only when no splat is attached")
    ap.add_argument("--min-obstacle-height", type=float, default=0.25,
                    help="obstacle colliders are at least this tall (m), so the robot can't climb low clutter")
    args = ap.parse_args()

    out = args.out or args.scene / "out"
    W = load_json(out / "walls.json")
    P = load_json(out / "waypoints.json")
    s = P["scale"]
    usd_path = args.usd or out / "scene.usd"

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().customLayerData = {"splat2sim": {"scale": s, "source": W["source"]}}

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    # extent of the map in meters, for the ground plane
    G = W["grid"]
    x0, x1 = G["x0"] * s, (G["x0"] + G["width"] * G["res"]) * s
    y0, y1 = (G["y1"] - G["height"] * G["res"]) * s, G["y1"] * s
    gc, gs = ((x0 + x1) / 2, (y0 + y1) / 2), (x1 - x0 + 4, y1 - y0 + 4)

    UsdGeom.Scope.Define(stage, "/World/Collision")
    box(stage, "/World/Collision/Ground", gc, gs, 0.1, 0, collision=True, z0=-0.1)
    UsdGeom.Scope.Define(stage, "/World/Collision/Walls")
    for w in W["walls"]:
        p0, p1 = np.array(w["p0"]) * s, np.array(w["p1"]) * s
        d = p1 - p0
        L = np.linalg.norm(d)
        box(stage, f"/World/Collision/Walls/wall_{w['id']:02d}", (p0 + p1) / 2,
            (L + w["thickness"] * s, w["thickness"] * s), w["height"] * s, np.arctan2(d[1], d[0]), collision=True)
    UsdGeom.Scope.Define(stage, "/World/Collision/Obstacles")
    for o in W["obstacles"]:
        poly = cv2.approxPolyDP(np.array(o["contour"], np.float32), 0.03, True)[:, 0] * s
        h = max(o["height"] * s, args.min_obstacle_height)
        prism(stage, f"/World/Collision/Obstacles/obst_{o['id']:02d}", poly, h, collision=True)

    # visible proxy geometry --------------------------------------------------------------
    proxy = UsdGeom.Xform.Define(stage, "/World/Proxy")
    box(stage, "/World/Proxy/Floor", gc, gs, 0.01, 0, color=(0.55, 0.55, 0.55), z0=-0.01)
    for w in W["walls"]:
        p0, p1 = np.array(w["p0"]) * s, np.array(w["p1"]) * s
        d = p1 - p0
        box(stage, f"/World/Proxy/wall_{w['id']:02d}", (p0 + p1) / 2,
            (np.linalg.norm(d) + w["thickness"] * s, w["thickness"] * s), w["height"] * s,
            np.arctan2(d[1], d[0]), color=(0.85, 0.3, 0.25))
    for o in W["obstacles"]:
        poly = cv2.approxPolyDP(np.array(o["contour"], np.float32), 0.03, True)[:, 0] * s
        prism(stage, f"/World/Proxy/obst_{o['id']:02d}", poly, max(o["height"] * s, args.min_obstacle_height),
              color=(0.95, 0.6, 0.1))
    show_proxy = args.proxy == "on" or (args.proxy == "auto" and not args.splat)
    if not show_proxy:
        proxy.MakeInvisible()

    # splat visual (NuRec USDZ from 3DGRUT, trained in the COLMAP frame) --------------------
    M = np.diag([s, s, s, 1.0]) @ np.array(W["to_world"])  # COLMAP -> world meters
    align = out / "splat_align.json"
    if args.splat and align.exists():  # splat frame differs from COLMAP: world <- COLMAP <- splat
        M = M @ np.linalg.inv(np.array(load_json(align)["colmap_to_splat"]))
    splat = UsdGeom.Xform.Define(stage, "/World/Splat")
    splat.AddTransformOp().Set(Gf.Matrix4d(M.T.tolist()))  # USD matrices are row-vector (transposed)
    if not args.splat:  # empty stand-in layer so the reference resolves until the real USDZ exists
        ph = Usd.Stage.CreateNew(str(usd_path.parent / SPLAT_PLACEHOLDER))
        UsdGeom.SetStageUpAxis(ph, UsdGeom.Tokens.z)
        ph.SetDefaultPrim(UsdGeom.Xform.Define(ph, "/SplatPlaceholder").GetPrim())
        ph.GetRootLayer().Save()
    splat.GetPrim().GetReferences().AddReference(args.splat or SPLAT_PLACEHOLDER)
    splat.GetPrim().SetCustomDataByKey("splat2sim:note",
                                       "COLMAP->world transform. Re-run export_usd.py --splat <file> to attach.")

    # plan --------------------------------------------------------------------------------
    plan = UsdGeom.Xform.Define(stage, "/World/Plan")
    wps = [(w["x"], w["y"]) for w in P["waypoints"]]
    plan.GetPrim().CreateAttribute("splat2sim:waypoints", Sdf.ValueTypeNames.Float3Array).Set(
        Vt.Vec3fArray([Gf.Vec3f(x, y, 0) for x, y in wps]))
    plan.GetPrim().CreateAttribute("splat2sim:robotRadius", Sdf.ValueTypeNames.Float).Set(P["robot_radius"])
    curve = UsdGeom.BasisCurves.Define(stage, "/World/Plan/Path")
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreatePointsAttr([Gf.Vec3f(x, y, 0.02) for x, y in wps])
    curve.CreateCurveVertexCountsAttr([len(wps)])
    curve.CreateWidthsAttr([0.04] * len(wps))
    curve.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.8, 1.0)])
    for i, (x, y) in enumerate(wps):
        sph = UsdGeom.Sphere.Define(stage, f"/World/Plan/wp_{i:02d}")
        sph.CreateRadiusAttr(0.06)
        set_xform(sph.GetPrim(), (x, y, 0.06))
        sph.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.8, 0.2) if i == 0 else
                                    Gf.Vec3f(0.9, 0.2, 0.2) if i == len(wps) - 1 else Gf.Vec3f(1, 1, 1)])
    start = UsdGeom.Xform.Define(stage, "/World/Plan/Start")
    set_xform(start.GetPrim(), (*P["start"]["xy"], 0), np.degrees(P["start"]["yaw"]))

    stage.GetRootLayer().Save()
    print(f"walls {len(W['walls'])}, obstacles {len(W['obstacles'])}, waypoints {len(wps)}, scale {s:.4f}, "
          f"splat {'-> ' + args.splat if args.splat else 'placeholder'}, proxy {'on' if show_proxy else 'off'}")
    print(f"-> {usd_path}")


if __name__ == "__main__":
    main()
