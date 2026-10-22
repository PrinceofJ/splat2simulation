"""Stage 4 - plan: A* on the occupancy grid, inflated by the robot radius -> waypoints JSON (meters).

Scale: walls.py guesses meters from the camera height. Tape-measure one detected wall (ids are drawn
in walls_07_walls.png) and pass it here; the factor is stored in waypoints.json and applied by
export_usd.py to everything (walls, obstacles, splat).

Examples:
    python stages/plan.py --scene data/room_sample                          # auto start/goal
    python stages/plan.py --scene data/my_room --wall-id 0 --wall-length 3.62 --start 0.5,-1 --goal 2,3
"""
import argparse
import heapq
import sys
from pathlib import Path

import cv2
import numpy as np

from common import load_json, save_json
from walls import label, occupancy_bgr, upscale

SQRT2 = np.sqrt(2)
NBRS = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def astar(ok, cost, start, goal):
    """8-connected A* on boolean grid `ok` with per-cell extra cost multiplier `cost` (>= 1)."""
    h, w = ok.shape
    gr, gc = goal
    heur = lambda r, c: max(abs(r - gr), abs(c - gc)) + (SQRT2 - 1) * min(abs(r - gr), abs(c - gc))
    g = np.full(ok.shape, np.inf)
    parent = -np.ones(ok.shape + (2,), int)
    g[start] = 0
    pq = [(heur(*start), 0.0, start)]
    while pq:
        _, gs, (r, c) = heapq.heappop(pq)
        if (r, c) == goal:
            break
        if gs > g[r, c]:
            continue
        for dr, dc, step in NBRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and ok[nr, nc]:
                if dr and dc and not (ok[r, nc] and ok[nr, c]):
                    continue  # no corner cutting
                ng = gs + step * cost[nr, nc]
                if ng < g[nr, nc]:
                    g[nr, nc] = ng
                    parent[nr, nc] = (r, c)
                    heapq.heappush(pq, (ng + heur(nr, nc), ng, (nr, nc)))
    if not np.isfinite(g[goal]):
        return None
    path, cur = [goal], goal
    while cur != start:
        cur = tuple(parent[cur])
        path.append(cur)
    return path[::-1]


def line_clear(ok, a, b):
    m = np.zeros(ok.shape, np.uint8)
    cv2.line(m, (a[1], a[0]), (b[1], b[0]), 1, 1)
    return bool(ok[m > 0].all())


def shortcut(path, ok):
    """Greedy line-of-sight simplification: keep only the corners the robot needs."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_clear(ok, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def bfs_far(ok, src, pick=None):
    """Geodesic (4-connected BFS) distance from src over `ok`; returns the farthest cell that is also in
    `pick` (default: any reachable cell)."""
    dist = np.full(ok.shape, -1, int)
    dist[src] = 0
    q, k = [src], 0
    while k < len(q):
        r, c = q[k]
        k += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < ok.shape[0] and 0 <= nc < ok.shape[1] and ok[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    if pick is not None and (pick & (dist >= 0)).any():
        dist = np.where(pick, dist, -1)
    far = np.unravel_index(np.argmax(dist), dist.shape)
    return (int(far[0]), int(far[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, help="scene folder (reads <scene>/out)")
    ap.add_argument("--out", type=Path, help="folder with walls.json / occupancy.npy (default <scene>/out)")
    ap.add_argument("--robot-radius", type=float, default=0.3, help="meters (Nova Carter ~0.3 inscribed)")
    ap.add_argument("--clearance", type=float, default=0.3,
                    help="extra distance (m) beyond the radius the planner prefers to keep from obstacles")
    sc = ap.add_mutually_exclusive_group()
    sc.add_argument("--scale", type=float, help="explicit scale factor applied to walls.py units")
    sc.add_argument("--wall-id", type=int, help="id of a wall you tape-measured (see walls_07_walls.png)")
    ap.add_argument("--wall-length", type=float, help="tape-measured length of --wall-id, meters")
    ap.add_argument("--start", type=lambda s: tuple(map(float, s.split(","))), help="x,y in meters")
    ap.add_argument("--goal", type=lambda s: tuple(map(float, s.split(","))), help="x,y in meters")
    args = ap.parse_args()

    out = args.out or args.scene / "out"
    W = load_json(out / "walls.json")
    occ = np.load(out / "occupancy.npy")
    G = W["grid"]

    # scale factor ------------------------------------------------------------------------
    s = 1.0
    if args.scale:
        s = args.scale
    elif args.wall_id is not None:
        if args.wall_length is None:
            sys.exit("--wall-id needs --wall-length")
        wall = next((w for w in W["walls"] if w["id"] == args.wall_id), None)
        if wall is None:
            sys.exit(f"No wall #{args.wall_id}; ids: {[w['id'] for w in W['walls']]}")
        s = args.wall_length / wall["length"]
        print(f"wall #{args.wall_id}: detected {wall['length']:.3f}, measured {args.wall_length:.3f} -> scale {s:.4f}")
    cell = G["res"] * s  # meters per grid cell

    def to_px(xy):  # meters -> (row, col)
        x, y = xy[0] / s, xy[1] / s
        return int((G["y1"] - y) / G["res"]), int((x - G["x0"]) / G["res"])

    def to_m(rc):
        r, c = rc
        return [(G["x0"] + (c + 0.5) * G["res"]) * s, (G["y1"] - (r + 0.5) * G["res"]) * s]

    # inflate by robot radius (unknown space counts as blocked) ---------------------------
    free = (occ == 0).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * cell  # meters to nearest non-free cell
    ok = dist > args.robot_radius
    # keep only the biggest connected traversable region
    n, cc, stats, _ = cv2.connectedComponentsWithStats(ok.astype(np.uint8), connectivity=8)
    if n < 2:
        sys.exit("No traversable space left after inflating by the robot radius. Lower --robot-radius "
                 "or check the scale (--wall-id/--wall-length).")
    main_cc = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    region = cc == main_cc
    # cost rises as the robot gets closer than radius + clearance to an obstacle
    pref = args.robot_radius + args.clearance
    cost = 1 + 4 * np.clip((pref - dist) / max(args.clearance, 1e-6), 0, 1)

    def snap(rc):  # nearest traversable cell
        if 0 <= rc[0] < occ.shape[0] and 0 <= rc[1] < occ.shape[1] and ok[rc] and region[rc]:
            return rc
        rr, cc_ = np.nonzero(region)
        i = np.argmin((rr - rc[0]) ** 2 + (cc_ - rc[1]) ** 2)
        print(f"  snapped {to_m(rc)} -> {to_m((rr[i], cc_[i]))} (was not traversable)")
        return int(rr[i]), int(cc_[i])

    # start / goal: given, or the two ends of the longest geodesic in the free region ------
    comfy = region & (dist > pref)  # endpoints should have some room around them
    if args.start:
        start = snap(to_px(args.start))
    else:
        start = bfs_far(region, tuple(np.argwhere(region)[0]), comfy)
    if args.goal:
        goal = snap(to_px(args.goal))
    else:
        goal = bfs_far(region, start, comfy)

    path = astar(ok & region, cost, start, goal)
    if path is None:
        sys.exit("A* found no path between start and goal.")
    # shortcuts must keep half the preferred clearance (or stay on the A* path's own cells)
    on_path = np.zeros(ok.shape, bool)
    on_path[tuple(np.array(path).T)] = True
    wps = shortcut(path, (region & (dist > args.robot_radius + 0.5 * args.clearance)) | on_path)
    wp_m = [to_m(p) for p in wps]
    seg = np.diff(np.array(wp_m), axis=0)
    length = float(np.linalg.norm(seg, axis=1).sum())
    yaws = [float(np.arctan2(d[1], d[0])) for d in seg] + [float(np.arctan2(seg[-1][1], seg[-1][0]))]
    print(f"path: {len(path)} cells -> {len(wps)} waypoints, {length:.2f} m; scale {s:.4f}")

    save_json(out / "waypoints.json", dict(
        frame="world: meters, Z up, floor at Z=0 (same frame as scene.usd)",
        scale=s, robot_radius=args.robot_radius,
        start=dict(xy=wp_m[0], yaw=yaws[0]), goal=dict(xy=wp_m[-1], yaw=yaws[-1]),
        waypoints=[dict(x=p[0], y=p[1], yaw=y) for p, y in zip(wp_m, yaws)],
        length_m=length, dense_path=[to_m(p) for p in path[::3]]))

    # debug images -------------------------------------------------------------------------
    vis = occupancy_bgr(occ)
    vis[(occ == 0) & ~ok] = (190, 190, 245)  # free but too close for the robot
    vis[ok & ~region] = (215, 215, 215)
    cv2.imwrite(str(out / "plan_01_inflated.png"),
                label(upscale(vis), f"Robot radius {args.robot_radius} m: pink = too close to obstacles"))

    topdown = cv2.imread(str(out / "topdown_rgb.png"))
    big = upscale(topdown)
    f = big.shape[1] / topdown.shape[1]
    px = lambda rc: (int((rc[1] + 0.5) * f), int((rc[0] + 0.5) * f))
    tint = big.copy()
    tint[cv2.resize((~ok).astype(np.uint8), (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST) > 0] //= 2
    big = cv2.addWeighted(big, 0.4, tint, 0.6, 0)
    for a, b in zip(path[:-1], path[1:]):
        cv2.line(big, px(a), px(b), (200, 200, 200), 1, cv2.LINE_AA)
    for a, b in zip(wps[:-1], wps[1:]):
        cv2.line(big, px(a), px(b), (255, 200, 0), max(2, int(f / 2)), cv2.LINE_AA)
    for p in wps:
        cv2.circle(big, px(p), max(3, int(f)), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(big, px(wps[0]), max(5, int(1.6 * f)), (60, 200, 60), -1, cv2.LINE_AA)
    cv2.circle(big, px(wps[-1]), max(5, int(1.6 * f)), (60, 60, 230), -1, cv2.LINE_AA)
    cv2.imwrite(str(out / "plan_02_path.png"),
                label(big, f"A* path: {len(wps)} waypoints, {length:.1f} m (green start, red goal)"))
    print(f"-> {out / 'waypoints.json'}")


if __name__ == "__main__":
    main()
