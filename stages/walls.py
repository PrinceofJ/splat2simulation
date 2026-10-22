"""Stage 3 - walls: points -> level floor (Z up) -> top-down density -> occupancy grid + wall segments.

Every step writes a debug PNG to <out>/walls_XX_*.png (meant for the demo video).

Units: COLMAP/splat scale is arbitrary. Here we pick an *approximate* metric scale by assuming the
phone was held --cam-height meters above the floor. `plan.py --wall-id N --wall-length L` later
corrects it with one tape-measured wall.

Examples:
    python stages/walls.py --scene data/room_sample
    python stages/walls.py --ply path/to/splat.ply --out data/my_room/out --up 0,-1,0
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from common import load_colmap_model, load_splat_ply, save_json

# BGR colors used in the debug images
C_WALL = (60, 60, 230)
C_OBST = (0, 170, 255)
C_CAM = (230, 160, 40)
C_FREE = (235, 235, 235)


# ----------------------------------------------------------------------------- geometry

def remove_outliers(xyz, k=12, std_ratio=2.0):
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)
    md = d[:, 1:].mean(1)
    return md < md.mean() + std_ratio * md.std()


def rotation_a_to_b(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def fit_plane_svd(pts):
    c = pts.mean(0)
    n = np.linalg.svd(pts - c)[2][-1]
    return n / np.linalg.norm(n), c


def ransac_horizontal_planes(xyz, up, thr, max_angle_deg=20, n_planes=3, iters=1500, seed=0):
    """Greedily extract up to n_planes roughly horizontal planes. Returns [(normal, point, inlier_mask)]."""
    rng = np.random.default_rng(seed)
    cos_max = np.cos(np.radians(max_angle_deg))
    remaining = np.ones(len(xyz), bool)
    planes = []
    for _ in range(n_planes):
        idx = np.flatnonzero(remaining)
        if len(idx) < 100:
            break
        pts = xyz[idx]
        best, best_cnt = None, 0
        for _ in range(iters):
            p = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            if abs(n @ up) < cos_max:
                continue
            cnt = np.count_nonzero(np.abs((pts - p[0]) @ n) < thr)
            if cnt > best_cnt:
                best, best_cnt = (n, p[0]), cnt
        if best is None:
            break
        inl = np.abs((pts - best[1]) @ best[0]) < thr
        n, c = fit_plane_svd(pts[inl])  # refine
        if n @ up < 0:
            n = -n
        mask = np.zeros(len(xyz), bool)
        mask[idx[np.abs((pts - c) @ n) < thr]] = True
        planes.append((n, c, mask))
        remaining &= ~mask
    return planes


def level_floor(xyz, cam_centers, cam_ups, up_hint, cam_height, units_scale):
    """Returns a 4x4 similarity transform (input -> world, floor at Z=0, Z up) and debug info."""
    if cam_ups is not None:
        up = cam_ups.mean(0)
    elif up_hint is not None:
        up = np.asarray(up_hint, float)
    else:  # no cameras: assume the thinnest axis of the scene is vertical
        up = np.linalg.svd(xyz - xyz.mean(0))[2][-1]
    up /= np.linalg.norm(up)

    h = xyz @ up
    extent = np.percentile(h, 95) - np.percentile(h, 5)
    planes = ransac_horizontal_planes(xyz, up, thr=0.01 * extent)
    if not planes:
        sys.exit("No horizontal plane found. Is the floor visible in the capture?")
    # the floor is the lowest plane that is still a substantial fraction of the biggest one
    biggest = max(p[2].sum() for p in planes)
    cands = [p for p in planes if p[2].sum() >= 0.3 * biggest]
    n, c, floor_mask = min(cands, key=lambda p: p[1] @ up)
    if cam_centers is not None and np.median((cam_centers - c) @ n) < 0:
        n = -n  # cameras must be above the floor

    R = rotation_a_to_b(n, np.array([0.0, 0.0, 1.0]))
    if cam_centers is not None:
        cam_h = np.median((cam_centers - c) @ n)
        s = cam_height / cam_h
        center_xy = np.median(cam_centers, 0)
    else:
        s = units_scale
        center_xy = np.median(xyz, 0)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = -s * R @ c
    off = apply(T, center_xy[None])[0]
    T[:2, 3] -= off[:2]  # center the scene in XY
    info = dict(n_planes=len(planes), floor_inliers=int(floor_mask.sum()),
                plane_heights=[float((p[1] - c) @ n * s) for p in planes], scale_guess=float(s))
    return T, floor_mask, info


def apply(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def rotz(theta):
    T = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


# ----------------------------------------------------------------------------- rasterizing

class Grid:
    """Top-down raster. Row 0 is the max-Y edge so images read like a map (Y up)."""

    def __init__(self, xy, res, pad):
        lo = np.percentile(xy, 0.5, 0) - pad
        hi = np.percentile(xy, 99.5, 0) + pad
        self.res, self.x0, self.y1 = res, float(lo[0]), float(hi[1])
        self.w = int(np.ceil((hi[0] - lo[0]) / res))
        self.h = int(np.ceil((hi[1] - lo[1]) / res))

    def to_px(self, xy):
        c = np.floor((xy[:, 0] - self.x0) / self.res).astype(int)
        r = np.floor((self.y1 - xy[:, 1]) / self.res).astype(int)
        return r, c

    def to_world(self, r, c):
        return self.x0 + (np.asarray(c) + 0.5) * self.res, self.y1 - (np.asarray(r) + 0.5) * self.res

    def inside(self, r, c):
        return (r >= 0) & (r < self.h) & (c >= 0) & (c < self.w)

    def hist(self, xy, weights=None):
        r, c = self.to_px(xy)
        ok = self.inside(r, c)
        img = np.zeros((self.h, self.w), np.float32)
        np.add.at(img, (r[ok], c[ok]), 1 if weights is None else weights[ok])
        return img

    def max_img(self, xy, val):
        r, c = self.to_px(xy)
        ok = self.inside(r, c)
        img = np.full((self.h, self.w), -np.inf, np.float32)
        np.maximum.at(img, (r[ok], c[ok]), val[ok])
        return img

    def meta(self):
        return dict(res=self.res, x0=self.x0, y1=self.y1, width=self.w, height=self.h)


def colorize(img01, cmap=cv2.COLORMAP_INFERNO):
    return cv2.applyColorMap((np.clip(img01, 0, 1) * 255).astype(np.uint8), cmap)


def log_norm(d):
    l = np.log1p(d)
    return l / max(l.max(), 1e-6)


def label(img, text, scale=1.0):
    """Title bar on top of a debug image."""
    s = max(1.0, img.shape[1] / 900) * scale
    bar = np.full((int(34 * s), img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, text, (int(10 * s), int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s,
                (255, 255, 255), max(1, int(1.5 * s)), cv2.LINE_AA)
    return np.vstack([bar, img])


def upscale(img, min_w=1000):
    f = max(1, int(np.ceil(min_w / img.shape[1])))
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) if f > 1 else img


def dominant_yaw(mask, res):
    """Dominant wall direction (mod 90 deg) from Hough segments, length-weighted."""
    lines = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=15, minLineLength=int(0.4 / res),
                            maxLineGap=int(0.15 / res))
    if lines is None:
        return 0.0
    l = lines.reshape(-1, 4).astype(float)
    ang = np.arctan2(-(l[:, 3] - l[:, 1]), l[:, 2] - l[:, 0])  # image rows point down
    length = np.hypot(l[:, 2] - l[:, 0], l[:, 3] - l[:, 1])
    a4 = 4 * ang  # mod 90 deg -> circular mean on 4x angle
    return float(np.arctan2((length * np.sin(a4)).sum(), (length * np.cos(a4)).sum()) / 4)


# ----------------------------------------------------------------------------- wall segments

def merge_segments(lines, ang_tol_deg, off_tol, gap_tol, snap_deg):
    """Cluster Hough segments by direction + offset and merge overlapping ones into walls.

    lines: Nx4 world coords (x1,y1,x2,y2). Returns list of (p0, p1)."""
    segs = []
    for x1, y1, x2, y2 in lines:
        th = np.arctan2(y2 - y1, x2 - x1) % np.pi
        for axis in (0, np.pi / 2, np.pi):  # snap near-axis lines (scene is Manhattan-aligned)
            if abs(th - axis) < np.radians(snap_deg):
                th = axis % np.pi
        d = np.array([np.cos(th), np.sin(th)])
        nrm = np.array([-d[1], d[0]])
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        t = sorted([(np.array([x1, y1]) - mid) @ d, (np.array([x2, y2]) - mid) @ d])
        off = mid @ nrm
        a = mid @ d
        segs.append(dict(th=th, off=off, t0=a + t[0], t1=a + t[1], n=1))

    def ang_diff(a, b):
        d = abs(a - b) % np.pi
        return min(d, np.pi - d)

    clusters = []
    for s in sorted(segs, key=lambda s: -(s["t1"] - s["t0"])):
        for c in clusters:
            if ang_diff(c["th"], s["th"]) < np.radians(ang_tol_deg) and abs(c["off"] - s["off"]) < off_tol:
                c["members"].append(s)
                break
        else:
            clusters.append(dict(th=s["th"], off=s["off"], members=[s]))

    walls = []
    for c in clusters:
        d = np.array([np.cos(c["th"]), np.sin(c["th"])])
        nrm = np.array([-d[1], d[0]])
        w = np.array([m["t1"] - m["t0"] for m in c["members"]])
        off = np.average([m["off"] for m in c["members"]], weights=w)
        ivs = sorted((m["t0"], m["t1"]) for m in c["members"])
        cur = list(ivs[0])
        for a, b in ivs[1:] + [(np.inf, np.inf)]:
            if a - cur[1] <= gap_tol:
                cur[1] = max(cur[1], b)
            else:
                walls.append((off * nrm + cur[0] * d, off * nrm + cur[1] * d))
                cur = [a, b]
    return walls


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", type=Path, help="scene folder with sparse/0 (COLMAP output)")
    src.add_argument("--ply", type=Path, help="Gaussian splat .ply instead of COLMAP points")
    ap.add_argument("--sparse", type=Path, help="override COLMAP model folder (default <scene>/sparse/0)")
    ap.add_argument("--out", type=Path, help="output folder (default <scene>/out)")
    ap.add_argument("--cam-height", type=float, default=1.4, help="assumed camera height above floor (m)")
    ap.add_argument("--units-scale", type=float, default=1.0, help="scale for --ply input without cameras")
    ap.add_argument("--up", type=lambda s: [float(v) for v in s.split(",")], help="up hint for --ply, e.g. 0,-1,0")
    ap.add_argument("--res", type=float, default=0.03, help="grid resolution (m/px)")
    ap.add_argument("--slice", type=float, nargs=2, default=[0.12, 2.0], metavar=("LO", "HI"),
                    help="height band (m above floor) whose points block the robot")
    ap.add_argument("--wall-band", type=float, nargs=2, default=[1.1, 2.4], metavar=("LO", "HI"),
                    help="height band used to find walls: above most furniture, below the ceiling")
    ap.add_argument("--any-angle", action="store_true",
                    help="keep wall segments at any angle (default: only X/Y-aligned walls after Manhattan alignment)")
    ap.add_argument("--floor-band", type=float, default=0.08, help="|z| below this = floor evidence (m)")
    ap.add_argument("--min-wall", type=float, default=0.8, help="drop wall segments shorter than this (m)")
    ap.add_argument("--wall-thickness", type=float, default=0.12)
    ap.add_argument("--no-manhattan", action="store_true", help="don't rotate the scene to align walls with X/Y")
    args = ap.parse_args()

    out = args.out or (args.scene / "out" if args.scene else args.ply.parent / "out")
    out.mkdir(parents=True, exist_ok=True)

    # 1. load + clean points --------------------------------------------------------------
    if args.scene:
        m = load_colmap_model(args.sparse or args.scene / "sparse" / "0")
        keep = (m["track_len"] >= 3) & (m["err"] < 2.0)
        source = "colmap"
    else:
        m = load_splat_ply(args.ply)
        keep = np.ones(len(m["xyz"]), bool)
        source = "splat_ply"
    xyz, rgb = m["xyz"][keep], m["rgb"][keep]
    inl = remove_outliers(xyz)
    xyz, rgb = xyz[inl], rgb[inl]
    print(f"{source}: {len(m['xyz'])} points -> {len(xyz)} after filtering")

    # 2. level the floor -------------------------------------------------------------------
    T, floor_mask, info = level_floor(xyz, m["cam_centers"], m["cam_ups"], args.up,
                                      args.cam_height, args.units_scale)
    P = apply(T, xyz)
    cams = apply(T, m["cam_centers"]) if m["cam_centers"] is not None else None
    print(f"floor: {info['floor_inliers']} inliers, horizontal planes at {np.round(info['plane_heights'], 2)} m,"
          f" scale guess {info['scale_guess']:.3f}")

    lo, hi = args.slice
    band = (P[:, 2] > lo) & (P[:, 2] < hi)

    # 3. Manhattan alignment: rotate so the dominant wall direction is the X axis -------
    yaw = 0.0
    if not args.no_manhattan:
        wb = (P[:, 2] > args.wall_band[0]) & (P[:, 2] < args.wall_band[1])
        if wb.sum() < 200:  # too little up high: fall back to the full obstacle band
            wb = band
        g0 = Grid(P[wb, :2], args.res, pad=0.3)
        d0 = cv2.GaussianBlur(g0.hist(P[wb, :2]), (0, 0), 1.0)
        m0 = otsu_mask(d0)
        yaw = dominant_yaw(m0, args.res)
        T = rotz(-yaw) @ T
        P = apply(T, xyz)
        cams = apply(T, m["cam_centers"]) if cams is not None else None
    print(f"manhattan yaw correction: {np.degrees(yaw):.1f} deg")

    # debug 00: side views before / after leveling
    save_leveling_png(out / "walls_00_leveling.png", xyz, P, floor_mask, args.slice)

    # 4. top-down images -------------------------------------------------------------------
    grid = Grid(P[:, :2][(P[:, 2] < hi)], args.res, pad=0.5)
    occ_pts = P[band]
    density = grid.hist(occ_pts[:, :2])
    floor_pts = P[np.abs(P[:, 2]) < args.floor_band]
    floor_density = grid.hist(floor_pts[:, :2])
    zmax = grid.max_img(occ_pts[:, :2], occ_pts[:, 2])

    # colored top-down view: mean color of points below the slice ceiling
    below = P[:, 2] < hi
    cnt = grid.hist(P[below, :2])
    rgb_img = np.stack([grid.hist(P[below, :2], rgb[below, 2 - i].astype(np.float32)) for i in range(3)], -1)
    rgb_img = (rgb_img / np.maximum(cnt, 1)[..., None]).astype(np.uint8)
    rgb_img = cv2.dilate(rgb_img, np.ones((3, 3), np.uint8))
    rgb_img[cv2.dilate((cnt > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0] = 20

    def draw_cams(img):
        if cams is not None:
            r, c = grid.to_px(cams[:, :2])
            for rr, cc in zip(r, c):
                cv2.circle(img, (int(cc), int(rr)), 1, C_CAM, -1)
        return img

    cv2.imwrite(str(out / "topdown_rgb.png"), rgb_img)  # unlabeled, 1 px per cell (used by plan.py)
    cv2.imwrite(str(out / "walls_01_topdown_rgb.png"),
                label(upscale(draw_cams(rgb_img.copy())), "1. Top-down view of the point cloud (blue = camera path)"))

    dens_blur = cv2.GaussianBlur(density, (0, 0), 1.0)
    cv2.imwrite(str(out / "walls_02_density.png"),
                label(upscale(colorize(log_norm(dens_blur))),
                      f"2. Density of points {lo:.2f}-{hi:.1f} m above the floor"))

    # 5. threshold (Otsu on log-density of non-empty pixels) ---------------------------
    k = lambda m_: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(m_ / args.res) + 1,) * 2)
    thr_mask = otsu_mask(dens_blur)
    cv2.imwrite(str(out / "walls_03_threshold.png"), label(upscale(cv2.cvtColor(thr_mask, cv2.COLOR_GRAY2BGR)),
                                                           "3. Otsu threshold"))

    # 6. morphology: close gaps, remove specks -------------------------------------------
    morph = clean_mask(thr_mask, k, close_m=0.06, min_area=0.04 / args.res ** 2)
    cv2.imwrite(str(out / "walls_04_morphology.png"), label(upscale(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)),
                                                            "4. Morphology (close + open, small blobs removed)"))

    # 7. occupancy grid: free = floor seen (or camera path), not occupied -----------------
    floor_seen = (cv2.GaussianBlur(floor_density, (0, 0), 1.5) > 0.05).astype(np.uint8) * 255
    if cams is not None:
        r, c = grid.to_px(cams[:, :2])
        for rr, cc_ in zip(r, c):
            cv2.circle(floor_seen, (int(cc_), int(rr)), int(0.3 / args.res), 255, -1)
    floor_seen = cv2.morphologyEx(floor_seen, cv2.MORPH_CLOSE, k(0.4))
    floor_seen = fill_holes(floor_seen)
    occupied = morph > 0
    free = (floor_seen > 0) & ~occupied
    occ = np.full(occupied.shape, 2, np.uint8)  # 0 free, 1 occupied, 2 unknown
    occ[free] = 0
    occ[occupied] = 1
    np.save(out / "occupancy.npy", occ)
    occ_vis = occupancy_bgr(occ)
    cv2.imwrite(str(out / "walls_05_occupancy.png"),
                label(upscale(draw_cams(occ_vis.copy())), "5. Occupancy grid (white free, black occupied, gray unknown)"))

    # 8. walls: high band only (above furniture) -> Hough lines -> merged segments ---------
    wlo, whi = args.wall_band
    wpts = P[(P[:, 2] > wlo) & (P[:, 2] < whi)]
    wall_dens = cv2.GaussianBlur(grid.hist(wpts[:, :2]), (0, 0), 1.0)
    wall_bin = clean_mask(otsu_mask(wall_dens), k, close_m=0.1, min_area=0.03 / args.res ** 2)
    lines = cv2.HoughLinesP(wall_bin, 1, np.pi / 180, threshold=int(0.25 / args.res),
                            minLineLength=int(0.5 / args.res), maxLineGap=int(0.3 / args.res))
    lines = np.zeros((0, 4), int) if lines is None else lines.reshape(-1, 4)
    if not (args.no_manhattan or args.any_angle):  # keep only X/Y-aligned lines
        ang = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])) % 90
        lines = lines[np.minimum(ang, 90 - ang) < 8]
    hough_vis = np.hstack([colorize(log_norm(wall_dens)), cv2.cvtColor(wall_bin // 3, cv2.COLOR_GRAY2BGR)])
    world_lines = []
    for x1, y1, x2, y2 in lines:
        cv2.line(hough_vis, (int(x1) + grid.w, int(y1)), (int(x2) + grid.w, int(y2)), (0, 255, 0), 1, cv2.LINE_AA)
        wx, wy = grid.to_world([y1, y2], [x1, x2])
        world_lines.append([wx[0], wy[0], wx[1], wy[1]])
    cv2.imwrite(str(out / "walls_06_hough.png"),
                label(upscale(hough_vis, 1400),
                      f"6. Wall band {wlo}-{whi} m: density | Hough ({len(lines)} axis-aligned segments)"))

    segs = merge_segments(world_lines, ang_tol_deg=8, off_tol=0.25, gap_tol=0.5, snap_deg=8)
    segs = [s for s in segs if np.linalg.norm(s[1] - s[0]) >= args.min_wall]

    def height_near(mask):
        z = zmax[(mask > 0) & np.isfinite(zmax)]
        return float(np.percentile(z, 90)) if len(z) else hi

    walls, wall_mask = [], np.zeros_like(morph)
    for i, (p0, p1) in enumerate(sorted(segs, key=lambda s: -np.linalg.norm(s[1] - s[0]))):
        (r0, r1), (c0, c1) = grid.to_px(np.array([p0, p1]))
        m1 = np.zeros_like(morph)
        cv2.line(m1, (int(c0), int(r0)), (int(c1), int(r1)), 255, max(1, int(args.wall_thickness / args.res)))
        wall_mask |= m1
        walls.append(dict(id=i, p0=[float(v) for v in p0], p1=[float(v) for v in p1],
                          length=float(np.linalg.norm(p1 - p0)), thickness=args.wall_thickness,
                          height=float(whi)))

    # 9. obstacles = occupied blobs not explained by walls ---------------------------------
    obst_mask = (morph > 0) & (cv2.dilate(wall_mask, k(0.1)) == 0)
    obst_mask = cv2.morphologyEx(obst_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, k(0.03))
    contours, _ = cv2.findContours(obst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    for cnt_ in contours:
        if cv2.contourArea(cnt_) * args.res ** 2 < 0.03:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(cnt_)
        m1 = np.zeros_like(morph)
        cv2.drawContours(m1, [cnt_], -1, 255, -1)
        wx, wy = grid.to_world(cy, cx)
        obstacles.append(dict(id=len(obstacles), center=[float(wx), float(wy)],
                              size=[float(max(w, 1) * args.res), float(max(h, 1) * args.res)],
                              yaw=float(-np.radians(ang)), height=height_near(m1),
                              contour=[list(map(float, grid.to_world(p[0][1], p[0][0]))) for p in cnt_]))

    # 10. final overlay with wall ids + lengths (pick one to tape-measure) ---------------
    final = upscale(rgb_img.copy())
    f = final.shape[1] / grid.w
    for o in obstacles:
        pts = np.array([grid.to_px(np.array([p]))[::-1] for p in o["contour"]])[:, :, 0]
        cv2.polylines(final, [((pts + 0.5) * f).astype(np.int32)], True, C_OBST, 2, cv2.LINE_AA)
    for w in walls:
        (r0, r1), (c0, c1) = grid.to_px(np.array([w["p0"], w["p1"]]))
        a, b = (int((c0 + .5) * f), int((r0 + .5) * f)), (int((c1 + .5) * f), int((r1 + .5) * f))
        cv2.line(final, a, b, C_WALL, max(2, int(w["thickness"] / args.res * f)), cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        txt = f"#{w['id']} {w['length']:.2f}m"
        cv2.putText(final, txt, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * f / 3, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(final, txt, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * f / 3, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "walls_07_walls.png"),
                label(final, f"7. Detected walls ({len(walls)}, red) and obstacles ({len(obstacles)}, orange)"))

    save_json(out / "walls.json", dict(
        source=source, units="approx_meters (scale from assumed camera height; refine in plan.py)",
        assumed_cam_height=args.cam_height if cams is not None else None,
        to_world=T.tolist(), grid=grid.meta(), slice=args.slice, manhattan_yaw_deg=float(np.degrees(yaw)),
        floor=info, walls=walls, obstacles=obstacles,
        cam_path=cams[:, :3].tolist() if cams is not None else None))
    print(f"{len(walls)} walls, {len(obstacles)} obstacles, grid {grid.w}x{grid.h} @ {args.res} m -> {out}")


def otsu_mask(density):
    """Otsu threshold on the log-density of non-empty pixels."""
    ld = np.log1p(density)
    nz = ld[ld > 0]
    if len(nz) == 0:
        return np.zeros(density.shape, np.uint8)
    t, _ = cv2.threshold((nz / nz.max() * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((ld / nz.max() * 255) > t).astype(np.uint8) * 255


def clean_mask(mask, k, close_m, min_area):
    """Close small gaps, open away specks, drop connected components under min_area px."""
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k(close_m))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k(0.03))
    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n_cc):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            m[cc == i] = 0
    return m


def fill_holes(mask):
    h, w = mask.shape
    ff = np.pad(mask, 1).copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
    return np.where(ff[1:-1, 1:-1] == 128, 0, 255).astype(np.uint8)


def occupancy_bgr(occ):
    img = np.empty(occ.shape + (3,), np.uint8)
    img[occ == 0] = C_FREE
    img[occ == 1] = (30, 30, 30)
    img[occ == 2] = (120, 120, 120)
    return img


def save_leveling_png(path, raw, P, floor_mask, band):
    """Side views (X-Z) of the raw cloud and the leveled cloud with the floor + slice band marked."""
    def panel(pts, title, mask, lines=None):
        W, H = 700, 420
        img = np.full((H, W, 3), 255, np.uint8)
        x, z = pts[:, 0], pts[:, 2]
        xl, xh = np.percentile(x, [1, 99])
        zl, zh = np.percentile(z, [1, 99])
        sc = min((W - 40) / max(xh - xl, 1e-6), (H - 60) / max(zh - zl, 1e-6))
        u = ((x - xl) * sc + 20).astype(int)
        v = (H - 20 - (z - zl) * sc).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        img[v[ok & ~mask], u[ok & ~mask]] = (150, 150, 150)
        img[v[ok & mask], u[ok & mask]] = (40, 40, 220)
        for zz, col in (lines or []):
            vv = int(H - 20 - (zz - zl) * sc)
            if 0 <= vv < H:
                cv2.line(img, (0, vv), (W, vv), col, 1)
        cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    left = panel(raw, "raw COLMAP frame (side view)", floor_mask)
    right = panel(P, "leveled: floor at Z=0, Z up", floor_mask,
                  [(0, (40, 40, 220)), (band[0], (0, 160, 0)), (band[1], (0, 160, 0))])
    cv2.imwrite(str(path), label(np.hstack([left, right]), "0. Floor plane fit (red) and leveling"))


if __name__ == "__main__":
    main()
