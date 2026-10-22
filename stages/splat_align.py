"""Check that a trained splat lives in the same frame as the COLMAP points (and fix it if not).

3DGRUT may normalize camera poses. If the exported splat is offset/rotated/scaled relative to COLMAP,
this estimates the similarity transform COLMAP -> splat with scaled ICP and writes
<out>/splat_align.json, which export_usd.py folds into the /World/Splat transform.

    python stages/splat_align.py --scene data/room_sample --ply data/room_sample/out/splat/splat.ply
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import load_colmap_model, load_splat_ply, save_json


def umeyama(src, dst):
    """Similarity (s, R, t) minimizing |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    m = load_colmap_model(args.scene / "sparse" / "0")
    src = m["xyz"][(m["track_len"] >= 3) & (m["err"] < 2.0)]
    dst = load_splat_ply(args.ply, min_opacity=0.5)["xyz"]
    rng = np.random.default_rng(0)
    src = src[rng.choice(len(src), min(len(src), 30000), replace=False)]
    tree = cKDTree(dst[rng.choice(len(dst), min(len(dst), 300000), replace=False)])

    scale_ref = np.median(np.linalg.norm(src - np.median(src, 0), axis=1))
    d0 = np.median(tree.query(src)[0]) / scale_ref

    def icp(s, R, t):
        for _ in range(args.iters):
            d, i = tree.query(s * src @ R.T + t)
            keep = d < np.percentile(d, 80)
            s, R, t = umeyama(src[keep], tree.data[i[keep]])
        return np.median(tree.query(s * src @ R.T + t)[0]) / scale_ref, s, R, t

    # seeds: identity (expected), plus centroid/spread/PCA-axis guesses in case 3DGRUT normalized the scene
    seeds = [(1.0, np.eye(3), np.zeros(3))]
    dsub = tree.data
    cs, cd = src.mean(0), dsub.mean(0)
    s0 = np.sqrt(((dsub - cd) ** 2).sum(1).mean() / ((src - cs) ** 2).sum(1).mean())
    Vs, Vd = np.linalg.svd(src - cs, full_matrices=False)[2], np.linalg.svd(dsub - cd, full_matrices=False)[2]
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        F = np.diag([sx, sy, sx * sy])
        R0 = Vd.T @ F @ Vs
        seeds += [(s0, R0, cd - s0 * R0 @ cs)]
    seeds += [(s0, np.eye(3), cd - s0 * cs)]
    d1, s, R, t = min((icp(*seed) for seed in seeds), key=lambda r: r[0])
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"median NN distance (relative to scene size): before {d0:.4f}, after ICP {d1:.4f}")
    print(f"COLMAP->splat: scale {s:.4f}, rotation {angle:.2f} deg, translation {np.round(t, 4)}")

    identity = abs(s - 1) < 0.01 and angle < 0.5 and np.linalg.norm(t) / scale_ref < 0.01
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = s * R, t
    save_json(args.scene / "out" / "splat_align.json",
              dict(colmap_to_splat=M.tolist(), identity=bool(identity), nn_before=d0, nn_after=d1))
    print("splat is in the COLMAP frame (no fix needed)" if identity else
          "splat is offset from COLMAP: export_usd.py will compensate using splat_align.json")


if __name__ == "__main__":
    main()
