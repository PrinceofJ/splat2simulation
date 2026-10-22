"""Shared helpers: COLMAP binary readers, splat .ply reader, small IO utilities."""
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLMAP_EXE = ROOT / "tools" / "colmap" / "bin" / "colmap.exe"

# COLMAP camera model id -> number of params
_CAMERA_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cam_id, model_id, w, h = _read(f, "iiQQ")
            params = _read(f, "d" * _CAMERA_NUM_PARAMS[model_id])
            cams[cam_id] = dict(model_id=model_id, width=w, height=h, params=params)
    return cams


def read_images_bin(path):
    """Returns {image_id: dict(name, qvec(wxyz), tvec, camera_id)}; skips 2D observations."""
    images = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n2d,) = _read(f, "Q")
            f.seek(n2d * 24, 1)  # x, y (double), point3D_id (int64)
            images[image_id] = dict(name=name.decode(), qvec=np.array([qw, qx, qy, qz]),
                                    tvec=np.array([tx, ty, tz]), camera_id=cam_id)
    return images


def read_points3d_bin(path):
    """Returns (xyz Nx3 float64, rgb Nx3 uint8, error N, track_len N)."""
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        xyz = np.empty((n, 3))
        rgb = np.empty((n, 3), np.uint8)
        err = np.empty(n)
        tl = np.empty(n, np.int32)
        for i in range(n):
            _pid, x, y, z, r, g, b, e = _read(f, "QdddBBBd")
            (t,) = _read(f, "Q")
            f.seek(t * 8, 1)  # image_id (int32), point2D_idx (int32)
            xyz[i] = (x, y, z)
            rgb[i] = (r, g, b)
            err[i] = e
            tl[i] = t
    return xyz, rgb, err, tl


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def camera_centers_and_ups(images):
    """World-space camera centers and 'up' directions (COLMAP camera +y points down)."""
    centers, ups = [], []
    for im in images.values():
        R = qvec_to_rotmat(im["qvec"])  # world -> camera
        centers.append(-R.T @ im["tvec"])
        ups.append(-R.T @ np.array([0.0, 1.0, 0.0]))
    return np.array(centers), np.array(ups)


def load_colmap_model(sparse_dir):
    sparse_dir = Path(sparse_dir)
    xyz, rgb, err, tl = read_points3d_bin(sparse_dir / "points3D.bin")
    images = read_images_bin(sparse_dir / "images.bin")
    centers, ups = camera_centers_and_ups(images)
    return dict(xyz=xyz, rgb=rgb, err=err, track_len=tl, cam_centers=centers, cam_ups=ups)


def load_splat_ply(path, min_opacity=0.3):
    """Gaussian centers from a 3DGS-style .ply (x,y,z, opacity logit, f_dc_* SH colors)."""
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    keep = np.ones(len(xyz), bool)
    if "opacity" in names:
        keep &= 1 / (1 + np.exp(-v["opacity"])) >= min_opacity
    if "f_dc_0" in names:
        sh_c0 = 0.28209479177387814
        rgb = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1) * sh_c0 + 0.5
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    elif "red" in names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 200, np.uint8)
    return dict(xyz=xyz[keep], rgb=rgb[keep], cam_centers=None, cam_ups=None)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())
