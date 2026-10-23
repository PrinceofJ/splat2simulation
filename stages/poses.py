"""Stage 1 - poses: run COLMAP on a phone video or a folder of photos.

Output layout (what 3DGRUT / gsplat expect):
    <work>/images/*.jpg      frames resized to --max-size
    <work>/database.db
    <work>/sparse/0/{cameras,images,points3D}.bin

Examples:
    python stages/poses.py --video data/my_room/scan.mp4 --work data/my_room
    python stages/poses.py --images data/room_sample/images --work data/room_sample
    python stages/poses.py --work data/my_room --remap --mapper incremental   # redo mapping only
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from common import COLMAP_EXE, camera_centers_and_ups, read_images_bin


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def resize_max(img, max_size):
    h, w = img.shape[:2]
    s = max_size / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def extract_video_frames(video, out_dir, target_frames, max_size):
    """Split the video into target_frames windows and keep the sharpest frame of each."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        sys.exit(f"Could not read frames from {video}")
    window = max(1, n // target_frames)
    best, best_score, saved, i = None, -1.0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = sharpness(cv2.resize(frame, None, fx=0.25, fy=0.25))
        if score > best_score:
            best, best_score = frame, score
        i += 1
        if i % window == 0:
            cv2.imwrite(str(out_dir / f"frame_{saved:05d}.jpg"), resize_max(best, max_size),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            best, best_score = None, -1.0
    cap.release()
    print(f"Extracted {saved} sharp frames from {n} video frames (window={window})")
    return saved


def copy_images(src, out_dir, max_size):
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in Path(src).iterdir() if p.suffix.lower() in exts)
    for p in files:
        img = cv2.imread(str(p))
        cv2.imwrite(str(out_dir / (p.stem + ".jpg")), resize_max(img, max_size),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Copied {len(files)} images (max side {max_size}px)")
    return len(files)


def colmap(*args):
    cmd = [str(COLMAP_EXE), *map(str, args)]
    print("\n>>", " ".join(cmd), flush=True)
    t = time.time()
    subprocess.run(cmd, check=True)
    print(f"   ({time.time() - t:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--video", type=Path)
    src.add_argument("--images", type=Path)
    src.add_argument("--remap", action="store_true",
                     help="reuse <work>/images and database.db (features + matches) and only re-run the mapper")
    ap.add_argument("--work", type=Path, required=True, help="scene folder to write into")
    ap.add_argument("--max-size", type=int, default=1600, help="longest image side in px")
    ap.add_argument("--frames", type=int, default=250, help="frames to keep from a video")
    ap.add_argument("--matcher", choices=["auto", "sequential", "exhaustive"], default="auto")
    ap.add_argument("--mapper", choices=["global", "incremental"], default="global",
                    help="global (GLOMAP, fast) or incremental (classic, slower, more robust)")
    ap.add_argument("--camera-model", default="OPENCV")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    if not COLMAP_EXE.exists():
        sys.exit(f"COLMAP not found at {COLMAP_EXE}. See README, 'Laptop setup'.")

    if not (args.video or args.images or args.remap):
        ap.error("one of --video, --images or --remap is required")

    work = args.work
    img_dir = work / "images"
    db = work / "database.db"
    sparse = work / "sparse"
    gpu = "0" if args.no_gpu else "1"

    if args.remap:
        if not db.exists():
            sys.exit(f"--remap needs an existing {db} from a previous run")
        if sparse.exists():
            shutil.rmtree(sparse)
        sparse.mkdir()
        run_mapper(args.mapper, db, img_dir, sparse)
        return

    if args.images and args.images.resolve() == img_dir.resolve():
        n = len(list(img_dir.glob("*.jpg")))  # already in place: use as-is
    else:
        if img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True)
        n = (extract_video_frames(args.video, img_dir, args.frames, args.max_size) if args.video
             else copy_images(args.images, img_dir, args.max_size))

    db.unlink(missing_ok=True)
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()

    def extract(use_gpu):
        colmap("feature_extractor", "--database_path", db, "--image_path", img_dir,
               "--ImageReader.camera_model", args.camera_model, "--ImageReader.single_camera", "1",
               "--FeatureExtraction.use_gpu", use_gpu, "--FeatureExtraction.max_image_size", args.max_size)

    try:
        extract(gpu)
    except subprocess.CalledProcessError:
        if gpu == "0":
            raise
        # COLMAP's CUDA build needs a recent NVIDIA driver; older drivers crash with no message
        print("GPU feature extraction crashed (NVIDIA driver too old for COLMAP's CUDA build?). "
              "Retrying on CPU; update the driver to get GPU speed.")
        gpu = "0"
        db.unlink(missing_ok=True)
        extract(gpu)

    matcher = args.matcher
    if matcher == "auto":
        matcher = "sequential" if (args.video or n > 200) else "exhaustive"
    if matcher == "sequential":
        colmap("sequential_matcher", "--database_path", db, "--FeatureMatching.use_gpu", gpu,
               "--SequentialMatching.overlap", "15")
    else:
        colmap("exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", gpu)

    run_mapper(args.mapper, db, img_dir, sparse)


def run_mapper(mapper, db, img_dir, sparse):
    if mapper == "global":
        colmap("global_mapper", "--database_path", db, "--image_path", img_dir, "--output_path", sparse)
    else:
        colmap("mapper", "--database_path", db, "--image_path", img_dir, "--output_path", sparse)

    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        sys.exit("COLMAP produced no model. Try --mapper incremental or --matcher exhaustive.")
    if len(models) > 1:
        # the incremental mapper doesn't order models by size: make the biggest one sparse/0
        sizes = {p: len(read_images_bin(p / "images.bin")) for p in models}
        biggest = max(models, key=sizes.get)
        if biggest != sparse / "0":
            (sparse / "0").rename(sparse / "tmp")
            biggest.rename(sparse / "0")
            (sparse / "tmp").rename(biggest)
        print(f"WARNING: COLMAP split the scene into {len(models)} models (sizes {sorted(sizes.values(), reverse=True)}); "
              "using the biggest as sparse/0. More overlap in the capture usually fixes this.")
    colmap("model_analyzer", "--path", sparse / "0")
    check_cameras(sparse / "0", mapper)
    print(f"\nDone. Model: {sparse / '0'}")


def check_cameras(model, mapper):
    """Flag cameras placed implausibly far from the rest (a known global-mapper failure on hard video)."""
    centers, _ = camera_centers_and_ups(read_images_bin(model / "images.bin"))
    d = np.linalg.norm(centers - np.median(centers, 0), axis=1)
    bad = d > 5 * np.median(d) + 1e-9
    if bad.mean() > 0.03:
        hint = (" Re-run just the mapping step with: --remap --mapper incremental" if mapper == "global"
                else " The capture probably needs more overlap or less blur.")
        print(f"WARNING: {bad.sum()} of {len(centers)} cameras sit more than 5x the median distance from the "
              f"rest of the path, so the model is likely broken.{hint}")
    else:
        print(f"camera check: {bad.sum()} of {len(centers)} cameras look misplaced (ok)")


if __name__ == "__main__":
    main()
