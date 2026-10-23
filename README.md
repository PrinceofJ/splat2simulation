# Splat2Sim

Phone scan of a room → COLMAP poses → detected walls and obstacles → A* path → `scene.usd` →
Nova Carter drives the path in Isaac Sim over a Gaussian-splat render of the real room.

A minimal vertical slice for a ~45 s demo video. See [SPLAT2SIM_KICKOFF.md](SPLAT2SIM_KICKOFF.md) for the why.
It's a **planner**, not a learned policy.

```
phone video ─► poses (COLMAP) ─► cloud: 3DGRUT splat ─► NuRec USDZ (visual)
                   │                                           │
                   └► walls ─► occupancy + wall segments ─► plan (A*) ─► export_usd ─► scene.usd
                                                                                       │
                                                         cloud: Isaac Sim drive.py ─► MP4s
```

| Stage | Script | Runs on | Output (in `<scene>/out/`) |
|---|---|---|---|
| 1 poses | `stages/poses.py` | laptop | `<scene>/images/`, `<scene>/sparse/0/` |
| 2 preview *(optional)* | not built yet: gsplat needs a CUDA toolkit on Windows (see below) | — | — |
| 3 walls | `stages/walls.py` | laptop (CPU) | `walls.json`, `occupancy.npy`, `walls_00..07_*.png` |
| 4 plan | `stages/plan.py` | laptop (CPU) | `waypoints.json`, `plan_01/02_*.png` |
| 5 export_usd | `stages/export_usd.py` | laptop (CPU) | `scene.usd`, `splat/placeholder.usda` |
| 6 train | `cloud/train_3dgrut.sh` | cloud RTX | `splat/splat.usdz`, `splat/splat.ply`, `splat_align.json` |
| 7 drive | `cloud/drive.py` | cloud RTX | `video/main.mp4`, `video/onboard.mp4`, `video/demo_inset.mp4`, `video/trajectory.csv` |

Stages 6 and 7 were first run end to end on a Brev L40S (Isaac Sim 5.1.0) on 2026-10-23 with the room sample:
the 30k-iteration splat renders through NuRec and lines up with the detected walls and obstacles.

---

## Laptop (Windows, RTX 3050 4 GB)

### Setup (once)

The project lives at `C:\dev\splat2sim`, outside OneDrive so datasets and the venv don't sync.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

COLMAP: download `colmap-x64-windows-cuda.zip` from the
[COLMAP releases](https://github.com/colmap/colmap/releases) and unzip it to `tools\colmap\`
(so `tools\colmap\bin\colmap.exe` exists). Tested with 4.2.1.

> **GPU note:** with NVIDIA driver 546.92 (CUDA 12.3), COLMAP 4.2.1's CUDA build crashes silently
> (exit code `0xC0000409`). `poses.py` detects this and falls back to CPU SIFT and matching, which takes about
> 15 min for 311 images. Updating the NVIDIA driver (GeForce Experience or nvidia.com) should enable the GPU path.

### Sample scene (end-to-end test)

Mip-NeRF 360 **room** (311 photos), from the Hugging Face mirror `nvs-bench/mipnerf360`.
It's already downloaded in `data/room_sample/raw_images` (plus the authors' COLMAP model in `sparse_ref/0`).

```powershell
$py = ".venv\Scripts\python"
& $py stages/poses.py --images data/room_sample/raw_images --work data/room_sample --matcher sequential
& $py stages/walls.py --scene data/room_sample
& $py stages/plan.py --scene data/room_sample
& $py stages/export_usd.py --scene data/room_sample
```

This sample was shot looking *inward* at a coffee table, so few points land on walls: expect ~1 wall and several
furniture obstacles. A capture of your own room, aimed at the walls, gives a much better wall map.

### Your own room

1. Record a 1–2 min video (see the capture checklist in the kickoff doc) and copy it to `data/my_room/scan.mp4`.
2. `poses.py --video data/my_room/scan.mp4 --work data/my_room`
   - It keeps the sharpest frame in each window (`--frames 250`), resizes to 1600 px, and uses the sequential
     matcher plus COLMAP's global mapper.
   - If the model splits into several pieces, try `--mapper incremental` or capture with more overlap.
3. `walls.py --scene data/my_room`, then open `out/walls_07_walls.png`. Every wall has an id and a length.
4. Tape-measure one of those walls, then plan in true meters:
   `plan.py --scene data/my_room --wall-id 0 --wall-length 3.62 [--start x,y --goal x,y]`.
   Without `--start/--goal` it picks the two farthest-apart reachable spots.
5. `export_usd.py --scene data/my_room`
6. Copy the project to the cloud box (`data/my_room/images`, `sparse/`, and `out/` are enough).

### What each stage does

**walls.py** (debug PNG per step, made for the demo video)
- `00_leveling`: cleans points (track length ≥ 3, reprojection error < 2 px, statistical outlier removal), then
  fits horizontal planes with RANSAC (normal within 20° of the cameras' mean up direction). The **lowest**
  large plane is taken as the floor and rotated to Z = 0, Z up. Scale is a first guess that assumes the phone was
  held `--cam-height 1.4` m above the floor.
- Manhattan alignment: a Hough pass over the wall band finds the dominant wall direction, and the scene is
  rotated so walls run along X and Y.
- `01_topdown_rgb`: colored top-down view with the camera path.
- `02_density`, `03_threshold`, `04_morphology`: density of points 0.12–2.0 m above the floor (anything that
  blocks the robot), Otsu threshold on log-density, then close/open and speck removal.
- `05_occupancy`: free = floor seen (or camera path walked) and not occupied; everything else is unknown, which
  the planner treats as blocked.
- `06_hough`: walls come from a **high band (1.1–2.4 m)** above most furniture. Probabilistic Hough keeps
  axis-aligned segments only, which are clustered by direction and offset and merged along their length.
- `07_walls`: walls (red, with id and length) and obstacles (orange: occupied blobs not explained by a wall).

**plan.py**: inflates obstacles by `--robot-radius` (0.3 m) with a distance transform. A* runs on an
8-connected grid, with a cost that rises within `--clearance` (0.3 m) of obstacles. Line-of-sight shortcutting
then reduces the path to a few corner waypoints. The scale factor is stored in `waypoints.json`.

**export_usd.py**: writes a Z-up, meters scene:
- `/World/Collision`: invisible static colliders. The ground is a plane (a thin box made Nova Carter's wheels
  jitter and slide); walls are boxes; obstacles are extruded footprints (triangle meshes, ear-clipped).
- `/World/Proxy`: visible stand-ins, shown only when no splat is attached (or with `--proxy on`).
- `/World/Splat`: a reference to the splat USDZ, carrying the COLMAP → world transform. It points at an empty
  `splat/placeholder.usda` until you run `export_usd.py --splat splat/splat.usdz`.
- `/World/Plan`: the path curve and waypoint spheres. Waypoints are also stored as the `splat2sim:waypoints`
  attribute, plus a `Start` pose.

### Optional: low-res gsplat preview (not built yet)

gsplat compiles CUDA kernels on first use, which needs a CUDA toolkit no newer than your driver
(12.3 today) plus a matching PyTorch. Neither is installed. The easiest route is WSL Ubuntu 22.04 (already
installed) with `pip install torch gsplat` and the gsplat `simple_trainer.py` at
`--data_factor 4 --max_steps 7000`. Updating the Windows driver first widens the CUDA options.

---

## Cloud (rented RTX GPU, one ~3–5 h session)

**GPU must be RTX-class** (L40S, A10G, L4). A100/H100 have no RT cores and can't render Isaac Sim.
What was used: **NVIDIA Brev, L40S 48 GB, "Isaac Lab 2.3.2 + Isaac Sim 5.1.0" launchable**.

### How the Brev box is laid out

- The terminal you get (and Claude Code) runs **inside a Docker container**. Isaac Sim is a binary install at
  `/root/isaacsim`; run scripts with `/root/isaacsim/python.sh`, not a venv.
- The VM host's filesystem is mounted in the container at `/workspace`, so the host's `/home/ubuntu` is
  `/workspace/home/ubuntu` from inside. `scp`/`brev copy` from the laptop land on the **host**, not in the container.
- The project lives at `/root/splat2sim` on the container's own filesystem. **It is not a mounted volume**: it
  survives a pause only if Brev keeps the container. Before pausing, copy results to `/workspace/home/ubuntu`
  (see "Before pausing" below).
- `/dev/shm` is only 64 MB, which breaks PyTorch DataLoader workers. `train_3dgrut.sh` handles it.
- There is no system CUDA toolkit (no `nvcc`); 3DGRUT downloads its own.

Everything installed lives under the project folder:

| Path | What |
|---|---|
| `.tools/uv/` | uv, plus `.tools/uv-python/` (its Python 3.11) and `.tools/uv-cache/` |
| `.venv/` | Python 3.11 for the pipeline scripts (`stages/*.py`): numpy, scipy, opencv-python-headless, plyfile, usd-core |
| `.tools/3dgrut/` | 3DGRUT clone, with its own `.venv/` and a local CUDA 12.8 toolkit (~18 GB total) |

`usd-core` must **not** go into Isaac Sim's Python (it clashes with Isaac's own `pxr`). Isaac Sim's Python already
ships imageio, imageio-ffmpeg and pillow, which is all `drive.py` needs.

### 0. Setup

Copy the project to the box (`images/`, `sparse/` and `out/` are enough for a scene), then:

```bash
cd /root/splat2sim
# uv + the pipeline venv, all inside the project
export UV_INSTALL_DIR=$PWD/.tools/uv UV_PYTHON_INSTALL_DIR=$PWD/.tools/uv-python UV_CACHE_DIR=$PWD/.tools/uv-cache
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
.tools/uv/uv venv --python 3.11 .venv
.tools/uv/uv pip install --python .venv/bin/python numpy scipy opencv-python-headless plyfile usd-core
```

3DGRUT installs itself on the first `train_3dgrut.sh` run (next step). Install Claude Code on the box too, so it
can run the scripts and read the logs.

If the walls, plan or export need re-running on the box: `.venv/bin/python stages/export_usd.py --scene ...`.

### 1. Train the splat (~20 min on an L40S at 30k iterations)

```bash
PY=.venv/bin/python bash cloud/train_3dgrut.sh data/room_sample             # default ITERS=30000
ITERS=2000 PY=.venv/bin/python bash cloud/train_3dgrut.sh data/room_sample  # ~1 min smoke test
```

- First run only: clones 3DGRUT into `.tools/3dgrut` and installs it with its local-CUDA route
  (`FORCE_LOCAL_CUDA=1 CUDA_VERSION=12 ./scripts/create_venv_cuda.sh`, then `./install_env_uv.sh`). This takes
  ~15 min; most of it is downloading CUDA 12.8 (the ~5 GB runfile is cached in `/tmp`, delete it afterwards)
  and compiling tiny-cuda-nn.
- Trains 3DGUT (`apps/colmap_3dgut_mcmc.yaml`) on `images/` + `sparse/0`.
- If `/dev/shm` is under 2 GB it passes `num_workers=0` and `test_last=false` (the test-set renderer hard-codes
  8 workers). With a bigger shm (`docker run --shm-size=8g`) you get workers and the test PSNR back.
- Exports **NuRec** USDZ (`export_usd.format=nurec`, what Isaac Sim 5.x renders) with
  `apply_normalizing_transform=false`, so the splat stays in the COLMAP frame.
- Runs `stages/splat_align.py` with `PY`: scaled ICP between the splat's Gaussians and the COLMAP points. If it
  measures an offset, it goes into `splat_align.json` and export_usd compensates. (On the 30k sample it reported
  ~3 cm / 0.3°, which is probably ICP noise from floaters; the 2k run measured identity.)
- Re-runs `export_usd.py --splat splat/splat.usdz`, which hides the proxy by default.

3DGRUT config keys change between versions. If `train.py` rejects one, check `configs/` and
`threedgrut/export/README.md` in the clone and adjust the script. All keys were accepted at 3DGRUT commit `e6f4552`.

### 2. Drive and record

```bash
/root/isaacsim/python.sh cloud/drive.py --scene data/room_sample --headless                  # chase cam + onboard cam
/root/isaacsim/python.sh cloud/drive.py --scene data/room_sample --headless --show-proxy     # + detected walls/obstacles
/root/isaacsim/python.sh cloud/drive.py --scene data/room_sample --headless --main overview  # top-down from below the ceiling
/root/isaacsim/python.sh cloud/drive.py --scene data/room_sample --livestream                # watch via the WebRTC client
```

Each run overwrites `out/video/`; rename it between runs. Rendering runs at roughly 6–9 s wall time per sim second.

- Spawns Nova Carter at the plan's start pose. In 5.1 the asset is
  `/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd` on the S3 asset root (the first of the candidates
  `drive.py` tries); override with `--robot-usd`. Wheel joints are `joint_wheel_left/right`.
- Proportional controller: heading error → angular velocity; distance → linear velocity, damped by cos²(heading
  error). Converted to wheel speeds with r = 0.14 m and wheel base 0.4132 m.
- Onboard camera: `chassis_link/sensors/front_hawk/left/camera_left`. Override with `--onboard-camera`.
- Main camera: chase cam (`--chase-back 1.4 --chase-height 1.7`), or `--main overview`, a straight-down view at
  `--overview-height 2.2` with the focal length fitted to the path. Keep both cameras inside the scanned room:
  outside it they see only floaters.
- Records `main.mp4`, `onboard.mp4`, and `demo_inset.mp4` (main view with the onboard view inset), plus
  `trajectory.csv`. `--hide-path` hides the path line.

### 3. Fallback if NuRec won't load the splat

Not needed on Isaac Sim 5.1: NuRec rendered the 3DGRUT USDZ directly and depth-composites it with the robot and
proxies. If a future version breaks it, use a mesh as the visual instead (3DGRUT/gsplat mesh extraction, or a
Polycam/Scaniverse mesh export aligned with `splat_align.py`), referenced from `/World/Splat` the same way, or
just run `drive.py --show-proxy` to show the detected geometry.

### Before pausing the instance

Bundle the results (skipping the 18 GB of reinstallable tools, the checkpoints and the input images) and put the
tarball on the host disk, where it survives even if the container is recreated:

```bash
cd /root && tar czf splat2sim_session.tgz \
  --exclude=splat2sim/.tools --exclude=splat2sim/.venv \
  --exclude='splat2sim/data/*/images' --exclude='splat2sim/data/*/sparse' --exclude='splat2sim/data/*/out/3dgrut_runs*' \
  splat2sim
cp splat2sim_session.tgz /workspace/home/ubuntu/
# then on the laptop:  scp <instance>:/home/ubuntu/splat2sim_session.tgz .
```

Keep `out/3dgrut_runs/` (~2 GB) too if you may want to re-export the USDZ without retraining.

---

## Known limitations

- The scale guess assumes camera height. Use `--wall-id/--wall-length` for real meters.
- Walls must be roughly Manhattan (X/Y-aligned). Use `walls.py --any-angle` for other layouts.
- Glass, mirrors, and featureless walls produce few points, so they can be missed. Clear the floor before
  capturing, because clutter becomes obstacles.
- The obstacle collider height comes from the tallest points in each blob, with a minimum of 0.25 m.
