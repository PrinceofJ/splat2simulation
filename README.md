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

Stages 6 and 7 have **not been run yet**. They were written on the laptop and are expected to need fixes on the
first cloud session (asset paths, 3DGRUT config names).

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
- `/World/Collision`: invisible static colliders. The ground is a box; walls are boxes; obstacles are extruded
  footprints (triangle meshes, ear-clipped).
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
Easiest: NVIDIA Brev's Isaac Sim launchable, or AWS g6/g5 with an Isaac Sim image. Ubuntu 22.04/24.04, ~50 GB free.

### 0. Setup

```bash
# Isaac Sim 5.x in its own Python 3.11 venv
python3.11 -m venv ~/isaac && source ~/isaac/bin/activate
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install imageio[ffmpeg] numpy scipy opencv-python-headless plyfile usd-core
# copy the project over (from the laptop):  scp -r C:\dev\splat2sim user@box:~/splat2sim  (skip .venv and tools)
```

Install Claude Code on the box too, so it can run the scripts and read the logs.

### 1. First, before anything else: check that NuRec loads a sample splat

Open a sample NuRec USDZ in Isaac Sim (see NVIDIA's NuRec docs or the 3DGRUT README samples) and confirm it
renders. If it doesn't, use the **mesh fallback** (step 4).

### 2. Train the splat (~30–60 min on an L40S)

```bash
bash cloud/train_3dgrut.sh data/room_sample        # ITERS=15000 for a quicker run
```

- Installs 3DGRUT into `~/3dgrut` (uv venv) and trains 3DGUT (MCMC config) on `images/` + `sparse/0`.
- Exports **NuRec** USDZ (`export_usd.format=nurec`, which Isaac Sim 5.x expects) with
  `apply_normalizing_transform=false`, so the splat stays in the COLMAP frame.
- Runs `stages/splat_align.py`: scaled ICP between the splat's Gaussians and the COLMAP points. If 3DGRUT moved
  the scene anyway, the measured transform goes into `splat_align.json` and export_usd compensates.
- Re-runs `export_usd.py --splat splat/splat.usdz`.

3DGRUT config keys change between versions. If `train.py` rejects one (for example `export_ply.*` or
`n_iterations`), check `configs/` and `threedgrut/export/README.md` in the clone and adjust the script.

### 3. Drive and record

```bash
source ~/isaac/bin/activate
python cloud/drive.py --scene data/room_sample --headless              # chase cam + onboard cam
python cloud/drive.py --scene data/room_sample --main overview          # fixed overview camera
python cloud/drive.py --scene data/room_sample --livestream             # watch via the WebRTC client
```

- Spawns Nova Carter at the plan's start pose. It tries several asset paths because the layout moved
  between releases; override with `--robot-usd`. Wheel joints are found by name (`joint_wheel_left/right`
  or anything containing "wheel" + "left/right").
- Proportional controller: heading error → angular velocity; distance → linear velocity, damped by cos²(heading
  error). Converted to wheel speeds with r = 0.14 m and wheel base 0.4132 m.
- Onboard camera: the first robot camera whose path contains "front"/"left". If none is found, a camera is mounted
  on the chassis. Override with `--onboard-camera`.
- Records `main.mp4`, `onboard.mp4`, and `demo_inset.mp4` (main view with the onboard view inset), plus
  `trajectory.csv`.
- Flags: `--show-proxy` shows the detected walls and obstacles over the splat, and `--hide-path` hides the
  path line.

### 4. Fallback if NuRec won't load the splat

Use a mesh as the visual instead: for example 3DGRUT/gsplat mesh extraction, or a Polycam/Scaniverse mesh
export aligned with `splat_align.py`. Reference it from `/World/Splat` the same way, or just run
`drive.py --show-proxy` to show the detected geometry.

---

## Known limitations

- The scale guess assumes camera height. Use `--wall-id/--wall-length` for real meters.
- Walls must be roughly Manhattan (X/Y-aligned). Use `walls.py --any-angle` for other layouts.
- Glass, mirrors, and featureless walls produce few points, so they can be missed. Clear the floor before
  capturing, because clutter becomes obstacles.
- The obstacle collider height comes from the tallest points in each blob, with a minimum of 0.25 m.
