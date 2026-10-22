#!/usr/bin/env bash
# Train a 3DGUT Gaussian splat with NVIDIA 3DGRUT from the laptop's COLMAP output and export a
# NuRec USDZ that Isaac Sim 5.x can render. Run on the rented RTX box (Linux). NOT run on the laptop.
#
#   bash cloud/train_3dgrut.sh data/room_sample            # scene folder with images/ + sparse/0
#   ITERS=15000 bash cloud/train_3dgrut.sh data/my_room     # quicker run
#
# Produces:
#   <scene>/out/splat/splat.usdz       NuRec USDZ, still in the COLMAP frame (no normalizing transform)
#   <scene>/out/splat/splat.ply        same Gaussians as .ply (for the alignment check / walls.py --ply)
#   <scene>/out/splat_align.json       measured COLMAP->splat offset (should be ~identity)
# then re-run export_usd.py --splat splat/splat.usdz (printed at the end).
set -euo pipefail

SCENE=$(realpath "${1:?usage: train_3dgrut.sh <scene folder>}")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GRUT=${GRUT_DIR:-$HOME/3dgrut}
ITERS=${ITERS:-30000}
CONFIG=${CONFIG:-apps/colmap_3dgut_mcmc.yaml}   # 3DGUT (not 3DGRT): rasterized, what NuRec expects
NAME=$(basename "$SCENE")
OUT="$SCENE/out/splat"
mkdir -p "$OUT"

[ -d "$SCENE/images" ] && [ -f "$SCENE/sparse/0/images.bin" ] || {
  echo "Expected $SCENE/images and $SCENE/sparse/0 (run stages/poses.py on the laptop first)"; exit 1; }

# ---- 1. install 3DGRUT once (uv-based installer; uses the system CUDA 12.x toolkit) -------------
if [ ! -d "$GRUT" ]; then
  git clone --recursive https://github.com/nv-tlabs/3dgrut.git "$GRUT"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  (cd "$GRUT" && ./install_env_uv.sh)
fi
# shellcheck disable=SC1091
source "$GRUT/.venv/bin/activate"
cd "$GRUT"
[ -f "configs/$CONFIG" ] || { echo "configs/$CONFIG missing, falling back to apps/colmap_3dgut.yaml"; CONFIG=apps/colmap_3dgut.yaml; }

# ---- 2. train + export ---------------------------------------------------------------------
# downsample_factor=1: poses.py already resized images to <=1600 px.
# apply_normalizing_transform=false keeps the splat in the COLMAP frame; scene.usd's /World/Splat
# prim carries the COLMAP -> world (floor at Z=0, meters) transform computed by walls.py.
python train.py --config-name "$CONFIG" \
  path="$SCENE" out_dir="$SCENE/out/3dgrut_runs" experiment_name="$NAME" \
  dataset.downsample_factor=1 n_iterations="$ITERS" \
  export_usd.enabled=true export_usd.format=nurec export_usd.apply_normalizing_transform=false \
  export_usd.path="$OUT/splat.usdz" \
  export_ply.enabled=true export_ply.path="$OUT/splat.ply"

# Some 3DGRUT versions ignore export_*.path and write into the run folder: collect the newest files.
RUN=$(ls -td "$SCENE/out/3dgrut_runs/$NAME"*/ | head -1)
[ -f "$OUT/splat.usdz" ] || cp "$(find "$RUN" -name '*.usdz' | head -1)" "$OUT/splat.usdz"
[ -f "$OUT/splat.ply" ]  || cp "$(find "$RUN" -name '*.ply'  | head -1)" "$OUT/splat.ply" || true
if [ ! -f "$OUT/splat.usdz" ]; then
  CKPT=$(find "$RUN" -name 'ckpt_last.pt' -o -name '*.pt' | sort | tail -1)
  echo "No USDZ from training; exporting from checkpoint $CKPT"
  python -m threedgrut.export.scripts.export_usd --checkpoint "$CKPT" --output "$OUT/splat.usdz" \
    --format nurec --no-transform
fi
deactivate

# ---- 3. check the splat is in the COLMAP frame, then attach it to scene.usd ------------------
cd "$ROOT"
PY=${PY:-python3}
[ -f "$OUT/splat.ply" ] && $PY stages/splat_align.py --scene "$SCENE" --ply "$OUT/splat.ply" || \
  echo "WARNING: no splat.ply, skipped alignment check"
$PY stages/export_usd.py --scene "$SCENE" --splat splat/splat.usdz
echo
echo "Done: $OUT/splat.usdz attached to $SCENE/out/scene.usd"
