#!/usr/bin/env bash
# End-to-end: phone video -> isolated, cleaned object mesh.
#
#   ./scripts/reconstruct_object.sh --video inputs/object2.mp4 --name object2
#   ./scripts/reconstruct_object.sh --video inputs/object2.mp4 --name object2 --texture
#   ./scripts/reconstruct_object.sh --video inputs/object2.mp4 --name object2 --bilateral
#
# Runs frame extraction -> sharp-frame selection (optional bilateral filter) -> COLMAP SfM ->
# vanilla 3DGS -> U2Net masks -> SuGaR ->
# visual-hull carve -> cleanup -> watertight safeguard -> vertex colors, producing a
# WATERTIGHT, vertex-coloured isolated object mesh at output/final/<name>_final.ply.
#
# Object isolation (default): full-supervision SuGaR meshes the whole scene, then the mesh-space
# CARVE keeps only what projects inside the U2Net silhouettes, and pymeshfix seals the unobserved
# bottom into a watertight solid. (--gaussian-prune is an experimental alternative that isolates in
# Gaussian space with a mask-restricted SuGaR loss; lower fidelity on small objects at this VRAM cap.)
#
# With --texture, a stage bakes an observation-only texture atlas (weighted-median blend)
# into output/final/<name>/visual/. With --collision, CoACD convex-decomposes the watertight
# mesh into a physics collision asset + URDF in output/final/<name>/collision/.
#
# Spans three conda envs on purpose (sugar / colmap / seg) so no stage's
# dependencies can perturb the pinned torch 2.0.1 + pytorch3d 0.7.4 stack.
#
# VRAM policy (this box is a 4 GB GTX 1650):
#   * vanilla 3DGS trains at -r 2                   (measured 2.3 GB peak vs 3.9 GB at full res)
#   * SUGAR_MAX_IMG_SIZE is DERIVED (H2), not fixed. SuGaR keeps every GT image resident on the
#     GPU, so its footprint is n*w*h*3*4 bytes; a value that fits 239 frames OOMs at 400. The
#     budget itself is derived from *measured free VRAM* minus SuGaR's non-GT footprint, so
#     resolution adapts to the machine and the frame count. See derive_max_img_size below.
#   * refinement runs short by default; --refine long is opt-in.
# Speed policy:
#   * SUGAR_SDF_SAMPLES=300k (H1) instead of the upstream 1M. The SDF-regularization phase is
#     ~92% of wall-clock and is memory-bound, while the loss is a .mean() so its scale is
#     unchanged by the count. Measured 2.69x faster on the field-eval hot path of the real
#     object3 model (300k vs 1M); sample count cut 3.3x. Override with --sdf-samples (use
#     1000000 to reproduce stock SuGaR quality exactly). Mesh quality at 300k is author-suggested
#     (their own code comment) but not yet confirmed by a full end-to-end run here.
# Every stage writes a .done marker, so re-running resumes instead of redoing.

set -euo pipefail

SUGAR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Consume the setup wizard's machine-specific conda, CUDA, compiler and model
# locations automatically. Existing hand-configured systems continue to use
# the legacy defaults below when no generated profile exists.
SETUP_RUNTIME_ENV="${SUGARRUSH_SETUP_ENV:-$SUGAR_DIR/.setup/runtime.env}"
if [[ -f "$SETUP_RUNTIME_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$SETUP_RUNTIME_ENV"
fi
# Everything the pipeline reads/writes lives under SuGaR/ so the repo is self-contained.
# Videos in SuGaR/inputs/, scenes in SuGaR/scenes/, artifacts in SuGaR/output/. Only the
# conda location is machine-specific: override it with  CONDA_SH=/path/to/conda.sh  if needed.
CONDA_SH="${CONDA_SH:-/home/acer/miniforge3/etc/profile.d/conda.sh}"
SCENES_ROOT="$SUGAR_DIR/scenes"

VIDEO=""; NAME=""; FPS=10; TARGET=240; REFINE=short; NVERT=500000
GS_ITERS=7000; KEEP_RATIO=0.85; FORCE=0
# Optional first-stage experiment: apply a mild bilateral filter only after sharp-frame selection.
# Off by default because spatial denoising can also suppress the high-frequency signal used by SIFT,
# 3DGS and texture reconstruction. OpenCV lives in the isolated seg environment.
BILATERAL=0; BILATERAL_DIAMETER=3; BILATERAL_SIGMA_COLOR=10; BILATERAL_SIGMA_SPACE=1; BILATERAL_JPEG_QUALITY=95
# Stage 10 (opt-in): bake an observation-only texture atlas from the source frames onto the
# final mesh (weighted-median project-and-blend). Off by default so the geometry/CAD path is
# unchanged and needs no extra deps. Enable with --texture (needs xatlas + trimesh in env sugar).
TEXTURE=0; ATLAS=2048; TEX_VIEWS=24
# Object isolation: default is the mesh-space visual-hull CARVE (masks isolate the object from the
# full-supervision SuGaR mesh), after which pymeshfix seals the unobserved bottom into a watertight
# solid. --gaussian-prune switches to the experimental Gaussian-space prune + mask-restricted SuGaR
# loss; it is kept for reference but under-supervises small objects at this VRAM-capped resolution
# (the object is a few % of a 480px frame), giving lower shape fidelity than the carve path.
GAUSSIAN_PRUNE=0; PRUNE_KEEP_RATIO=0.6; PRUNE_MIN_VIEWS=8; PRUNE_MIN_SUPPORT=0
# Stage 7's visual-hull test: a face is rejected when its silhouette agreement n_vis/n_geom falls
# below this. It is the guard against a lump FUSED to the object -- such a lump is the outermost
# surface so it reads as observed, and only its projection outside the silhouette betrays it. But it
# is also, measured, the single binding constraint on THIN structures. On object7 (Meccano forklift)
# at the 0.75 default it amputated both masts and every wheel: the wheels had rho 1.000 and 54
# visible views yet 0% survived. Sweeping it while holding everything else (raw mesh mast height
# 1.399):
#   0.75 -> mast 57.4% wheels  0.0% extentY 0.824      0.40 -> mast 67.3% wheels 39.5% extentY 1.260
#   0.60 -> mast 61.8% wheels  0.0% extentY 0.877      0.30 -> mast 68.7% wheels 79.9% extentY 1.269
#   0.50 -> mast 65.2% wheels 18.2% extentY 1.259      0.20 -> mast 70.1% wheels 98.4% extentY 1.272
# Neither --px-per-face (2->24 made it slightly WORSE) nor --rho0 (0.15->0.05 changed NOTHING, since
# those faces were already rejected here) moves it. Lower it for thin/spindly objects; keep 0.75 for
# compact ones, where the fused-lump guard is worth more than thin-feature recall.
RHO_KEEP_RATIO=0.75
# Focal-Anchored masked training (only active with --gaussian-prune): area-normalized L1 so object
# gradients keep full magnitude (weight 1.0), SSIM eroded to keep its window off the mask boundary,
# an L2 anchor tethering Gaussians to their pruned positions (prevents topology tearing), and an
# S^2 focal crop that fills the VRAM-capped frame with the object (~4x object resolution).
MASK_LOSS_WEIGHT=1.0; SSIM_ERODE=5; ANCHOR_LAMBDA=0.2; ANCHOR_UNTIL=15000; FOCAL_CROP=1
# Environment override is useful for controlled A/B tests against an older run whose effective
# crop size is already recorded in its log. Normal CLI runs retain the validated 384 px default.
FOCAL_SIZE="${SUGAR_FOCAL_SIZE_OVERRIDE:-384}"
# Visual-hull cage: black-composite the GT + clamp Gaussian scales at this quantile of the initial
# pruned-cube scales, so splats cannot stretch into depth spikes ("hook"). 0 disables the clamp.
SCALE_CLAMP=0.98
# Poisson budget for the isolated object (--gaussian-prune only). "auto" derives the octree depth
# from the measured sample spacing: a fixed depth right for a whole scene makes the leaf cell as
# small as the gap between samples here, so Poisson fits the sampling pattern and shatters the
# surface (depth 9 = 370 components/genus 29; depth 7 = 17 components/genus -1 on identical data).
# Upstream depth 10 also puts a 1024^3 octree around a ~1-unit object: 6.0M triangles, whose quadric
# decimation exhausted this box's 7.7 GB and killed the WSL VM outright. Auto lands near depth 7,
# which is both stable and small enough that decimation never has to touch the mesh. The point cap
# bounds the Poisson input. The carve path keeps upstream values (depth 10, no cap) -- it meshes a
# whole scene and is already validated at PSNR 18.35.
GP_POISSON_DEPTH=auto; GP_SURFACE_MAX_POINTS=1500000
# Poisson's density trim (upstream 0.1) strips hallucinated far-field surface, which is surgical on
# a full scene but not on an isolated object: there the low-density vertices are scattered specks
# over the whole object, so trimming punched 13,173 boundary edges into a mesh Poisson had handed
# over CLOSED. That lace is what stage 9 then had to bridge -- pymeshfix by cutting, Poisson by
# doming. 0 keeps the surface closed; the fg-bbox filter, largest-component keep and the stage-9
# --crop-ply still guard against far-field junk. The carve path keeps upstream 0.1.
GP_DENSITY_QUANTILE=0.0
# Stage-9 Poisson re-fit depth, kept for close_mesh.py --poisson-repair. No longer used by the
# --gaussian-prune path: with the stage-7 observation-confidence filter upstream, the closure input
# is clean and a smooth-energy re-fit would only dome across the unobserved face.
CLOSE_POISSON_DEPTH=7
# Stage 9b: TV-normal flattening of the completed patch (--gaussian-prune only). OFF by default.
# TV is the right regulariser in principle -- an l1 penalty on dihedral angle has piecewise-constant
# minimisers, so a flat face rather than a dome (Sec 7.2/Q4) -- but measured here it buys only ~1%
# of TV(n) while BREAKING exact watertightness: moving vertices without a self-intersection guard
# turned watertight=True into selfX=True. Sec 8 requires every local rule to preserve the invariant
# (proactive TransforMesh-style detection); this implementation has no such guard, so it is opt-in
# via --tv-flatten until it does. The dome was already cured upstream by the rho filter plus
# dropping the smooth-energy closure, so there is little left for it to win here.
TV_FLATTEN=0; TV_ITERS=40
# Stage 7b: visual-hull completion of the wound (--gaussian-prune only). The grid resolution is the
# real lever on quality: the hull's boundary voxels are sampled one per voxel, so an oblique face
# carries ripple at the voxel frequency, and it only disappears once that is finer than the Poisson
# cell size (depth 9 over this bbox is ~0.0035, grid 256 gives ~0.0041). Raising the grid costs
# carve time roughly as N^3.
HULL_GRID=256; HULL_DEPTH=9; HULL_TARGET_FACES=150000
# Per-vertex RGB baked onto the final mesh by sampling the Gaussians (default ON).
VERTEX_COLOR=1
# Collision asset (opt-in --collision): CoACD convex decomposition of the watertight mesh -> URDF.
COLLISION=0; COACD_THRESH=0.05
# H1 + H-final: SDF-regularization sample count. Upstream is 1M; the estimator plateaus by ~200k
# and even 50k is <0.5% per-step error, which 6000-step averaging crushes. We default to 50k made
# safe by error-guided sampling (below). Cost is ~linear in this count -> ~20x faster SDF phase.
SDF_SAMPLES=50000
# Error-guided sampling (H-final refinement): concentrate the reduced budget on high-SDF-residual
# Gaussians so thin/high-curvature regions are not starved, with a coverage floor (1 - mix) that
# keeps the whole surface sampled. On by default in the pipeline; disable with --no-error-guided.
ERROR_GUIDED=1
ERROR_MIX=0.5      # fraction of the budget that chases residual (rest holds the coverage floor)
ERROR_EMA=0.9      # per-Gaussian residual EMA decay
# H2: the GT-image VRAM budget is DERIVED, not fixed. SuGaR keeps all GT images resident, and
# its non-GT footprint (model + optimizer + rendering) measured ~2.5 GB peak on this scene, so
# the budget = (free VRAM - reserve), clamped. This adapts resolution to the machine's actual
# free memory and to frame count, instead of a constant that is right for exactly one operating point.
NONGT_RESERVE_GB=2.8
IMG_BUDGET_MIN_GB=0.6
IMG_BUDGET_MAX_GB=1.4

usage() { sed -n '2,20p' "$0"; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    --target) TARGET="$2"; shift 2;;
    --bilateral) BILATERAL=1; shift;;
    --bilateral-diameter) BILATERAL_DIAMETER="$2"; shift 2;;
    --bilateral-sigma-color) BILATERAL_SIGMA_COLOR="$2"; shift 2;;
    --bilateral-sigma-space) BILATERAL_SIGMA_SPACE="$2"; shift 2;;
    --bilateral-jpeg-quality) BILATERAL_JPEG_QUALITY="$2"; shift 2;;
    --refine) REFINE="$2"; shift 2;;
    --vertices) NVERT="$2"; shift 2;;
    --sdf-samples) SDF_SAMPLES="$2"; shift 2;;
    --no-error-guided) ERROR_GUIDED=0; shift;;
    --error-mix) ERROR_MIX="$2"; shift 2;;
    --gs-iters) GS_ITERS="$2"; shift 2;;
    --keep-ratio) KEEP_RATIO="$2"; shift 2;;
    --hull-grid) HULL_GRID="$2"; shift 2;;
    --hull-target-faces) HULL_TARGET_FACES="$2"; shift 2;;
    --texture) TEXTURE=1; shift;;
    --atlas) ATLAS="$2"; shift 2;;
    --tex-views) TEX_VIEWS="$2"; shift 2;;
    --gaussian-prune) GAUSSIAN_PRUNE=1; shift;;
    --prune-keep-ratio) PRUNE_KEEP_RATIO="$2"; shift 2;;
    --rho-keep-ratio) RHO_KEEP_RATIO="$2"; shift 2;;
    --prune-min-views) PRUNE_MIN_VIEWS="$2"; shift 2;;
    --prune-min-support) PRUNE_MIN_SUPPORT="$2"; shift 2;;
    # An int pins the octree depth; "auto" searches it by measured topology. The search judges the
    # COARSE mesh (components/genus), but what governs the final surface is the topology of the
    # RHO-FILTERED mesh, and those diverge: on object6 auto picked depth 7 (15 components, genus 5
    # -- clean by its own test), yet after rho the observed region came apart into 331 components
    # with 6,088 boundary edges, versus 3 and 440 at depth 6. The closure bridged every one of them
    # and the final mesh went genus 8 with visible seams, coverage 90.0% -> 82.7%.
    --gp-poisson-depth) GP_POISSON_DEPTH="$2"; shift 2;;
    --mask-loss-weight) MASK_LOSS_WEIGHT="$2"; shift 2;;
    --focal-crop) FOCAL_CROP=1; shift;;
    --no-focal-crop) FOCAL_CROP=0; shift;;
    --focal-size) FOCAL_SIZE="$2"; shift 2;;
    --anchor-lambda) ANCHOR_LAMBDA="$2"; shift 2;;
    --ssim-erode) SSIM_ERODE="$2"; shift 2;;
    --no-vertex-color) VERTEX_COLOR=0; shift;;
    --tv-flatten) TV_FLATTEN=1; shift;;
    --no-tv-flatten) TV_FLATTEN=0; shift;;
    --tv-iters) TV_ITERS="$2"; shift 2;;
    --collision) COLLISION=1; shift;;
    --coacd-threshold) COACD_THRESH="$2"; shift 2;;
    --force) FORCE=1; shift;;
    -h|--help) usage;;
    *) echo "unknown arg $1"; usage;;
  esac
done
[[ -n "$VIDEO" ]] || { echo "ERROR: --video required"; usage; }
[[ -f "$VIDEO" ]] || { echo "ERROR: no such video: $VIDEO"; exit 1; }
[[ -n "$NAME" ]] || NAME="$(basename "${VIDEO%.*}")"

SCENE="$SCENES_ROOT/$NAME"
LOGS="$SCENE/logs"
mkdir -p "$SCENE" "$LOGS"
cd "$SUGAR_DIR"

# --- timing: SECONDS is a bash builtin that counts seconds since it was last set ---
SECONDS=0
LAST_STAGE_T=0
fmt_hms() { printf '%02d:%02d:%02d' $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 )); }
# say() marks a stage boundary; it reports how long the PREVIOUS stage took and the running total.
say()  {
  local now=$SECONDS d=$(( SECONDS - LAST_STAGE_T ))
  [[ $LAST_STAGE_T -gt 0 && $d -gt 0 ]] && printf '    \033[2m(previous stage took %s)\033[0m\n' "$(fmt_hms $d)"
  printf '\n\033[1m=== %s ===\033[0m  \033[2m[total %s]\033[0m\n' "$*" "$(fmt_hms $now)"
  LAST_STAGE_T=$now
}
info() { printf '    %s\n' "$*"; }
die()  { printf '\033[31mFAILED after %s: %s\033[0m\n' "$(fmt_hms $SECONDS)" "$*" >&2; exit 1; }
done_marker() { echo "$SCENE/.done_$1"; }
is_done() { [[ $FORCE -eq 0 && -f "$(done_marker "$1")" ]]; }
mark_done() { touch "$(done_marker "$1")"; }

in_env() { local e="$1"; shift; ( set +u; source "$CONDA_SH"; conda activate "$e"; "$@" ); }

free_vram_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

require_vram() {
  local need=$1 have
  have=$(free_vram_mb)
  info "GPU free: ${have} MiB (stage wants >= ${need} MiB)"
  if (( have < need )); then
    echo "  waiting for GPU memory to free up..."
    for _ in $(seq 1 30); do sleep 10; have=$(free_vram_mb); (( have >= need )) && break; done
    (( have >= need )) || die "not enough free VRAM (${have} MiB < ${need} MiB). Close other GPU users."
  fi
}

# GT images live on the GPU for the whole SuGaR run: n * w * h * 3 * 4 bytes.
# H2: derive the byte budget from *measured free VRAM* minus SuGaR's non-GT footprint, then
# solve footprint <= budget for the image side w (using w=side, h=aspect*side):
#     side = sqrt(budget / (12 * n * aspect))
# Emits: "<side> <budget_gb> <footprint_gb> <over_budget_flag>". The flag is 1 only when the
# 480px quality floor is hit AND the resulting footprint still exceeds budget (the n>~700 edge
# my H2 validation found) — surfaced as a warning instead of a silent OOM risk.
derive_max_img_size() {
  local n=$1 w=$2 h=$3 free_mb=$4
  in_env sugar python - "$n" "$w" "$h" "$free_mb" "$NONGT_RESERVE_GB" "$IMG_BUDGET_MIN_GB" "$IMG_BUDGET_MAX_GB" <<'PY'
import sys, math
n, w, h = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
free_gb = float(sys.argv[4]) / 1024.0
reserve, bmin, bmax = float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
budget = max(bmin, min(bmax, free_gb - reserve))     # adaptive to actual free memory
aspect = h / w
side = math.sqrt((budget * 1e9) / (12.0 * n * aspect))
side = int(max(480, min(1600, round(side / 32) * 32)))   # clamp + round to 32
foot = n * side * (side * h / w) * 12 / 1e9
over = 1 if (side <= 480 and foot > budget * 1.02) else 0
print(f"{side} {budget:.2f} {foot:.2f} {over}")
PY
}

say "Scene: $NAME"
info "video : $VIDEO"
info "scene : $SCENE"

# ---------------------------------------------------------------- 1. frames --
if is_done frames; then say "1/12 frames (cached: $(ls "$SCENE/input" | wc -l))"; else
  say "1/12 Extracting + sharpness-filtering frames"
  rm -rf "$SCENE/_allframes" "$SCENE/input"; mkdir -p "$SCENE/_allframes"
  in_env colmap ffmpeg -loglevel error -i "$VIDEO" -qscale:v 1 -qmin 1 \
      -vf "fps=$FPS" "$SCENE/_allframes/%05d.jpg" || die "ffmpeg extraction"
  info "extracted $(ls "$SCENE/_allframes" | wc -l) frames at ${FPS} fps"
  SELECT_ENV=sugar
  SELECT_ARGS=(--src "$SCENE/_allframes" --dst "$SCENE/input" --target "$TARGET")
  if [[ "$BILATERAL" == "1" ]]; then
    SELECT_ENV=seg
    SELECT_ARGS+=(--bilateral --bilateral-diameter "$BILATERAL_DIAMETER"
      --bilateral-sigma-color "$BILATERAL_SIGMA_COLOR"
      --bilateral-sigma-space "$BILATERAL_SIGMA_SPACE"
      --jpeg-quality "$BILATERAL_JPEG_QUALITY")
  fi
  in_env "$SELECT_ENV" python scripts/select_sharp.py "${SELECT_ARGS[@]}" \
      2>&1 | tee "$LOGS/frames.log" | grep -E "Keeping|sharpness|bilateral" || true
  N=$(ls "$SCENE/input" | wc -l); (( N >= 40 )) || die "only $N frames kept"
  rm -rf "$SCENE/_allframes"          # ~1 GB of intermediates, not needed again
  mark_done frames
fi
NFRAMES=$(ls "$SCENE/input" | wc -l)

# ---------------------------------------------------------------- 2. COLMAP --
if is_done colmap; then say "2/12 COLMAP (cached)"; else
  say "2/12 COLMAP SfM on $NFRAMES frames"
  rm -rf "$SCENE/distorted" "$SCENE/sparse" "$SCENE/images" "$SCENE/stereo"
  mkdir -p "$SCENE/distorted/sparse"
  # Low peak_threshold: matte objects and dark mats yield far fewer SIFT features
  # than the 8192 default cap. Measured on object1: 2/70 registered at defaults,
  # 182/185 at 0.002.
  in_env colmap colmap feature_extractor \
      --database_path "$SCENE/distorted/database.db" --image_path "$SCENE/input" \
      --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
      --SiftExtraction.peak_threshold 0.002 --SiftExtraction.max_num_features 16384 \
      > "$LOGS/colmap_feat.log" 2>&1 || die "feature extraction"
  in_env colmap colmap exhaustive_matcher \
      --database_path "$SCENE/distorted/database.db" \
      > "$LOGS/colmap_match.log" 2>&1 || die "feature matching"
  in_env colmap colmap mapper \
      --database_path "$SCENE/distorted/database.db" --image_path "$SCENE/input" \
      --output_path "$SCENE/distorted/sparse" \
      --Mapper.ba_global_function_tolerance=0.000001 \
      > "$LOGS/colmap_map.log" 2>&1 || die "mapper"

  # COLMAP routinely emits several sub-models; keep the one with most images.
  BEST=$(in_env sugar python - "$SCENE" <<'PY'
import sys, os
sys.path.insert(0, "gaussian_splatting")
from scene.colmap_loader import read_extrinsics_binary
root = os.path.join(sys.argv[1], "distorted", "sparse")
best, n = None, -1
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d, "images.bin")
    if os.path.isfile(p):
        k = len(read_extrinsics_binary(p))
        print(f"  submodel {d}: {k} images", file=sys.stderr)
        if k > n: best, n = d, k
print(best)
PY
) || die "submodel inspection"
  info "largest submodel: $BEST"
  if [[ "$BEST" != "0" ]]; then
    rm -rf "$SCENE/distorted/sparse/0"
    mv "$SCENE/distorted/sparse/$BEST" "$SCENE/distorted/sparse/0"
  fi
  find "$SCENE/distorted/sparse" -mindepth 1 -maxdepth 1 -type d ! -name 0 -exec rm -rf {} +

  in_env colmap colmap image_undistorter --image_path "$SCENE/input" \
      --input_path "$SCENE/distorted/sparse/0" --output_path "$SCENE" \
      --output_type COLMAP > "$LOGS/colmap_undist.log" 2>&1 || die "undistort"
  mkdir -p "$SCENE/sparse/0"
  for f in "$SCENE"/sparse/*; do
    [[ "$(basename "$f")" != "0" ]] && mv "$f" "$SCENE/sparse/0/" || true
  done

  REG=$(in_env sugar python - "$SCENE" <<'PY'
import sys; sys.path.insert(0, "gaussian_splatting")
from scene.colmap_loader import read_extrinsics_binary
print(len(read_extrinsics_binary(sys.argv[1] + "/sparse/0/images.bin")))
PY
)
  info "registered $REG / $NFRAMES images"
  in_env sugar python - "$REG" "$NFRAMES" <<'PY' || die "too few images registered - capture problem, not a settings problem"
import sys
reg, n = int(sys.argv[1]), int(sys.argv[2])
sys.exit(0 if reg >= max(30, 0.5 * n) else 1)
PY
  mark_done colmap
fi

read -r IMG_W IMG_H < <(in_env sugar python - "$SCENE" <<'PY'
import os, sys
from PIL import Image
d = os.path.join(sys.argv[1], "images")
w, h = Image.open(os.path.join(d, sorted(os.listdir(d))[0])).size
print(w, h)
PY
)
NREG=$(ls "$SCENE/images" | wc -l)
info "undistorted: $NREG images at ${IMG_W}x${IMG_H}"

# ------------------------------------------------------------- 3. 3DGS ------
GS_OUT="output/vanilla_gs/$NAME"
if is_done gs; then say "3/12 vanilla 3DGS (cached)"; else
  say "3/12 Vanilla 3DGS, $GS_ITERS iters at -r 2"
  require_vram 2600
  rm -rf "$GS_OUT"; mkdir -p "$GS_OUT"
  in_env sugar python gaussian_splatting/train.py -s "$SCENE" -m "$GS_OUT" \
      -r 2 --iterations "$GS_ITERS" --test_iterations "$GS_ITERS" \
      --save_iterations "$GS_ITERS" > "$LOGS/gs.log" 2>&1 || die "3DGS training"
  grep -oE "\[ITER $GS_ITERS\] Evaluating train: L1 [0-9.]+ PSNR [0-9.]+" "$LOGS/gs.log" | tail -1 | sed 's/^/    /'
  mark_done gs
fi

# ------------------------------------------------------------- 4. masks -----
# U2Net silhouettes. Generated early now: they drive the Gaussian prune (stage 5) that isolates
# the object before SuGaR, and (with --texture) the bake at stage 10.
if is_done masks; then say "4/12 masks (cached: $(ls "$SCENE/masks" | wc -l))"; else
  say "4/12 U2Net object masks for $NREG views"
  mkdir -p "$SCENE/masks"
  in_env seg python - "$SCENE" <<'PY' > "$LOGS/masks.log" 2>&1 || die "mask generation"
import os, sys
import numpy as np
from PIL import Image
from rembg import remove, new_session
S = sys.argv[1]
sess = new_session("u2net")
names = sorted(os.listdir(f"{S}/images"))
for i, n in enumerate(names):
    dst = f"{S}/masks/{os.path.splitext(n)[0]}.png"
    if os.path.exists(dst):
        continue
    im = Image.open(f"{S}/images/{n}").convert("RGB")
    im.thumbnail((640, 640))
    m = remove(im, session=sess, only_mask=True)
    Image.fromarray((np.array(m) > 127).astype("uint8") * 255).save(dst)
    if i % 50 == 0:
        print(f"{i}/{len(names)}", flush=True)
print("done")
PY
  COV=$(in_env sugar python - "$SCENE" <<'PY'
import os, sys, numpy as np
from PIL import Image
S = sys.argv[1]; d = f"{S}/masks"
c = [np.array(Image.open(os.path.join(d, f))).mean() / 255 for f in sorted(os.listdir(d))[::10]]
print(f"{100*np.mean(c):.1f}")
PY
)
  info "mean mask coverage: ${COV}% of frame"
  mark_done masks
fi

# ------------------------------------------------------------- 5. prune -----
# Default isolation is the mesh-space carve (stage 7) on the full-supervision SuGaR mesh.
# --gaussian-prune instead isolates in Gaussian space: forward-project every trained Gaussian into
# the U2Net masks, drop the background ones, and train SuGaR on the pruned object with a
# mask-restricted loss (stage 6). Kept for reference; lower fidelity on small objects here.
PRUNED_GS="output/pruned_gs/$NAME"
PRUNED_PLY="$PRUNED_GS/point_cloud/iteration_${GS_ITERS}/point_cloud.ply"
GS_PLY="$GS_OUT/point_cloud/iteration_${GS_ITERS}/point_cloud.ply"
if [[ "$GAUSSIAN_PRUNE" == "1" ]]; then
  SUGAR_GS="$PRUNED_GS"; COLOR_GS_PLY="$PRUNED_PLY"
  if is_done prune; then say "5/12 gaussian prune (cached)"; else
    say "5/12 Gaussian-space prune -> isolate object (keep-ratio $PRUNE_KEEP_RATIO, min-views $PRUNE_MIN_VIEWS, support auto)"
    mkdir -p "$(dirname "$PRUNED_GS")"; rm -rf "$PRUNED_GS"; cp -r "$GS_OUT" "$PRUNED_GS"
    # --min-support 0 = auto. The agreement ratio alone is not a volumetric test: its denominator
    # counts only the views that framed the centre, so distant junk is judged by a handful of
    # cameras and survives. On object6 that left 477 centres a median 7.1 diagonals out, and the
    # coarse mesh spanned [22.7, 4.8, 16.3] in 30 components. See prune_gaussians.calibrate_support.
    in_env sugar python scripts/prune_gaussians.py --gs-dir "$GS_OUT" --scene "$SCENE" \
        --out-ply "$PRUNED_PLY" --iteration "$GS_ITERS" \
        --min-views "$PRUNE_MIN_VIEWS" --keep-ratio "$PRUNE_KEEP_RATIO" \
        --min-support "$PRUNE_MIN_SUPPORT" --dilate 2 \
        --report-json "output/final/${NAME}_prune.json" \
        2>&1 | tee "$LOGS/prune.log" \
      | grep -E "kept|gaussians|support threshold|fails |extent" | sed 's/^/    /' || die "gaussian prune"
    mark_done prune
  fi
else
  SUGAR_GS="$GS_OUT"; COLOR_GS_PLY="$GS_PLY"
  say "5/12 gaussian prune (skipped: carve-based isolation, default)"
fi

# ------------------------------------------------------------- 6. SuGaR -----
# SuGaR meshes SUGAR_GS (the full scene by default; the pruned object with --gaussian-prune). A
# per-mode cache tag means switching modes re-runs SuGaR rather than reusing the other mode's mesh.
SUGAR_TAG=$([[ "$GAUSSIAN_PRUNE" == "1" ]] && echo sugar_gp || echo sugar)
if is_done "$SUGAR_TAG"; then say "6/12 SuGaR (cached)"; else
  say "6/12 SuGaR: coarse regularization -> Poisson mesh -> refine ($REFINE)"
  require_vram 3400
  # H2: derive the GT resolution cap AFTER clearing the GPU, from the free VRAM we actually have.
  read -r MAXIMG IMG_BUDGET GTGB OVERBUDGET < <(derive_max_img_size "$NREG" "$IMG_W" "$IMG_H" "$(free_vram_mb)")
  info "SUGAR_MAX_IMG_SIZE=$MAXIMG (H2: adaptive budget ${IMG_BUDGET} GB) -> ~${GTGB} GB GT resident for $NREG images"
  if [[ "$OVERBUDGET" == "1" ]]; then
    info "WARNING: at the 480px floor, $NREG frames need ~${GTGB} GB of GT (> budget). Lower --target if you hit OOM."
  fi
  info "H1/H-final: SUGAR_SDF_SAMPLES=$SDF_SAMPLES ($(awk "BEGIN{printf \"%.0f\", 1000000/$SDF_SAMPLES}")x fewer than upstream 1M)"
  if [[ "$ERROR_GUIDED" == "1" ]]; then
    info "  error-guided sampling ON (mix=$ERROR_MIX, ema=$ERROR_EMA): reduced budget concentrates on high-residual regions"
  else
    info "  error-guided sampling OFF (uniform sampling)"
  fi
  # --gaussian-prune only: Focal-Anchored masked training. The pruned object Gaussians are trained
  # with a mask-restricted, area-normalized photometric loss (full-magnitude gradients), an eroded
  # SSIM window, an L2 spatial anchor to the pruned positions, and an S^2 focal crop that fills the
  # frame with the object. The default carve path leaves all of these off (SUGAR_MASK_LOSS=0).
  ANCHOR_EFF=$([[ "$GAUSSIAN_PRUNE" == "1" ]] && echo "$ANCHOR_LAMBDA" || echo 0.0)
  FOCAL_EFF=$([[ "$GAUSSIAN_PRUNE" == "1" ]] && echo "$FOCAL_CROP" || echo 0)
  [[ "$GAUSSIAN_PRUNE" == "1" ]] && info "Focal-Anchored masked training: L1 area-norm (w=$MASK_LOSS_WEIGHT), SSIM erode=$SSIM_ERODE, anchor=$ANCHOR_EFF, focal_crop=$FOCAL_EFF (S=$FOCAL_SIZE)"
  [[ "$GAUSSIAN_PRUNE" == "1" ]] && info "Visual-hull cage: black-GT + unmasked L1, scale clamp q$SCALE_CLAMP; Poisson depth $GP_POISSON_DEPTH, surface cap $GP_SURFACE_MAX_POINTS pts, density trim $GP_DENSITY_QUANTILE (0 = keep Poisson closed)"
  # SuGaR is the long stage: coarse ~8-9h on this GPU (fast phase ~1h, then the SDF-regularization
  # phase from iter 9000 dominates), plus ~25 min for Poisson + refine + texture. H1 cuts the SDF phase.
  ( set +u; source "$CONDA_SH"; conda activate sugar; \
    export SUGAR_MAX_IMG_SIZE="$MAXIMG"; \
    export SUGAR_SDF_SAMPLES="$SDF_SAMPLES"; \
    export SUGAR_ERROR_GUIDED="$ERROR_GUIDED"; \
    export SUGAR_ERROR_MIX="$ERROR_MIX"; \
    export SUGAR_ERROR_EMA="$ERROR_EMA"; \
    export SUGAR_MASK_LOSS="$GAUSSIAN_PRUNE"; \
    export SUGAR_MASK_LOSS_WEIGHT="$MASK_LOSS_WEIGHT"; \
    export SUGAR_SSIM_ERODE="$SSIM_ERODE"; \
    export SUGAR_ANCHOR_LAMBDA="$ANCHOR_EFF"; \
    export SUGAR_ANCHOR_UNTIL="$ANCHOR_UNTIL"; \
    export SUGAR_FOCAL_CROP="$FOCAL_EFF"; \
    export SUGAR_FOCAL_SIZE="$FOCAL_SIZE"; \
    export SUGAR_SCALE_CLAMP="$([[ "$GAUSSIAN_PRUNE" == "1" ]] && echo "$SCALE_CLAMP" || echo 0.0)"; \
    export SUGAR_POISSON_DEPTH="$([[ "$GAUSSIAN_PRUNE" == "1" ]] && echo "$GP_POISSON_DEPTH" || echo 10)"; \
    export SUGAR_SURFACE_MAX_POINTS="$([[ "$GAUSSIAN_PRUNE" == "1" ]] && echo "$GP_SURFACE_MAX_POINTS" || echo 0)"; \
    export SUGAR_DENSITY_QUANTILE="$([[ "$GAUSSIAN_PRUNE" == "1" ]] && echo "$GP_DENSITY_QUANTILE" || echo 0.1)"; \
    python train_full_pipeline.py -s "$SCENE" -r density \
        --gs_output_dir "$SUGAR_GS/" --refinement_time "$REFINE" \
        -v "$NVERT" --eval False --export_obj True \
  ) > "$LOGS/sugar.log" 2>&1 || die "SuGaR pipeline (see $LOGS/sugar.log)"
  mark_done "$SUGAR_TAG"
fi

REFINED_OBJ=$(ls -t "output/refined_mesh/$NAME/"*.obj 2>/dev/null | head -1) \
  || die "no refined mesh produced"
[[ -n "$REFINED_OBJ" ]] || die "no refined mesh in output/refined_mesh/$NAME"
info "refined mesh: $REFINED_OBJ"

# --------------------------------------------- 7. isolate the object --------
# Default: visual-hull carve keeps only the mesh that projects inside the U2Net silhouettes across
# the views that see it (full-supervision SuGaR -> a clean object). --gaussian-prune: the SuGaR
# mesh is already the object, so skip carve and crop any residual drift by the object bbox.
mkdir -p output/final "output/final/${NAME}/visual"
FINAL="output/final/${NAME}_final.ply"
CLEAN_MID="output/final/${NAME}_clean.ply"
CLOSE_ARGS=(--report-json "output/final/${NAME}_watertight.json")
if [[ "$GAUSSIAN_PRUNE" == "1" ]]; then
  # The Gaussian prune isolates in Gaussian space but never checks the MESH against the cameras, so
  # geometry in regions no camera saw survives to the end: detached blobs and spikes off the unseen
  # face. The surface reconstruction cannot tell that junk from a face seen 300 times, fits it, and
  # the closure wraps it into a bulge. Filter on observation confidence instead (Sec 3/4 of
  # morphogenetic_completion.md): a face no camera ever saw is not evidence about the object.
  # Measured on object4: floater components mean rho 0.033 / 73% never-seen vs the body's 0.594,
  # and the wound is one coherent cap (58% of it within 60deg of a single axis, vs 10% of A_obs).
  RHO_FILT="output/final/${NAME}_rhofilt.ply"
  if is_done rhofilter; then say "7/12 observation-confidence filter (cached)"; else
    say "7/12 Observation confidence -> drop never-seen geometry (cameras decide)"
    in_env sugar python scripts/observation_confidence.py --mesh "$REFINED_OBJ" --scene "$SCENE" \
        --out "$RHO_FILT" --drop-unobserved --keep-largest \
        --keep-ratio "$RHO_KEEP_RATIO" \
        --rho-ply "output/final/${NAME}/visual/${NAME}_rho.ply" \
        --report-json "output/final/${NAME}_rho.json" 2>&1 | tee "$LOGS/rho.log" \
        | grep -E "never seen|A_obs|angular|kept|largest component" | sed 's/^/    /' \
        || die "observation-confidence filter"
    mark_done rhofilter
  fi
  # ------------------------------------------ 7b. hull completion ------------
  # The rho filter leaves the mesh honest but OPEN, and what fills that wound decides the result.
  # Every filler that reasons from a prior rather than from the data failed here, each in the way
  # its own energy predicts: Poisson minimises a smooth energy so it returns a DOME; pymeshfix
  # triangulates a big non-planar rim into a folded tent; and a planar cap fits one plane to a rim
  # that (measured) snakes over 52% of the object diagonal, so the plane passes through the solid
  # and the cap becomes a plate slicing a corner off the cube -- 327 triangles carrying 23.7% of
  # the surface area.
  #
  # The visual hull is admissible in a way none of those are: it is a deterministic function of the
  # observations, so completing with it adds no information the cameras did not supply (Sec 4,
  # I(C;D|O)=0). For a box seen from an orbit it yields planar walls and a flat bottom by
  # measurement -- the piecewise-planar answer Sec Q4 argues for, without iterating a normal flow
  # toward it. Measured: hull extent [0.808 0.917 0.973] vs the measured mesh's
  # [0.824 0.920 0.972], and observed geometry moves by a median 0.00003 (Sec 7.4).
  HULL_MESH="output/final/${NAME}_hullcomp.ply"
  if is_done hullcomplete; then say "7b/12 hull completion (cached)"; else
    say "7b/12 Visual-hull completion of the wound (silhouettes decide the missing face)"
    in_env sugar python scripts/hull_complete.py --mesh "$RHO_FILT" --scene "$SCENE" \
        --out "$HULL_MESH" --grid "$HULL_GRID" --depth "$HULL_DEPTH" \
        --target-faces "$HULL_TARGET_FACES" \
        --hull-ply "output/final/${NAME}/visual/${NAME}_hull.ply" \
        --report-json "output/final/${NAME}_hull.json" 2>&1 | tee "$LOGS/hull.log" \
        | grep -E "hull carved|hull surface|inside the wound|poisson depth|snapped|decimated|completed mesh|extent|invented|deviation" \
        | sed 's/^/    /' || die "hull completion"
    mark_done hullcomplete
  fi
  # Stage 8's feature-preserving cleanup is skipped on this path. It was built for the raw SuGaR
  # shell; on the hull-completed mesh its topology cleanup opened 305 boundary edges and drove euler
  # to -132, after which pymeshfix (which repairs by DELETING) cut the object down to volume 0.1442
  # and extent [0.809 0.680 0.913]. Going straight to the closure gave watertight, euler 2, volume
  # 0.2261, extent [0.833 0.920 0.968] against a hull volume of 0.2230.
  CLEAN_SRC="$HULL_MESH"; SKIP_CLEAN=1; CLOSE_ARGS+=(--crop-ply "$PRUNED_PLY")
  # The wound here is one large face of a piecewise-planar object. pymeshfix triangulates a rim that
  # big into a folded, pinched surface (the "tent"), and a local normal flow cannot descend out of a
  # fold. Sec Q4 says the TV-of-the-normal minimiser over that face is simply a FLAT face, so
  # construct it directly: fit a plane to the rim and cap it, leaving every observed vertex fixed.
  # NOTE: --planar-cap is deliberately NOT passed any more. With stage 7b the wound is already
  # filled by the hull, so there is no large rim left to cap; and on this object the rim it used to
  # be handed was not planar at all, which is what produced the corner-slicing plate. The guard
  # inside planar_cap() now refuses such a rim outright, but the right fix is not to create the
  # situation in the first place.
  # NOTE: deliberately NOT --poisson-repair here. That existed to give pymeshfix a low-genus
  # surface when this path handed it genus-613 lace. With the rho filter upstream the input is a
  # clean single-component shell with one hole, and a Poisson re-fit is then actively harmful: it
  # is a smooth-energy closure, so its minimiser over the unobserved face is a DOME (Sec Q4).
  # Plain pymeshfix triangulates the hole instead, and yields a genuinely watertight result.
else
  CARVED="output/final/${NAME}_carved.ply"
  if is_done carve; then say "7/12 carve (cached)"; else
    say "7/12 Visual-hull carve -> isolate the object (masks decide)"
    in_env sugar python scripts/carve_mesh.py --mesh "$REFINED_OBJ" --scene "$SCENE" \
        --out "$CARVED" --keep-ratio "$KEEP_RATIO" --min-views 8 \
        --tri-rule majority --dilate 2 2>&1 | tee "$LOGS/carve.log" \
        | grep -E "kept|wrote|cameras" | sed 's/^/    /' || die "carve"
    mark_done carve
  fi
  CLEAN_SRC="$CARVED"
fi

# ------------------------------------------------------------- 8. clean -----
if [[ "${SKIP_CLEAN:-0}" == "1" ]]; then
  say "8/12 Feature-preserving cleanup (skipped: stage 7b already produced a clean single shell)"
  cp "$CLEAN_SRC" "$CLEAN_MID"
else
  say "8/12 Feature-preserving cleanup"
  in_env sugar python scripts/clean_mesh_v2.py --in "$CLEAN_SRC" --out "$CLEAN_MID" \
      --min-component-tris 120 --attach-radius 0.04 \
      --smooth-iters 3 --preserve-angle 35 2>&1 | tee "$LOGS/clean.log" \
      | grep -vE "Open3D WARNING" | sed 's/^/    /' || die "cleanup"
fi

# --------------------------------------------- 9. watertight safeguard ------
# pymeshfix seals the unobserved bottom (and residual holes) into a guaranteed watertight manifold
# -- the minimal-surface closure a physics/collision asset needs.
say "9/12 Watertight safeguard (pymeshfix: seal the unobserved bottom)"
in_env sugar python scripts/close_mesh.py --in "$CLEAN_MID" --out "$FINAL" \
    "${CLOSE_ARGS[@]}" 2>&1 | tee "$LOGS/close.log" \
    | grep -iE "watertight|boundary|cropped|warn" | sed 's/^/    /' || die "watertight safeguard"

# ------------------------------------- 9b. TV-normal flattening of the patch --
# Any closure that minimises a quadratic returns a DOME over the unobserved face -- that is the
# correct minimiser of the energy it poses, not a tuning artefact. The total variation of the normal
# is an l1 penalty on dihedral angle, and l1 minimisers are sparse, so the normal comes out
# piecewise CONSTANT: flat faces meeting at sharp creases (Sec 7.2 / Q4). Only the invented patch
# moves; measured vertices are frozen, so this cannot alter observed geometry (Sec 7.4). Provenance
# comes from the pre-closure surface, never from rho recomputed after closure -- the patch is the
# outermost surface there and would certify itself as observed.
if [[ "$GAUSSIAN_PRUNE" == "1" && "$TV_FLATTEN" == "1" ]]; then
  say "9b/12 TV-normal flattening of the completed patch (observed geometry frozen)"
  in_env sugar python scripts/tv_normal_patch.py --in "$FINAL" --out "$FINAL" \
      --observed-mesh "$RHO_FILT" --iters "$TV_ITERS" \
      --report-json "output/final/${NAME}_tv.json" 2>&1 | tee "$LOGS/tv.log" \
      | grep -E "provenance|movable|TV\(n\)" | sed 's/^/    /' || die "TV flattening"
fi

# ------------------------------------------------------- 10. vertex colour --
if [[ "$VERTEX_COLOR" == "1" ]]; then
  say "10/12 Vertex colours from Gaussians"
  in_env sugar python scripts/color_from_gaussians.py --mesh "$FINAL" --gs-ply "$COLOR_GS_PLY" \
      --glb "output/final/${NAME}/visual/${NAME}_colored.glb" 2>&1 | tee "$LOGS/color.log" \
      | grep -iE "colored" | sed 's/^/    /' || die "vertex colour"
fi

# ------------------------------------------------------------ 10. texture ---
# Opt-in (--texture): bake an observation-only texture atlas from the GT frames onto $FINAL.
# Colour comes only from what a camera saw; a weighted-median blend rejects the specular
# highlight a mean would smear. Per-mode cache tag so it re-bakes on the new watertight mesh.
TEX_COV=""
if [[ "$TEXTURE" == "1" ]]; then
  if in_env sugar python -c "import xatlas, trimesh" >/dev/null 2>&1; then
    TEX_TAG=$([[ "$GAUSSIAN_PRUNE" == "1" ]] && echo texture_gp || echo texture)
    if is_done "$TEX_TAG"; then say "11/12 texture (cached)"; else
      say "11/12 Texture bake -> weighted-median atlas (${ATLAS}^2, K=$TEX_VIEWS)"
      in_env sugar python scripts/bake_texture.py --mesh "$FINAL" --scene "$SCENE" \
          --out-dir "output/final/${NAME}/visual" --name "$NAME" --atlas "$ATLAS" \
          --views-per-texel "$TEX_VIEWS" --validate 8 2>&1 | tee "$LOGS/texture.log" \
          | grep -E "^(atlas:|COVERAGE|VALIDATION|wrote|      )" | sed 's/^/    /' || die "texture bake"
      mark_done "$TEX_TAG"
    fi
    TEX_COV=$(grep -oE "COVERAGE [0-9.]+%" "$LOGS/texture.log" 2>/dev/null | tail -1)
  else
    info "note: --texture requested but xatlas/trimesh missing in env 'sugar'; skipping the bake"
    info "      install once with:  conda run -n sugar pip install xatlas trimesh"
  fi
fi

# ---------------------------------------------------------- 11. collision ---
# Opt-in (--collision): CoACD convex-decompose the watertight mesh into a physics collision
# asset + URDF (Path B). Invents no geometry; strictly bounds the shell into convex hulls.
COLL_URDF=""
if [[ "$COLLISION" == "1" ]]; then
  if in_env sugar python -c "import coacd, trimesh" >/dev/null 2>&1; then
    if is_done collision; then say "11/12 collision (cached)"; else
      say "11/12 CoACD convex decomposition -> collision asset + URDF (threshold $COACD_THRESH)"
      mkdir -p "output/final/${NAME}/collision"
      in_env sugar python scripts/make_collision.py --mesh "$FINAL" \
          --out-dir "output/final/${NAME}/collision" --name "$NAME" \
          --threshold "$COACD_THRESH" 2>&1 | tee "$LOGS/collision.log" \
          | grep -iE "parts|mass|com|convex|warn" | sed 's/^/    /' || die "collision"
      mark_done collision
    fi
    COLL_URDF="output/final/${NAME}/collision/${NAME}.urdf"
  else
    info "note: --collision requested but coacd/trimesh missing in env 'sugar'; skipping."
    info "      install once with:  conda run -n sugar pip install coacd trimesh"
  fi
fi

say "DONE"
in_env sugar python - "$FINAL" <<'PY'
import sys, numpy as np, open3d as o3d
m = o3d.io.read_triangle_mesh(sys.argv[1])
bb = m.get_axis_aligned_bounding_box()
V, F = np.asarray(m.vertices), np.asarray(m.triangles)
a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
vol = abs(np.einsum("ij,ij->i", np.cross(a, b), c).sum()) / 6.0
# "closed" is what volume/inertia actually need; o3d's is_watertight() additionally demands no
# self-intersection, so report both rather than conflating them.
closed = m.is_edge_manifold(allow_boundary_edges=False) and m.is_vertex_manifold()
print(f"    {sys.argv[1]}")
print(f"    {len(F):,} triangles, {len(V):,} vertices")
print(f"    extent {np.round(bb.get_extent(), 3)}   enclosed volume {vol:.4f}")
print(f"    closed manifold: {closed}   o3d watertight: {m.is_watertight()}   "
      f"vertex_colors: {m.has_vertex_colors()}")
PY
info "logs in $LOGS"
printf '\n\033[1;32m>>> TOTAL PIPELINE WALL-CLOCK: %s (%d frames, refine=%s, sdf_samples=%s)\033[0m\n' \
  "$(fmt_hms $SECONDS)" "$NREG" "$REFINE" "$SDF_SAMPLES"
printf '\033[1;32m>>> MESH: output/final/%s_final.ply%s\033[0m\n' \
  "$NAME" "$([[ "$VERTEX_COLOR" == "1" ]] && echo ' (closed manifold, vertex-coloured)' || echo ' (closed manifold)')"
if [[ -n "${TEX_COV:-}" ]]; then
  printf '\033[1;32m>>> TEXTURE: %s  ->  output/final/%s/visual/%s_textured.glb\033[0m\n' \
    "$TEX_COV" "$NAME" "$NAME"
fi
if [[ -n "${COLL_URDF:-}" ]]; then
  printf '\033[1;32m>>> COLLISION: %s\033[0m\n' "$COLL_URDF"
fi
