#!/usr/bin/env bash
# End-to-end: phone video -> isolated, cleaned object mesh.
#
#   ./scripts/reconstruct_object.sh --video inputs/object2.mp4 --name object2
#   ./scripts/reconstruct_object.sh --video inputs/object2.mp4 --name object2 --texture
#
# Runs frame extraction -> COLMAP SfM -> vanilla 3DGS -> SuGaR -> U2Net masks ->
# visual-hull carve -> feature-preserving cleanup, producing the same class of
# output as output/final/object1_final.ply. With --texture, an 8th stage bakes an
# observation-only texture atlas from the source frames (weighted-median blend)
# -> output/final/<name>_textured.glb (+ _texture.png, _textured_obj/, _texbake.json).
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
# Everything the pipeline reads/writes lives under SuGaR/ so the repo is self-contained.
# Videos in SuGaR/inputs/, scenes in SuGaR/scenes/, artifacts in SuGaR/output/. Only the
# conda location is machine-specific: override it with  CONDA_SH=/path/to/conda.sh  if needed.
CONDA_SH="${CONDA_SH:-/home/acer/miniforge3/etc/profile.d/conda.sh}"
SCENES_ROOT="$SUGAR_DIR/scenes"

VIDEO=""; NAME=""; FPS=10; TARGET=240; REFINE=short; NVERT=500000
GS_ITERS=7000; KEEP_RATIO=0.85; FORCE=0
# Stage 8 (opt-in): bake an observation-only texture atlas from the source frames onto the
# final mesh (weighted-median project-and-blend). Off by default so the geometry/CAD path is
# unchanged and needs no extra deps. Enable with --texture (needs xatlas + trimesh in env sugar).
TEXTURE=0; ATLAS=2048; TEX_VIEWS=24
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
    --refine) REFINE="$2"; shift 2;;
    --vertices) NVERT="$2"; shift 2;;
    --sdf-samples) SDF_SAMPLES="$2"; shift 2;;
    --no-error-guided) ERROR_GUIDED=0; shift;;
    --error-mix) ERROR_MIX="$2"; shift 2;;
    --gs-iters) GS_ITERS="$2"; shift 2;;
    --keep-ratio) KEEP_RATIO="$2"; shift 2;;
    --texture) TEXTURE=1; shift;;
    --atlas) ATLAS="$2"; shift 2;;
    --tex-views) TEX_VIEWS="$2"; shift 2;;
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
if is_done frames; then say "1/8 frames (cached: $(ls "$SCENE/input" | wc -l))"; else
  say "1/8 Extracting + sharpness-filtering frames"
  rm -rf "$SCENE/_allframes" "$SCENE/input"; mkdir -p "$SCENE/_allframes"
  in_env colmap ffmpeg -loglevel error -i "$VIDEO" -qscale:v 1 -qmin 1 \
      -vf "fps=$FPS" "$SCENE/_allframes/%05d.jpg" || die "ffmpeg extraction"
  info "extracted $(ls "$SCENE/_allframes" | wc -l) frames at ${FPS} fps"
  in_env sugar python scripts/select_sharp.py --src "$SCENE/_allframes" \
      --dst "$SCENE/input" --target "$TARGET" 2>&1 | tee "$LOGS/frames.log" | grep -E "Keeping|sharpness" || true
  N=$(ls "$SCENE/input" | wc -l); (( N >= 40 )) || die "only $N frames kept"
  rm -rf "$SCENE/_allframes"          # ~1 GB of intermediates, not needed again
  mark_done frames
fi
NFRAMES=$(ls "$SCENE/input" | wc -l)

# ---------------------------------------------------------------- 2. COLMAP --
if is_done colmap; then say "2/8 COLMAP (cached)"; else
  say "2/8 COLMAP SfM on $NFRAMES frames"
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
if is_done gs; then say "3/8 vanilla 3DGS (cached)"; else
  say "3/8 Vanilla 3DGS, $GS_ITERS iters at -r 2"
  require_vram 2600
  rm -rf "$GS_OUT"; mkdir -p "$GS_OUT"
  in_env sugar python gaussian_splatting/train.py -s "$SCENE" -m "$GS_OUT" \
      -r 2 --iterations "$GS_ITERS" --test_iterations "$GS_ITERS" \
      --save_iterations "$GS_ITERS" > "$LOGS/gs.log" 2>&1 || die "3DGS training"
  grep -oE "\[ITER $GS_ITERS\] Evaluating train: L1 [0-9.]+ PSNR [0-9.]+" "$LOGS/gs.log" | tail -1 | sed 's/^/    /'
  mark_done gs
fi

# ------------------------------------------------------------- 4. SuGaR -----
if is_done sugar; then say "4/8 SuGaR (cached)"; else
  say "4/8 SuGaR: coarse regularization -> Poisson mesh -> refine ($REFINE)"
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
  # SuGaR is the long stage: coarse ~8-9h on this GPU (fast phase ~1h, then the SDF-regularization
  # phase from iter 9000 dominates), plus ~25 min for Poisson + refine + texture. H1 cuts the SDF phase.
  ( set +u; source "$CONDA_SH"; conda activate sugar; \
    export SUGAR_MAX_IMG_SIZE="$MAXIMG"; \
    export SUGAR_SDF_SAMPLES="$SDF_SAMPLES"; \
    export SUGAR_ERROR_GUIDED="$ERROR_GUIDED"; \
    export SUGAR_ERROR_MIX="$ERROR_MIX"; \
    export SUGAR_ERROR_EMA="$ERROR_EMA"; \
    python train_full_pipeline.py -s "$SCENE" -r density \
        --gs_output_dir "$GS_OUT/" --refinement_time "$REFINE" \
        -v "$NVERT" --eval False --export_obj True \
  ) > "$LOGS/sugar.log" 2>&1 || die "SuGaR pipeline (see $LOGS/sugar.log)"
  mark_done sugar
fi

REFINED_OBJ=$(ls -t "output/refined_mesh/$NAME/"*.obj 2>/dev/null | head -1) \
  || die "no refined mesh produced"
[[ -n "$REFINED_OBJ" ]] || die "no refined mesh in output/refined_mesh/$NAME"
info "refined mesh: $REFINED_OBJ"

# ------------------------------------------------------------- 5. masks -----
if is_done masks; then say "5/8 masks (cached: $(ls "$SCENE/masks" | wc -l))"; else
  say "5/8 U2Net object masks for $NREG views"
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

# ------------------------------------------------------------- 6. carve -----
mkdir -p output/final
CARVED="output/final/${NAME}_carved.ply"
if is_done carve; then say "6/8 carve (cached)"; else
  say "6/8 Visual-hull carve (no bounding box; masks decide)"
  in_env sugar python scripts/carve_mesh.py --mesh "$REFINED_OBJ" --scene "$SCENE" \
      --out "$CARVED" --keep-ratio "$KEEP_RATIO" --min-views 8 \
      --tri-rule majority --dilate 2 2>&1 | tee "$LOGS/carve.log" \
      | grep -E "kept|wrote|cameras" | sed 's/^/    /' || die "carve"
  mark_done carve
fi

# ------------------------------------------------------------- 7. clean -----
FINAL="output/final/${NAME}_final.ply"
say "7/8 Feature-preserving cleanup"
in_env sugar python scripts/clean_mesh_v2.py --in "$CARVED" --out "$FINAL" \
    --min-component-tris 120 --attach-radius 0.04 \
    --smooth-iters 3 --preserve-angle 35 2>&1 | tee "$LOGS/clean.log" \
    | grep -vE "Open3D WARNING" | sed 's/^/    /' || die "cleanup"

# ------------------------------------------------------------- 8. texture ---
# Opt-in (--texture): bake an observation-only texture atlas from the GT frames
# onto $FINAL. Colour comes only from what a camera saw; a weighted-median blend
# rejects the specular highlight a mean would smear. Resumable (.done_texture),
# geometry path untouched, and skips cleanly if xatlas/trimesh are absent.
TEX_COV=""
if [[ "$TEXTURE" == "1" ]]; then
  if in_env sugar python -c "import xatlas, trimesh" >/dev/null 2>&1; then
    if is_done texture; then say "8/8 texture (cached)"; else
      say "8/8 Texture bake -> weighted-median atlas (${ATLAS}^2, K=$TEX_VIEWS)"
      in_env sugar python scripts/bake_texture.py --mesh "$FINAL" --scene "$SCENE" \
          --out-dir output/final --atlas "$ATLAS" --views-per-texel "$TEX_VIEWS" \
          --validate 8 2>&1 | tee "$LOGS/texture.log" \
          | grep -E "^(atlas:|COVERAGE|VALIDATION|wrote|      )" | sed 's/^/    /' || die "texture bake"
      mark_done texture
    fi
    TEX_COV=$(grep -oE "COVERAGE [0-9.]+%" "$LOGS/texture.log" 2>/dev/null | tail -1)
  else
    info "note: --texture requested but xatlas/trimesh missing in env 'sugar'; skipping the bake"
    info "      install once with:  conda run -n sugar pip install xatlas trimesh"
  fi
fi

say "DONE"
in_env sugar python - "$FINAL" <<'PY'
import sys, open3d as o3d, numpy as np
m = o3d.io.read_triangle_mesh(sys.argv[1])
bb = m.get_axis_aligned_bounding_box()
print(f"    {sys.argv[1]}")
print(f"    {len(m.triangles):,} triangles, {len(m.vertices):,} vertices")
print(f"    extent {np.round(bb.get_extent(), 3)}")
PY
info "logs in $LOGS"
printf '\n\033[1;32m>>> TOTAL PIPELINE WALL-CLOCK: %s (%d frames, refine=%s, sdf_samples=%s)\033[0m\n' \
  "$(fmt_hms $SECONDS)" "$NREG" "$REFINE" "$SDF_SAMPLES"
[[ -n "${TEX_COV:-}" ]] && printf '\033[1;32m>>> TEXTURE: %s  ->  output/final/%s_textured.glb (+ _texture.png, _textured_obj/)\033[0m\n' \
  "$TEX_COV" "$NAME"
