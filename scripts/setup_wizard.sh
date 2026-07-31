#!/usr/bin/env bash
# Dynamic, resumable installer for the complete SuGaRrush reconstruction stack.
#
# Safe discovery:
#   bash scripts/setup_wizard.sh --plan
#
# Interactive installation:
#   bash scripts/setup_wizard.sh
#
# CI/unattended installation:
#   bash scripts/setup_wizard.sh --yes

set -Eeuo pipefail

SETUP_VERSION=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPERS="$ROOT/scripts/setup_wizard"
SETUP_ROOT="$ROOT/.setup"
STATE_DIR="$SETUP_ROOT/state"
LOG_DIR="$SETUP_ROOT/logs"
REPORT_DIR="$SETUP_ROOT/reports"
RUNTIME_ENV="$SETUP_ROOT/runtime.env"
MODEL_DIR="$SETUP_ROOT/models"
SOURCE_DIR="$SETUP_ROOT/sources"
LOCAL_CONDA_ROOT="$SETUP_ROOT/miniforge"

MODE=install
ASSUME_YES=0
CHECK_ONLY=0
SKIP_MODEL=0
BOOTSTRAP_CONDA=1
CONDA_OVERRIDE=""
CUDA_OVERRIDE=""
CC_OVERRIDE=""
CXX_OVERRIDE=""
ARCH_OVERRIDE=""
JOBS_OVERRIDE=""
FORCED_STAGES=","
TEMP_DIR=""
ACTIVE_STAGE=""
ACTIVE_LOG=""

usage() {
  sed -n '2,12p' "$0"
  cat <<'EOF'

Options:
  --plan, --dry-run        Detect the machine and print the plan; change nothing.
  --yes                    Run non-interactively after preflight succeeds.
  --check-only             Re-run dry checks using an existing .setup/runtime.env.
  --skip-model             Do not download or execute the U²-Net model check.
  --no-bootstrap-conda     Fail instead of installing private Miniforge.
  --conda PATH             Use this conda executable.
  --cuda-home PATH         Override generated CUDA_HOME after env creation.
  --cc PATH                Override the generated C compiler.
  --cxx PATH               Override the generated C++ compiler.
  --arch-list LIST         Override TORCH_CUDA_ARCH_LIST (for example 7.5+PTX).
  --jobs N                 Override memory-derived MAX_JOBS.
  --force-stage NAME       Re-run one stage even when its marker is current.
                           Names: conda, source, sugar, runtime, native,
                                  colmap, seg, model, verify, or all.
  -h, --help               Show this help.

The wizard never installs an NVIDIA driver or writes system package paths.
All generated environments, model weights, build sources, logs and reports
live under .setup/, which is ignored by git.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan|--dry-run)
      MODE=plan
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --skip-model)
      SKIP_MODEL=1
      shift
      ;;
    --no-bootstrap-conda)
      BOOTSTRAP_CONDA=0
      shift
      ;;
    --conda)
      CONDA_OVERRIDE="$2"
      shift 2
      ;;
    --cuda-home)
      CUDA_OVERRIDE="$2"
      shift 2
      ;;
    --cc)
      CC_OVERRIDE="$2"
      shift 2
      ;;
    --cxx)
      CXX_OVERRIDE="$2"
      shift 2
      ;;
    --arch-list)
      ARCH_OVERRIDE="$2"
      shift 2
      ;;
    --jobs)
      JOBS_OVERRIDE="$2"
      shift 2
      ;;
    --force-stage)
      FORCED_STAGES+="$2,"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" == plan && "$CHECK_ONLY" == 1 ]]; then
  echo "ERROR: --plan and --check-only are mutually exclusive." >&2
  exit 2
fi
if [[ -n "$JOBS_OVERRIDE" && ! "$JOBS_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --jobs must be a positive integer." >&2
  exit 2
fi

valid_stage() {
  case "$1" in
    conda|source|sugar|runtime|native|colmap|seg|model|verify|all) return 0 ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a forced_tokens <<< "$FORCED_STAGES"
for token in "${forced_tokens[@]}"; do
  [[ -z "$token" ]] && continue
  if ! valid_stage "$token"; then
    echo "ERROR: invalid --force-stage value: $token" >&2
    exit 2
  fi
done

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'
  DIM=$'\033[2m'
  GREEN=$'\033[32m'
  YELLOW=$'\033[33m'
  RED=$'\033[31m'
  RESET=$'\033[0m'
else
  BOLD=""
  DIM=""
  GREEN=""
  YELLOW=""
  RED=""
  RESET=""
fi

heading() {
  printf '\n%s%s%s\n' "$BOLD" "$1" "$RESET"
}

info() {
  printf '  %s\n' "$*"
}

warn() {
  printf '  %sWARN%s  %s\n' "$YELLOW" "$RESET" "$*" >&2
}

die() {
  printf '\n%sERROR%s  %s\n' "$RED" "$RESET" "$*" >&2
  if [[ -n "$ACTIVE_STAGE" ]]; then
    printf 'Stage: %s\n' "$ACTIVE_STAGE" >&2
  fi
  if [[ -n "$ACTIVE_LOG" && -f "$ACTIVE_LOG" ]]; then
    printf 'Log: %s\n' "$ACTIVE_LOG" >&2
  fi
  exit 1
}

cleanup() {
  if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
    rm -rf -- "$TEMP_DIR"
  fi
}
trap cleanup EXIT
trap 'die "unexpected failure at line $LINENO"' ERR

command -v python3 >/dev/null 2>&1 || die \
  "python3 is required for the bootstrap probe. Install Python 3 and rerun."
command -v git >/dev/null 2>&1 || die \
  "git is required (the repository should have been cloned with it)."

if [[ "$CHECK_ONLY" == 1 ]]; then
  [[ -f "$RUNTIME_ENV" ]] || die \
    "no generated runtime profile exists; run the wizard first."
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV"
  [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]] || die \
    "the generated CONDA_EXE is unavailable: ${CONDA_EXE:-unset}"
  mkdir -p "$REPORT_DIR"
  CHECK_ARGS=(
    --repo "$ROOT"
    --conda-exe "$CONDA_EXE"
    --report "$REPORT_DIR/dry-check.json"
  )
  [[ "$SKIP_MODEL" == 1 ]] && CHECK_ARGS+=(--skip-model)
  exec python3 "$HELPERS/dry_check.py" "${CHECK_ARGS[@]}"
fi

if [[ "$MODE" == plan ]]; then
  TEMP_DIR="$(mktemp -d -t sugarrush-plan.XXXXXX)"
  PROFILE_JSON="$TEMP_DIR/system.json"
  PROFILE_ENV="$TEMP_DIR/system.env"
else
  mkdir -p "$SETUP_ROOT" "$STATE_DIR" "$LOG_DIR" "$REPORT_DIR" \
    "$MODEL_DIR" "$SOURCE_DIR" "$SETUP_ROOT/cache/numba"
  PROFILE_JSON="$SETUP_ROOT/system.json"
  PROFILE_ENV="$SETUP_ROOT/system.env"
fi

python3 "$HELPERS/probe_system.py" \
  --repo "$ROOT" \
  --json-out "$PROFILE_JSON" \
  --shell-out "$PROFILE_ENV"
# shellcheck disable=SC1090
source "$PROFILE_ENV"

if [[ -n "$CONDA_OVERRIDE" ]]; then
  [[ -x "$CONDA_OVERRIDE" ]] || die \
    "--conda is not an executable file: $CONDA_OVERRIDE"
  PROBE_CONDA_EXE="$(realpath "$CONDA_OVERRIDE")"
fi
if [[ -z "$PROBE_CONDA_EXE" && -x "$LOCAL_CONDA_ROOT/bin/conda" ]]; then
  PROBE_CONDA_EXE="$LOCAL_CONDA_ROOT/bin/conda"
fi

heading "SuGaRrush setup plan"
WSL_LABEL=""
[[ "$PROBE_WSL" == 1 ]] && WSL_LABEL=" (WSL2)"
info "Repository      $ROOT"
info "Platform        ${PROBE_DISTRO:-$PROBE_OS} ${PROBE_ARCH}${WSL_LABEL}"
info "CPU / memory    $PROBE_CPU_COUNT logical CPUs, $PROBE_RAM_GIB GiB RAM, $PROBE_SWAP_GIB GiB swap"
info "Free disk       $PROBE_DISK_FREE_GIB GiB"
info "NVIDIA GPU      ${PROBE_GPU_NAMES:-not detected}"
info "Driver          ${PROBE_DRIVER_VERSIONS:-not detected}"
info "GPU targets     ${ARCH_OVERRIDE:-${PROBE_ARCH_LIST:-detected after PyTorch install}}"
info "Build workers   ${JOBS_OVERRIDE:-$PROBE_MAX_JOBS}"
if [[ -n "$PROBE_CONDA_EXE" ]]; then
  info "Conda           $PROBE_CONDA_EXE"
else
  info "Conda           private Miniforge bootstrap under .setup/miniforge"
fi
info "CUDA/GCC        private CUDA 11.8 + GCC 11 toolchain in the sugar environment"

heading "Installation stages"
cat <<'EOF'
  1. Bootstrap or select conda
  2. Validate repository sources
  3. Create/update the pinned sugar environment and private build toolchain
  4. Generate the machine-specific runtime profile
  5. Compile simple-knn, the Gaussian rasterizer and nvdiffrast
  6. Create/update the isolated COLMAP/ffmpeg environment
  7. Create/update the isolated U²-Net/OpenCV environment
  8. Cache U²-Net weights
  9. Run CLI, Python, geometry and real CUDA-kernel dry checks
EOF

if [[ "$PROBE_WARNING_COUNT" -gt 0 ]]; then
  heading "Warnings"
  while IFS= read -r line; do
    [[ -n "$line" ]] && warn "$line"
  done <<< "$PROBE_WARNINGS"
fi
if [[ "$PROBE_BLOCKER_COUNT" -gt 0 ]]; then
  heading "Blocking prerequisites"
  while IFS= read -r line; do
    [[ -n "$line" ]] && printf '  %s- %s%s\n' "$RED" "$line" "$RESET"
  done <<< "$PROBE_BLOCKERS"
fi

if [[ "$MODE" == plan ]]; then
  if [[ "$PROBE_BLOCKER_COUNT" -gt 0 ]]; then
    info "Planning completed; installation would stop at preflight."
  else
    info "Planning completed; no blocking prerequisite was detected."
  fi
  exit 0
fi

[[ "$PROBE_BLOCKER_COUNT" -eq 0 ]] || die \
  "preflight found blocking prerequisites; fix them and rerun --plan."

if [[ "$ASSUME_YES" == 0 ]]; then
  printf '\nContinue with this user-local installation? [y/N] '
  read -r answer
  [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]] || {
    info "Setup cancelled; no environment stage was started."
    exit 0
  }
fi

if [[ ! -x "$PROBE_CONDA_EXE" && "$BOOTSTRAP_CONDA" == 0 ]]; then
  die "conda was not found and --no-bootstrap-conda was supplied."
fi

fingerprint() {
  {
    printf '%s\n' "$SETUP_VERSION"
    sha256sum \
      "$ROOT/scripts/setup_wizard.sh" \
      "$HELPERS/probe_system.py" \
      "$HELPERS/dry_check.py" \
      "$HELPERS/environment.sugar.yml" \
      "$HELPERS/environment.colmap.yml" \
      "$HELPERS/environment.seg.yml"
  } | sha256sum | awk '{print $1}'
}
SETUP_FINGERPRINT="$(fingerprint)"

stage_forced() {
  [[ "$FORCED_STAGES" == *",all,"* || "$FORCED_STAGES" == *",$1,"* ]]
}

stage_artifact_exists() {
  case "$1" in
    conda)
      [[ -x "${CONDA_EXE:-${PROBE_CONDA_EXE:-}}" ]]
      ;;
    source)
      [[ -f "$ROOT/gaussian_splatting/submodules/simple-knn/setup.py" &&
         -f "$ROOT/gaussian_splatting/submodules/diff-gaussian-rasterization/setup.py" ]]
      ;;
    sugar)
      "${CONDA_EXE:-false}" run -n sugar python -c "import torch" \
        >/dev/null 2>&1
      ;;
    runtime)
      [[ -f "$RUNTIME_ENV" ]]
      ;;
    native)
      "${CONDA_EXE:-false}" run -n sugar python -c \
        "from simple_knn import _C; from diff_gaussian_rasterization import GaussianRasterizer; import nvdiffrast.torch" \
        >/dev/null 2>&1
      ;;
    colmap)
      "${CONDA_EXE:-false}" run -n colmap colmap -h >/dev/null 2>&1
      ;;
    seg)
      NUMBA_CACHE_DIR="$SETUP_ROOT/cache/numba" \
        "${CONDA_EXE:-false}" run -n seg python -c \
        "import rembg, cv2, onnxruntime" >/dev/null 2>&1
      ;;
    model)
      [[ "$SKIP_MODEL" == 1 || -s "$MODEL_DIR/u2net.onnx" ]]
      ;;
    verify)
      [[ -f "$REPORT_DIR/dry-check.json" ]] &&
        python3 - "$REPORT_DIR/dry-check.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
raise SystemExit(0 if report["summary"]["status"] == "passed" else 1)
PY
      ;;
    *)
      return 1
      ;;
  esac
}

stage_current() {
  local stage="$1"
  local marker="$STATE_DIR/$stage.done"
  [[ -f "$marker" ]] || return 1
  [[ "$(sed -n '1p' "$marker")" == "$SETUP_FINGERPRINT" ]] || return 1
  stage_artifact_exists "$stage"
}

run_stage() {
  local stage="$1"
  local title="$2"
  local function_name="$3"
  local marker="$STATE_DIR/$stage.done"
  local log="$LOG_DIR/$stage.log"

  if ! stage_forced "$stage" && stage_current "$stage"; then
    printf '\n%sSKIP%s  %-8s %s\n' "$DIM" "$RESET" "$stage" "$title"
    return
  fi

  printf '\n%sRUN%s   %-8s %s\n' "$BOLD" "$RESET" "$stage" "$title"
  ACTIVE_STAGE="$stage"
  ACTIVE_LOG="$log"
  local status
  if "$function_name" 2>&1 | tee "$log"; then
    status=0
  else
    status="${PIPESTATUS[0]}"
  fi
  if [[ "$status" -ne 0 ]]; then
    die "$title failed with exit code $status."
  fi
  printf '%s\n' "$SETUP_FINGERPRINT" > "$marker"
  ACTIVE_STAGE=""
  ACTIVE_LOG=""
  printf '%sPASS%s  %-8s %s\n' "$GREEN" "$RESET" "$stage" "$title"
}

download() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="$output" "$url"
  else
    die "curl or wget is required to download Miniforge and U²-Net dependencies."
  fi
}

stage_conda() {
  if [[ -n "$PROBE_CONDA_EXE" && -x "$PROBE_CONDA_EXE" ]]; then
    CONDA_EXE="$PROBE_CONDA_EXE"
    "$CONDA_EXE" --version
    return
  fi

  local asset="Miniforge3-Linux-x86_64.sh"
  local base="https://github.com/conda-forge/miniforge/releases/latest/download"
  local temporary
  temporary="$(mktemp -d -t sugarrush-miniforge.XXXXXX)"
  download "$base/$asset" "$temporary/$asset"
  download "$base/$asset.sha256" "$temporary/$asset.sha256"
  (
    cd "$temporary"
    sha256sum --check "$asset.sha256"
  )
  bash "$temporary/$asset" -b -p "$LOCAL_CONDA_ROOT"
  rm -rf -- "$temporary"
  CONDA_EXE="$LOCAL_CONDA_ROOT/bin/conda"
  "$CONDA_EXE" --version
}

resolve_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    :
  elif [[ -n "$PROBE_CONDA_EXE" && -x "$PROBE_CONDA_EXE" ]]; then
    CONDA_EXE="$PROBE_CONDA_EXE"
  elif [[ -x "$LOCAL_CONDA_ROOT/bin/conda" ]]; then
    CONDA_EXE="$LOCAL_CONDA_ROOT/bin/conda"
  else
    die "conda stage completed without an executable."
  fi
  CONDA_BASE="$("$CONDA_EXE" info --base)"
  CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
  [[ -f "$CONDA_SH" ]] || die "conda initialization script is missing: $CONDA_SH"
}

stage_source() {
  if [[ -f "$ROOT/.gitmodules" ]]; then
    git -C "$ROOT" submodule update --init --recursive
  fi
  local required=(
    "$ROOT/gaussian_splatting/train.py"
    "$ROOT/gaussian_splatting/submodules/diff-gaussian-rasterization/setup.py"
    "$ROOT/gaussian_splatting/submodules/simple-knn/setup.py"
    "$ROOT/scripts/reconstruct_object.sh"
  )
  local item
  for item in "${required[@]}"; do
    [[ -f "$item" ]] || {
      echo "missing required source: $item" >&2
      return 1
    }
  done
  git -C "$ROOT" rev-parse --short HEAD
}

conda_env_exists() {
  local environment="$1"
  "$CONDA_EXE" env list --json | python3 -c \
    'import json, os, sys
name = sys.argv[1]
data = json.load(sys.stdin)
raise SystemExit(0 if any(os.path.basename(path) == name for path in data["envs"]) else 1)' \
    "$environment"
}

create_or_update_env() {
  local environment="$1"
  local specification="$2"
  if conda_env_exists "$environment"; then
    "$CONDA_EXE" env update --name "$environment" --file "$specification"
  else
    "$CONDA_EXE" env create --file "$specification"
  fi
}

stage_sugar() {
  create_or_update_env sugar "$HELPERS/environment.sugar.yml"
  "$CONDA_EXE" run -n sugar python -c \
    "import torch; print('torch', torch.__version__, 'cuda runtime', torch.version.cuda)"
}

find_first_executable() {
  local candidate
  for candidate in "$@"; do
    if [[ -x "$candidate" ]]; then
      realpath "$candidate"
      return 0
    fi
  done
  return 1
}

detect_arch_with_torch() {
  "$CONDA_EXE" run -n sugar python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA GPU")
caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
values = [f"{major}.{minor}" for major, minor in caps]
values[-1] += "+PTX"
print(";".join(values))
'
}

stage_runtime() {
  local prefix cuda_home cc cxx arch_list jobs library_path
  prefix="$("$CONDA_EXE" run -n sugar python -c 'import sys; print(sys.prefix)')"
  cuda_home="${CUDA_OVERRIDE:-$prefix}"
  [[ -x "$cuda_home/bin/nvcc" ]] || {
    echo "CUDA compiler is missing: $cuda_home/bin/nvcc" >&2
    return 1
  }

  cc="${CC_OVERRIDE:-}"
  cxx="${CXX_OVERRIDE:-}"
  if [[ -z "$cc" ]]; then
    cc="$(find_first_executable \
      "$prefix/bin/x86_64-conda-linux-gnu-gcc" \
      "$prefix/bin/gcc" \
      "${PROBE_SYSTEM_CC:-}")"
  fi
  if [[ -z "$cxx" ]]; then
    cxx="$(find_first_executable \
      "$prefix/bin/x86_64-conda-linux-gnu-g++" \
      "$prefix/bin/x86_64-conda-linux-gnu-c++" \
      "$prefix/bin/g++" \
      "${PROBE_SYSTEM_CXX:-}")"
  fi
  [[ -x "$cc" && -x "$cxx" ]] || {
    echo "a usable C/C++ compiler pair was not found" >&2
    return 1
  }

  arch_list="${ARCH_OVERRIDE:-$PROBE_ARCH_LIST}"
  if [[ -z "$arch_list" ]]; then
    arch_list="$(detect_arch_with_torch)"
  fi
  [[ "$arch_list" =~ ^[0-9]+\.[0-9]+(\+PTX)?(;[0-9]+\.[0-9]+(\+PTX)?)*$ ]] || {
    echo "invalid computed TORCH_CUDA_ARCH_LIST: $arch_list" >&2
    return 1
  }
  jobs="${JOBS_OVERRIDE:-$PROBE_MAX_JOBS}"

  library_path="$prefix/lib:$prefix/lib64"
  if [[ -d "$prefix/targets/x86_64-linux/lib" ]]; then
    library_path="$library_path:$prefix/targets/x86_64-linux/lib"
  fi

  {
    echo "# Generated by scripts/setup_wizard.sh; do not commit."
    printf 'export SUGARRUSH_REPO=%q\n' "$ROOT"
    printf 'export SUGARRUSH_SETUP_ROOT=%q\n' "$SETUP_ROOT"
    printf 'export CONDA_EXE=%q\n' "$CONDA_EXE"
    printf 'export CONDA_SH=%q\n' "$CONDA_SH"
    printf 'export CUDA_HOME=%q\n' "$cuda_home"
    printf 'export CC=%q\n' "$cc"
    printf 'export CXX=%q\n' "$cxx"
    printf 'export NVCC_PREPEND_FLAGS=%q\n' "-ccbin $cxx"
    printf 'export TORCH_CUDA_ARCH_LIST=%q\n' "$arch_list"
    printf 'export MAX_JOBS=%q\n' "$jobs"
    printf 'export U2NET_HOME=%q\n' "$MODEL_DIR"
    printf 'export NUMBA_CACHE_DIR=%q\n' "$SETUP_ROOT/cache/numba"
    printf 'export PATH=%q${PATH:+:${PATH}}\n' "$cuda_home/bin:$prefix/bin"
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n' "$library_path"
  } > "$RUNTIME_ENV"

  # shellcheck disable=SC1090
  source "$RUNTIME_ENV"
  printf 'CUDA_HOME=%s\nCC=%s\nCXX=%s\nTORCH_CUDA_ARCH_LIST=%s\nMAX_JOBS=%s\n' \
    "$CUDA_HOME" "$CC" "$CXX" "$TORCH_CUDA_ARCH_LIST" "$MAX_JOBS"
}

run_sugar_pip() {
  "$CONDA_EXE" run -n sugar env \
    "CUDA_HOME=$CUDA_HOME" \
    "CC=$CC" \
    "CXX=$CXX" \
    "NVCC_PREPEND_FLAGS=$NVCC_PREPEND_FLAGS" \
    "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" \
    "MAX_JOBS=$MAX_JOBS" \
    python -m pip install --no-build-isolation "$@"
}

stage_native() {
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV"
  run_sugar_pip -e \
    "$ROOT/gaussian_splatting/submodules/diff-gaussian-rasterization"
  run_sugar_pip -e \
    "$ROOT/gaussian_splatting/submodules/simple-knn"

  local nvdiffrast_source="$ROOT/nvdiffrast"
  if [[ ! -f "$nvdiffrast_source/setup.py" ]]; then
    nvdiffrast_source="$SOURCE_DIR/nvdiffrast"
    if [[ ! -f "$nvdiffrast_source/setup.py" ]]; then
      git clone --depth 1 --branch v0.4.0 \
        https://github.com/NVlabs/nvdiffrast.git "$nvdiffrast_source"
    fi
  fi
  run_sugar_pip -e "$nvdiffrast_source"
}

stage_colmap() {
  create_or_update_env colmap "$HELPERS/environment.colmap.yml"
  "$CONDA_EXE" run -n colmap colmap -h | sed -n '1,5p'
  "$CONDA_EXE" run -n colmap ffmpeg -version | sed -n '1,2p'
}

stage_seg() {
  create_or_update_env seg "$HELPERS/environment.seg.yml"
  NUMBA_CACHE_DIR="$SETUP_ROOT/cache/numba" \
    "$CONDA_EXE" run -n seg python -c \
    "import cv2, onnxruntime, rembg; print('segmentation imports ready')"
}

stage_model() {
  if [[ "$SKIP_MODEL" == 1 ]]; then
    info "U²-Net preload skipped by request."
    return
  fi
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV"
  mkdir -p "$U2NET_HOME"
  "$CONDA_EXE" run -n seg python -c \
    "from rembg import new_session; new_session('u2net'); print('U2Net weights ready')"
  [[ -s "$U2NET_HOME/u2net.onnx" ]]
}

stage_verify() {
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV"
  local arguments=(
    --repo "$ROOT"
    --conda-exe "$CONDA_EXE"
    --report "$REPORT_DIR/dry-check.json"
  )
  [[ "$SKIP_MODEL" == 1 ]] && arguments+=(--skip-model)
  python3 "$HELPERS/dry_check.py" "${arguments[@]}"
}

run_stage conda "Bootstrap or select conda" stage_conda
resolve_conda
run_stage source "Validate repository sources" stage_source
run_stage sugar "Create/update the sugar environment" stage_sugar
run_stage runtime "Generate the machine runtime profile" stage_runtime
# Runtime variables are required by the remaining artifact checks and stages.
# shellcheck disable=SC1090
source "$RUNTIME_ENV"
run_stage native "Build project CUDA extensions" stage_native
run_stage colmap "Create/update COLMAP and ffmpeg" stage_colmap
run_stage seg "Create/update U²-Net and OpenCV" stage_seg
run_stage model "Cache U²-Net model weights" stage_model
run_stage verify "Execute required dry checks" stage_verify

heading "Setup complete"
info "Runtime profile  $RUNTIME_ENV"
info "System profile   $PROFILE_JSON"
info "Validation       $REPORT_DIR/dry-check.json"
info "Stage logs       $LOG_DIR/"
info "Next command:"
printf '\n  source %q\n' "$RUNTIME_ENV"
printf '  bash scripts/reconstruct_object.sh --video inputs/myobject.mp4 --name myobject\n\n'
