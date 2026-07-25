# SuGaR build/run environment — source this before building extensions or running the pipeline.
#   source /home/acer/workspace/dirac/SuGaR/env.sh

# --- CUDA 11.8 toolkit (toolkit only; the Windows driver provides the GPU under WSL2) ---
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- CUDA 11.8 supports gcc <= 11, Ubuntu 24.04 defaults to gcc-13 ---
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-11'

# --- GPU architecture. GTX 1650 is Turing = compute capability 7.5.
# +PTX embeds forward-compatible PTX so the binary still JITs on a newer card.
# NOTE: the runbook's "8.9" is for RTX 40xx and produces binaries that CANNOT
# run here; the failure surfaces as a bogus "out of memory" from simple-knn.
export TORCH_CUDA_ARCH_LIST="7.5+PTX"

# --- Build parallelism. This box has 7.7 GB RAM / 2 GB swap; nvcc peaks around
# 2-3 GB per translation unit, so the default (nproc=8) OOMs the machine. ---
export MAX_JOBS=2
