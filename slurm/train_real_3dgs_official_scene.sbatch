#!/bin/bash
#SBATCH --job-name=hw3_real_3dgs
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# 本脚本用于在 Slurm 集群上启动真实数据的 3DGS 短时验证训练。
# 数据源默认使用 3D Gaussian Splatting 官方提供的 T&T+DB COLMAP 数据包：
#   data/raw/3dgs_official/tandt/train
# 该数据已经包含 images/ 与 sparse/，因此不依赖 COLMAP 命令本身，适合集群上
# 先验证 CUDA 扩展、PyTorch、数据格式和训练入口是否都能跑通。
module load slurm || true
module load python/3.11.7

PROJECT_DIR="${PROJECT_DIR:-/mnt/users/xj62kv/AIGC-3DGS-Fusion}"
GS_ROOT="${GS_ROOT:-${PROJECT_DIR}/third_party/gaussian-splatting}"
SCENE_DIR="${SCENE_DIR:-${PROJECT_DIR}/data/raw/3dgs_official/tandt/train}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/outputs/real_3dgs/tandt_train_short}"
ITERATIONS="${ITERATIONS:-1500}"

# 集群驱动兼容 CUDA 12.8。PyTorch 采用 CUDA 12.x wheel，CUDA 编译器显式使用
# /usr/local/cuda-12.8，避免 shell 环境未加载 nvcc 时编译 3DGS 扩展失败。
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

PYTHON="${PROJECT_DIR}/.venv/bin/python"

mkdir -p "${PROJECT_DIR}/logs" "${MODEL_DIR}"
cd "${PROJECT_DIR}"

# 首次运行时创建项目本地虚拟环境；后续作业会复用同一个 .venv，减少排队后
# 的环境安装时间。所有 Python 包都安装在项目 .venv 内，避免污染系统环境。
if [ ! -x "${PYTHON}" ]; then
  python3 -m venv "${PROJECT_DIR}/.venv"
fi

"${PYTHON}" -m pip install --upgrade pip setuptools wheel ninja
"${PYTHON}" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
"${PYTHON}" -m pip install plyfile tqdm
"${PYTHON}" -m pip install -r "${PROJECT_DIR}/requirements.txt"

cd "${GS_ROOT}"
"${PYTHON}" -m pip install "${GS_ROOT}/submodules/diff-gaussian-rasterization"
"${PYTHON}" -m pip install "${GS_ROOT}/submodules/simple-knn"

# 新版 3DGS 仓库包含 fused-ssim 子模块，用于加速/增强训练损失；如果当前
# checkout 存在该目录，就一并安装。旧版仓库没有该子模块时不会影响训练。
if [ -d "${GS_ROOT}/submodules/fused-ssim" ]; then
  "${PYTHON}" -m pip install "${GS_ROOT}/submodules/fused-ssim"
fi

# 这是短时验证训练，不追求最终质量；正式实验可把 ITERATIONS 提高到 30000，
# 并将 QOS 改回 normal、time 改为 12 小时或更长。
"${PYTHON}" train.py \
  -s "${SCENE_DIR}" \
  -m "${MODEL_DIR}" \
  --iterations "${ITERATIONS}" \
  --eval \
  --data_device cpu \
  --test_iterations 500 1000 "${ITERATIONS}" \
  --save_iterations "${ITERATIONS}"
