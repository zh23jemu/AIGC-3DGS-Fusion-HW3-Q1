#!/bin/bash
#SBATCH --job-name=hw3_bg_3dgs
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# 本脚本用于背景场景重建。建议选择 Mip-NeRF 360 garden、bicycle、
# counter 等开源场景中的一个，或使用自己采集的背景多视角图片。
module load slurm || true

PROJECT_DIR="${PROJECT_DIR:-/mnt/users/xj62kv/underwater-crack-correction/AIGC-3DGS-Fusion}"
GS_ROOT="${GS_ROOT:-${PROJECT_DIR}/third_party/gaussian-splatting}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/data/raw/background_scene/images}"
COLMAP_DIR="${COLMAP_DIR:-${PROJECT_DIR}/outputs/background_colmap}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/outputs/background_3dgs}"

mkdir -p "${PROJECT_DIR}/logs" "${COLMAP_DIR}" "${MODEL_DIR}"

cd "${PROJECT_DIR}"
colmap automatic_reconstructor \
  --workspace_path "${COLMAP_DIR}" \
  --image_path "${IMAGE_DIR}" \
  --dense 0

cd "${GS_ROOT}"
"${PROJECT_DIR}/.venv/bin/python" convert.py -s "${COLMAP_DIR}"
"${PROJECT_DIR}/.venv/bin/python" train.py -s "${COLMAP_DIR}" -m "${MODEL_DIR}"
"${PROJECT_DIR}/.venv/bin/python" render.py -m "${MODEL_DIR}"

