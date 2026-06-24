#!/bin/bash
#SBATCH --job-name=hw3_obj_a_3dgs
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# 本脚本用于真实物体 A 的多视角重建：先由 COLMAP 估计位姿，再训练 3DGS。
# 使用前请把 PROJECT_DIR 和 GS_ROOT 改成集群上的实际路径。
module load slurm || true

PROJECT_DIR="${PROJECT_DIR:-/mnt/users/xj62kv/underwater-crack-correction/AIGC-3DGS-Fusion}"
GS_ROOT="${GS_ROOT:-${PROJECT_DIR}/third_party/gaussian-splatting}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/data/raw/object_a_multiview/images}"
COLMAP_DIR="${COLMAP_DIR:-${PROJECT_DIR}/outputs/object_a_colmap}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/outputs/object_a_3dgs}"

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

