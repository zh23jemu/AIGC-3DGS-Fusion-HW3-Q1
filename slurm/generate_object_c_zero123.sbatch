#!/bin/bash
#SBATCH --job-name=hw3_obj_c_i23d
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# 本脚本用于物体 C 的单图到 3D 生成。不同 Zero123 实现入口差异较大，
# 默认命令是占位式标准入口，请按实际仓库 README 调整脚本名和参数。
module load slurm || true

PROJECT_DIR="${PROJECT_DIR:-/mnt/users/xj62kv/AIGC-3DGS-Fusion}"
ZERO123_ROOT="${ZERO123_ROOT:-${PROJECT_DIR}/third_party/zero123}"
INPUT_IMAGE="${INPUT_IMAGE:-${PROJECT_DIR}/data/raw/object_c_single_view/object_c.png}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/object_c_zero123}"

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"

cd "${ZERO123_ROOT}"
"${PROJECT_DIR}/.venv/bin/python" run_zero123.py \
  --input "${INPUT_IMAGE}" \
  --output_dir "${OUTPUT_DIR}"
