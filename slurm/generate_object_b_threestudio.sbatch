#!/bin/bash
#SBATCH --job-name=hw3_obj_b_t23d
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# 本脚本用于物体 B 的文本到 3D 生成。Prompt 可以通过环境变量 PROMPT 覆盖。
# threestudio 的配置名会随版本变化，如当前配置不存在，请替换为本机可用配置。
module load slurm || true

PROJECT_DIR="${PROJECT_DIR:-/mnt/users/xj62kv/AIGC-3DGS-Fusion}"
THREESTUDIO_ROOT="${THREESTUDIO_ROOT:-${PROJECT_DIR}/third_party/threestudio}"
PROMPT="${PROMPT:-a small ceramic robot toy, clean geometry, colorful texture, studio lighting}"
TRIAL_NAME="${TRIAL_NAME:-hw3_object_b_text_to_3d}"

mkdir -p "${PROJECT_DIR}/logs"

cd "${THREESTUDIO_ROOT}"
"${PROJECT_DIR}/.venv/bin/python" launch.py --config configs/prolificdreamer.yaml --train \
  system.prompt_processor.prompt="${PROMPT}" \
  trial_name="${TRIAL_NAME}"
