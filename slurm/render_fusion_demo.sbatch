#!/bin/bash
#SBATCH --job-name=hw3_fusion_demo
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# 本脚本用于 CPU 端轻量融合渲染，可快速验证配置、空间变换和输出结果。
# 如果只是短时验证，也可以提交时覆盖为：sbatch --qos=shortjobs --time=01:00:00 ...
module load slurm || true

PROJECT_DIR="${PROJECT_DIR:-/mnt/users/xj62kv/underwater-crack-correction/AIGC-3DGS-Fusion}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/hw3_q1.yaml}"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

"${PROJECT_DIR}/.venv/bin/python" -m pip install -r requirements.txt
"${PROJECT_DIR}/.venv/bin/python" -m pip install -e .
"${PROJECT_DIR}/.venv/bin/python" -m aigc_3dgs_fusion demo --config "${CONFIG_PATH}"

