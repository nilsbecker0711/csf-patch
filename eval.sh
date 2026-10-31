#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#
# Evaluate a saved patch. CHECKPOINT takes a best.pt; PATCH_PNG takes a
# rendered PNG instead, pasted square at the frame centre.

set -euo pipefail

module --ignore_cache load "cuda/11.8"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-csf-patch}"
export PYTHONNOUSERSITE=1

CS="${CITYSCAPES_ROOT:?set CITYSCAPES_ROOT to the Cityscapes root}"

ARCH="${ARCH:-segformer_b0}"
IMAGES="${IMAGES:-all}"

BASE="--arch $ARCH --cityscapes_root $CS --img_h 1024 --img_w 2048"

if [[ -n "${PATCH_PNG:-}" ]]; then
    SOURCE="--patch_png $PATCH_PNG --patch_scale ${PATCH_SCALE:-0.25}"
else
    SOURCE="--checkpoint ${CHECKPOINT:?set CHECKPOINT to a best.pt, or PATCH_PNG to a rendered patch}"
fi

python scripts/evaluate.py $BASE $SOURCE \
    --images "$IMAGES" \
    --inference auto --slide_checkpoint \
    --out_root results/eval --tag "${TAG:-eval}"
