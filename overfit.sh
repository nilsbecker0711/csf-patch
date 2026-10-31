#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#
# Single-image overfit: one patch optimised against ONE image.

set -euo pipefail

module --ignore_cache load "cuda/11.8"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-csf-patch}"
export PYTHONNOUSERSITE=1

CS="${CITYSCAPES_ROOT:?set CITYSCAPES_ROOT to the Cityscapes root}"

ARCH="${ARCH:-segformer_b0}"
IMAGE="${IMAGE:-420}"
LOSS="${LOSS:-cospgd}"
MODE="${MODE:-csf}"
TAU="${TAU:-0.25}"
STEPS="${STEPS:-1000}"
LR="${LR:-0.2}"
SEEDS="${SEEDS:-1}"

BASE="--arch $ARCH --cityscapes_root $CS --img_h 1024 --img_w 2048"
GEOM="--patch_mode $MODE --patch_size 256 --patch_scale 0.25 --placement center"
CSF="--csf_threshold $TAU --csf_enforce realised"

python scripts/overfit.py $BASE $GEOM $CSF \
    --from_image \
    --loss_fn "$LOSS" \
    --image "$IMAGE" \
    --steps "$STEPS" --lr "$LR" --lr_schedule cosine \
    --seeds "$SEEDS" --log_every 25 \
    --inference auto --slide_checkpoint \
    --out_root results/overfit --tag "${TAG:-overfit}"
