#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#
# Universal patch training: ONE patch shared across the whole train split.
# universal_csf refuses to run unless --patch_size == int(img_h * patch_scale).

set -euo pipefail

module --ignore_cache load "cuda/11.8"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-csf-patch}"
export PYTHONNOUSERSITE=1

CS="${CITYSCAPES_ROOT:?set CITYSCAPES_ROOT to the Cityscapes root}"

ARCH="${ARCH:-segformer_b0}"
MODE="${MODE:-universal_csf}"
LOSS="${LOSS:-cospgd}"
TAU="${TAU:-0.25}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.01}"

BASE="--arch $ARCH --cityscapes_root $CS --img_h 1024 --img_w 2048"
GEOM="--patch_mode $MODE --patch_size 256 --patch_scale 0.25 --placement center"
OPT="--loss_fn $LOSS --lr $LR --lr_schedule cosine --batch_size 4 --num_workers 16"

python scripts/train.py $BASE $GEOM $OPT \
    --csf_threshold "$TAU" \
    --epochs "$EPOCHS" --val_every 5 --val_images 500 \
    --inference auto --slide_checkpoint \
    --out_root results/universal --tag "${TAG:-universal}"
