#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#
# Per-image overfit across the WHOLE Cityscapes validation set: the same
# attack as overfit.sh, run once per image, reported as a distribution.
# Add --resume to the command below to continue an interrupted run.

set -euo pipefail

module --ignore_cache load "cuda/11.8"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-csf-patch}"
export PYTHONNOUSERSITE=1

CS="${CITYSCAPES_ROOT:?set CITYSCAPES_ROOT to the Cityscapes root}"

ARCH="${ARCH:-segformer_b0}"
LOSS="${LOSS:-cospgd}"
MODE="${MODE:-csf}"
TAU="${TAU:-0.25}"
STEPS="${STEPS:-1000}"
LR="${LR:-0.2}"

BASE="--arch $ARCH --cityscapes_root $CS --img_h 1024 --img_w 2048"
GEOM="--patch_mode $MODE --patch_size 256 --patch_scale 0.25 --placement center"
CSF="--csf_threshold $TAU --csf_enforce realised"
RUN="--steps $STEPS --lr $LR --lr_schedule cosine --log_every 25"
PANELS="--n_panels 3 --panel_select best --select_key drop_remote"

OUT="results/validation/${ARCH}_${MODE}_${LOSS}_tau${TAU}_s${STEPS}_val"

python scripts/overfit_population.py $BASE $GEOM $CSF $RUN $PANELS \
    --from_image \
    --loss_fn "$LOSS" \
    --images all \
    --inference auto --slide_checkpoint \
    --out_dir "$OUT"
