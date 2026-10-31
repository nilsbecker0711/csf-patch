# csf-patch (Master Thesis)

Adversarial patches for semantic segmentation under a human contrast
sensitivity (CSF) constraint, on Cityscapes.

The patch is a residual whose spectrum is bounded by the CSF, so the amplitude
budget at each spatial frequency is set by what a human observer can see at
that frequency rather than by a uniform L-infinity ball. The budget is scaled
by a single visibility threshold, **tau**, in JND units: `tau = 1.0` is
nominally the detection threshold.

---

## 1. Install

```bash
conda env create -f env.yml
conda activate csf-patch
pip install -e .
```

`env.yml` pins the whole environment (Python 3.9, torch 1.11 + cu113,
mmcv-full 1.5, mmsegmentation 0.27). For an existing interpreter use
`requirements.txt`, installing `numpy` and `setuptools` first because
`mmcv-full` and `pycocotools` compile against them:

```bash
pip install numpy==1.26.4 setuptools==60.2.0 && pip install -r requirements.txt
```

The `internimage_*` architectures also need the DCNv3 CUDA extension, built
from the InternImage source tree. Every other architecture runs without it.

## 2. Point it at your data and checkpoints

```bash
export CITYSCAPES_ROOT=/path/to/cityscapes
export PATCHREACH_CHECKPOINTS=/path/to/checkpoints
```

`patchreach/models/registry.py` maps an architecture name to a config and a
checkpoint under `PATCHREACH_CHECKPOINTS`. Available names:

```
segformer_b0  segformer_b5  setr_pup  internimage_t/l/xl  swin_t  convnext_t
deeplabv3plus_r18/r50/r101  unet_s5d16  pidnet_s/m/l
```

`--cfg_path` and `--weights` override the registry per call. The config must
match the checkpoint variant — a B0 config will not build B5.

## 3. Run

Four SLURM launchers, each configured from the environment. Every variable has
a default except `CITYSCAPES_ROOT`.

```bash
sbatch overfit.sh      # one patch against ONE image
sbatch universal.sh    # one patch shared across the train split
sbatch validation.sh   # per-image overfit across the whole val set
sbatch eval.sh         # evaluate a saved patch
```

```bash
ARCH=segformer_b5 IMAGE=420 TAU=0.25 STEPS=1000 sbatch overfit.sh
MODE=universal_csf EPOCHS=20 LR=0.01 sbatch universal.sh
CHECKPOINT=results/overfit/<run>/best.pt sbatch eval.sh
```

To run directly instead of through SLURM, call the Python entry point; the
launcher shows the exact invocation.

---

## Mechanic 1 — overfitting a single image

One patch optimised against one image. This is the strongest the attack gets:
nothing has to generalise, so it measures what the constraint permits at all.

**What happens per step** (`patchreach/patch/optimise.py`):

1. **Clean forward.** `clean_pred = model(img).argmax(1)` — the reference the
   loss and the metrics are scored against.
2. **Resolve placement.** Where the patch goes. `semantic` and `gradcam` read
   the clean prediction, which is why step 1 comes first; reversing the two
   silently falls back to centre.
3. **Render the patch.** The parameter becomes pixels. Under `csf` the
   parameter is a residual, projected so every rfft bin sits inside its CSF
   budget at `tau`.
4. **Paste** it into the frame at the resolved position.
5. **Forward and loss.** The objective is scored *outside* the patch footprint,
   so the number measures reach into the scene, not the patch covering its own
   pixels.
6. **Backward, step, re-project.** Adam updates the parameter; the projection
   puts it back inside the budget.
7. **Log** every `--log_every` steps, and checkpoint `best.pt` if this is the
   best score so far.

**Run it:**

```bash
python scripts/overfit.py \
    --arch segformer_b0 --cityscapes_root $CITYSCAPES_ROOT \
    --img_h 1024 --img_w 2048 \
    --patch_mode csf --from_image \
    --patch_size 256 --patch_scale 0.25 --placement center \
    --csf_threshold 0.25 --csf_enforce realised \
    --loss_fn cospgd --image 420 \
    --steps 1000 --lr 0.2 --lr_schedule cosine --log_every 25
```

**Parameters** (`scripts/overfit.py`)

| flag | default | what it does |
|---|---|---|
| `--image` | `2` | which validation image to attack, by index |
| `--steps` | `300` | attack iterations. 400 was measured truncating the run; 1000 converges |
| `--lr` | `0.01` | step size. Use `0.2` for `csf`, whose parameter is effectively a direction |
| `--lr_schedule` | `cosine` | anneal lr to zero. A flat lr was measured swinging 9.6 mIoU across identical runs |
| `--optimiser` | `adam` | `adam`, or `sign` for PGD's update at equal lr |
| `--seeds` | `1` | repeat the whole attack N times and report mean ± sd |
| `--log_every` | `20` | logging interval — and the granularity at which `best.pt` is written |
| `--from_image` | off | initialise the patch base with the image region it replaces. Use with `csf` |
| `--loss_fn` | `cospgd` | `ce`, `cospgd`, `ce_target`, `ipatch_cospgd`, `tsallis`, `margin`, `cos_margin` |
| `--target_class` | `8` | target trainId for the targeted objectives |
| `--exclude_footprint` | off | score the loss outside the patch only |

## Mechanic 2 — the universal patch

**One** patch shared across the whole training split, pasted onto whatever
content it lands on. Nothing is fitted per image, so this is the honest threat
model and a much harder problem than Mechanic 1.

**What happens** (`scripts/train.py`):

1. One patch parameter is created and kept for the entire run.
2. For each batch: paste the *same* patch into every image, forward, accumulate
   the loss, backward, step, re-project onto the CSF budget.
3. Every `--val_every` epochs, evaluate on `--val_images` validation images and
   checkpoint if it improved.
4. `--resume` restores the patch, Adam's moments, the scheduler position and
   the RNG, so a long run can be fed to a queue in slices.

**Run it:**

```bash
python scripts/train.py \
    --arch segformer_b0 --cityscapes_root $CITYSCAPES_ROOT \
    --img_h 1024 --img_w 2048 \
    --patch_mode universal_csf \
    --patch_size 256 --patch_scale 0.25 --placement center \
    --csf_threshold 0.25 \
    --loss_fn cospgd --lr 0.01 --lr_schedule cosine \
    --batch_size 4 --num_workers 16 \
    --epochs 20 --val_every 5 --val_images 500
```

**Parameters** (`scripts/train.py`)

| flag | default | what it does |
|---|---|---|
| `--epochs` | `100` | passes over the training split |
| `--lr` | `0.005` | step size. Use `0.01` for `universal_csf`; its parameter *is* the residual in pixel units, so `0.2` would discard the previous residual each step |
| `--lr_schedule` | `plateau` | use `cosine` for `universal_csf` |
| `--batch_size` | `4` | images per step |
| `--num_workers` | `4` | DataLoader workers. Raise to 8–16 — Cityscapes decodes a 2048x1024 PNG per sample and can starve the GPU |
| `--val_every` | `5` | validate every N epochs |
| `--val_images` | `20` | validation images per evaluation. 500 is the full val set |
| `--resume` | off | continue from `train_state.pt`. Requires `--out_dir` |
| `--out_dir` | auto | explicit output directory. Required with `--resume` |

> `universal_csf` is `train.py` only; on a single image it *is* `csf`. It
> requires `--shape square`, because the spectral bound is defined on the full
> footprint. `--placement gradcam` is per-image by nature and is rejected here
> (`scripts/train.py`), since one shared patch cannot follow a different
> image's sensitivity map each batch.

## Mechanic 3 — the whole validation set

Mechanic 1, run once per validation image, reported as a distribution. One
image is an anecdote; this is the measurement.

```bash
python scripts/overfit_population.py \
    --arch segformer_b0 --cityscapes_root $CITYSCAPES_ROOT \
    --img_h 1024 --img_w 2048 \
    --patch_mode csf --from_image \
    --patch_size 256 --patch_scale 0.25 --placement center \
    --csf_threshold 0.25 --csf_enforce realised \
    --loss_fn cospgd --steps 1000 --lr 0.2 --lr_schedule cosine \
    --images all \
    --out_dir results/validation/b0_csf
```

| flag | default | what it does |
|---|---|---|
| `--images` | `random` | `all` (full val set), `random`, `fixed10`, or a list like `'2 5 45'` |
| `--n_images` | `100` | population size when `--images random` |
| `--sample_seed` | `0` | seed for the random sample. Record it — it is what makes the subset reproducible, and raising `--n_images` with the same seed extends the sample rather than replacing it |
| `--resume` | off | skip images already done. Requires `--out_dir` |
| `--n_panels` | — | how many per-image figure panels to write |
| `--panel_select` | — | `best` picks the strongest images; the distribution figure shows where they sit |

Output: `summary.json` with the per-image drop and its bootstrap CI,
`distribution.png`, pooled diagnostics in `aggregate/`, and every image's patch
under `patches/`.

## Mechanic 4 — evaluation

Scores a saved patch. No training.

```bash
python scripts/evaluate.py \
    --arch segformer_b0 --cityscapes_root $CITYSCAPES_ROOT \
    --img_h 1024 --img_w 2048 \
    --checkpoint results/overfit/<run>/best.pt \
    --images all --inference auto
```

| flag | default | what it does |
|---|---|---|
| `--checkpoint` | — | a checkpoint **file** written by `Patch.save`, e.g. `best.pt`, not a run directory |
| `--patch_png` | — | evaluate the pixels of this PNG instead. With a checkpoint it replaces only the parameter, the checkpoint still supplying the geometry; on its own it is a square `raw` patch at the frame centre |
| `--patch_scale` | from ckpt | patch side as a fraction of image height. Only for `--patch_png` *without* a checkpoint; ignored when a checkpoint supplies the geometry |
| `--images` | `fixed10` | `all`, `fixed10`, or an explicit list |
| `--diagnostics_on` | `0` | how many images get the full diagnostic suite |
| `--no_panels` | off | skip the figures |

---

## The patch itself

These apply to every task above.

| flag | default | what it does |
|---|---|---|
| `--patch_mode` | `raw` | `raw` (free pixels), `csf` (per-image CSF residual), `universal_csf` (shared CSF residual), `lap` (Tan et al. regularisers), `gan` (BigGAN latent), `raw_ganinit` |
| `--patch_size` | `128` | the parameter's own resolution |
| `--patch_scale` | `0.25` | pasted side as a fraction of image height |
| `--patch_scale_ref` | `height` | `height`: side = `scale*H`. `area`: side = `scale*sqrt(H*W)`, constant frame coverage at any input shape |
| `--placement` | `center` | `center`, `fixed`, `semantic` (onto a class), `gradcam` (the sensitivity hotspot) |
| `--placement_class` | `0` | trainId for `semantic`. 0 = road, omnipresent in Cityscapes |
| `--pixel_param` | `sigmoid` | `raw` only. `sigmoid` never has a gradient killed by a bound; `direct` is PGD's projected pixel |
| `--raw_init` | `grey` | `random` gives the random-patch baseline — combine with `--lr 0` |

**The CSF constraint**

| flag | default | what it does |
|---|---|---|
| `--csf_threshold` | `0.25` | **tau**, the visibility budget in JND. 1.0 is nominally the detection threshold |
| `--csf_enforce` | `nominal` | `nominal` bounds the residual we *intend* to add; `realised` bounds what survives compositing, i.e. what an observer sees. Clipping against real content was measured inflating visibility to 2.8x tau. **Use `realised` for any number quoted with a tau attached** |
| `--csf_param` | `pgd` | `csf` only. `pgd` clamps each rfft bin to its budget and requires `--shape square`; `squash` is the original reparameterisation, kept so earlier runs reproduce and needed for shaped patches. The taus are not interchangeable — compare at matched *realised* visibility |
| `--csf_composite` | `clip` | `universal_csf` only. `clip` is the honest default and reports `frac_clipped`; `fit` rescales instead, at the cost of a per-image scale |
| `--csf_lref` | `0.0` | reference linear luminance. 0 keeps the legacy mu=0.5 convention; `0.0971` is the measured Cityscapes train median and is ~1.9x tighter at Nyquist |
| `--csf_model` | `barten` | `barten` or `sso` |

**Model and resolution**

| flag | default | what it does |
|---|---|---|
| `--img_h` / `--img_w` | `512` / `1024` | input resolution. Cityscapes is natively 1024x2048 |
| `--scale` | `resize` | `resize` squeezes the whole frame; `crop` cuts a window from the native frame, preserving pixel density |
| `--inference` | `whole` | `whole` is one forward and what every reach measurement uses. `auto` honours the checkpoint's `test_cfg`. `slide` makes each window an independent forward, which caps reach — do not use it for cross-architecture comparisons |
| `--slide_checkpoint` | off | recompute each window in the backward pass instead of holding every window's graph. Use when the attack OOMs at native resolution |
| `--low_pr` | off | 16-bit forward. Roughly halves activation memory; compare only against a baseline run with the same flag |
| `--seed` | `42` | RNG seed |

---

## Two things that will cost you a run

**Geometry must be pinned.** The patch is pasted without resampling only when
`--patch_size == int(img_h * patch_scale)` — at 1024 with scale 0.25 that is
256. `universal_csf` refuses to run otherwise, because resampling a residual
resamples the spectrum its budget was projected onto. The other modes rescale
silently.

**`--csf_threshold` is read only in `csf` and `universal_csf`.** In `raw`,
`lap` and `gan` it is accepted and has no effect, so a "tau 0.25 raw" run is
simply unconstrained.

## Layout

```
patchreach/              the library — no argparse, no scripts
  patch/spec.py            PatchConfig + Patch, the central abstraction
  patch/csf.py             contrast sensitivity, budgets, projection
  patch/optimise.py        the single-image attack loop
  patch/iap.py             Invisible Adversarial Patch (Dutta & Zhang, 2025)
  patch/lap.py             LAP regularisers (Tan et al., 2021)
  patch/placement.py       center / fixed / semantic / gradcam
  patch/shape.py           silhouette masking
  patch/conditional_generator.py   image-conditioned generator
  losses/                  ce, ce_target, cospgd, ipatch_cospgd, margin, tsallis, reach
  models/registry.py       one entry per checkpoint
  models/wrapper.py        gradient-preserving mmseg wrappers
  data/cityscapes.py       dataset, palette, remap, normalisation
  metrics/                 mIoU, attack effect, reach curves, population stats
  diagnostics/             geometric, semantic, spectral, universal reporting

scripts/                 entry points; all logic lives in the library
analysis/                result aggregation and figures
defences/                JEDI, SAC, localisation metrics, defended evaluation
configs/                 mmseg PIDNet configs, experiment YAMLs
GANLatentDiscovery/      vendored, for --patch_mode gan (see its licenses/)
```

Results are written flat, one directory per run holding `config.json`,
`results.json` and the patch. `analysis/build_index.py` walks them into a
single CSV, so a new experimental axis becomes a new column rather than a new
directory level.
