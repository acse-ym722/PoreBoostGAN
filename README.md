# PoreBoostGAN

PoreBoostGAN is a lightweight digital-rock super-resolution repository for carbonate cores (grayscale XY slices).

This repo contains two related projects with shared code:

1. `PoreBoostGAN` (main): EDSR / ESRGAN / SwinIR / SwinIR+GAN
2. `DistillSR` (secondary): RRDB feature distillation without GAN

This repository does **not** target medical imaging, segmentation, denoising, video restoration, or generic BasicSR development.
It only handles XY super-resolution and slice-wise inference. Z-axis reconstruction is external (for example ImageJ).

## Scope

- Domain: carbonate digital rocks
- Data type: grayscale
- Pipeline: single-channel SR + optional distillation
- Packaging: local `poreboostgan` package, no external `basicsr` dependency

## Open Dataset (Raw 3D TIFF)

- Mendeley dataset: https://data.mendeley.com/datasets/6kvtfb5kts/1
- Raw data format: 3D digital-rock TIFF volumes (`.tif`)

Provided raw volumes:

- `Biogenic_16um.tif`
- `Biogenic_4um_filter.tif`
- `Biogenic_4um.tif`
- `MRCCM_10.72um.tif`
- `MRCCM_2.68um.tif`

## Data Policy in This Repo

`data/` is **not** blanket ignored anymore. We keep:

- data format documentation
- dataset preparation scripts
- directory skeleton (`.gitkeep`)

But raw/derived dataset binaries are ignored and must not be uploaded to GitHub:

- `.tif/.tiff`, image patches, arrays, caches, archives, etc.

## Expected Local Data Layout

```text
data/mengyang/CARBONATES/3DSR/
  High/
    train/
    validation/
  Low/
    train/
    validation/
  High_sub/
    train/
    validation/
  Low_sub/
    train/
    validation/
  meta_info/
```

## From 3D Volume to XY SR Dataset

1. Slice raw 3D TIFF volumes into aligned 2D XY slices (external tool, e.g. ImageJ).
2. Put aligned full-slice pairs into:
   - `High/train`, `High/validation`
   - `Low/train`, `Low/validation`
3. Extract paired patches:

```bash
python tools/extract_xy_patches.py \
  --hr-dir data/mengyang/CARBONATES/3DSR/High/train \
  --lr-dir data/mengyang/CARBONATES/3DSR/Low/train \
  --hr-out-dir data/mengyang/CARBONATES/3DSR/High_sub/train \
  --lr-out-dir data/mengyang/CARBONATES/3DSR/Low_sub/train \
  --scale 4 \
  --hr-crop-size 384 \
  --step 320
```

4. Generate meta info:

```bash
python tools/generate_meta_info.py \
  --gt-dir data/mengyang/CARBONATES/3DSR/High_sub/train \
  --output data/mengyang/CARBONATES/3DSR/meta_info/meta_info_high_carbon.txt
```

## Install

Install PyTorch/TorchVision for your CUDA first, then:

```bash
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

## Main Training

SwinIR+GAN (default main setup):

```bash
python src/train.py -opt configs/train/poreboostgan_swinir_gan_x4_gray.yml
```

## DistillSR (Second Project)

DistillSR reuses the same pipeline/data, but switches objective:

- teacher: pretrained RRDB (`frozen`)
- student: lightweight RRDB (`trainable`)
- deep supervision: align final RRDB trunk feature (`return_feats=True`)
- GAN loss: removed

Loss structure:

- `L = w_feat * ||F_s - F_t||_1 + w_pix * ||SR_s - SR_t||_1 + w_gt * ||SR_s - HR||_1`

Default teacher is set to your 300k ESRGAN checkpoint:

- `experiments/poreboostgan_esrgan_x4_gray/models/net_g_300000.pth`

Train DistillSR:

```bash
python src/train.py -opt configs/train/poreboostgan_distill_rrdb_x4_gray.yml
```

Infer DistillSR student:

```bash
python src/infer.py -opt configs/infer/poreboostgan_distill_rrdb_x4_gray.yml
```

## Released Weights

- Teacher checkpoint (ESRGAN RRDB, 300k):
  - https://github.com/acse-ym722/PoreBoostGAN/releases/tag/v0.2.0-distillsr-teacher-300k
  - asset: `net_g_300000.pth`
  - sha256: `8d9a80a55bad0fa0be8c0f41e0523ecddf911820549abbbb0e3b0f8dd1371262`

## Inference (Four Main Experiments)

EDSR:

```bash
python src/infer.py -opt configs/infer/poreboostgan_edsr_x4_gray.yml
```

ESRGAN:

```bash
python src/infer.py -opt configs/infer/poreboostgan_esrgan_x4_gray.yml
```

SwinIR:

```bash
python src/infer.py -opt configs/infer/poreboostgan_swinir_x4_gray.yml
```

SwinIR+GAN:

```bash
python src/infer.py -opt configs/infer/poreboostgan_swinir_gan_x4_gray.yml
```

## Quick Smoke Training (1~5 Steps)

EDSR smoke:

```bash
python src/train.py -opt configs/train/poreboostgan_edsr_x4_gray.yml \
  --force_yml train:total_iter=5 val=none logger:use_tb_logger=false logger:save_checkpoint_freq=999999999 \
  datasets:train:num_worker_per_gpu=0 datasets:train:batch_size_per_gpu=1 logger:print_freq=1
```

ESRGAN smoke:

```bash
python src/train.py -opt configs/train/poreboostgan_esrgan_x4_gray.yml \
  --force_yml train:total_iter=5 val=none logger:use_tb_logger=false logger:save_checkpoint_freq=999999999 \
  datasets:train:num_worker_per_gpu=0 datasets:train:batch_size_per_gpu=1 logger:print_freq=1
```

SwinIR smoke:

```bash
python src/train.py -opt configs/train/poreboostgan_swinir_x4_gray.yml \
  --force_yml train:total_iter=5 val=none logger:use_tb_logger=false logger:save_checkpoint_freq=999999999 \
  datasets:train:num_worker_per_gpu=0 datasets:train:batch_size_per_gpu=1 logger:print_freq=1
```

SwinIR+GAN smoke:

```bash
python src/train.py -opt configs/train/poreboostgan_swinir_gan_x4_gray.yml \
  --force_yml train:total_iter=5 val=none logger:use_tb_logger=false logger:save_checkpoint_freq=999999999 \
  datasets:train:num_worker_per_gpu=0 datasets:train:batch_size_per_gpu=1 logger:print_freq=1
```

DistillSR smoke:

```bash
python src/train.py -opt configs/train/poreboostgan_distill_rrdb_x4_gray.yml \
  --force_yml train:total_iter=5 val=none logger:use_tb_logger=false logger:save_checkpoint_freq=999999999 \
  datasets:train:num_worker_per_gpu=0 datasets:train:batch_size_per_gpu=1 logger:print_freq=1
```

## Extrapolation Workflow

Extrapolation is repeated inference:

1. run inference on current input
2. use previous output as next `dataroot_lq`
3. run inference again

## Downsampling Study

```bash
python tools/downsample_folder.py --input-dir <input> --output-dir <output> --scale 4
```

## Notes

- Grayscale is enforced by `img_flag: grayscale`.
- If perceptual loss is used, channel repeat to 3 only happens inside feature extractor.
