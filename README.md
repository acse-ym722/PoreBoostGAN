# PoreBoostGAN

PoreBoostGAN is a lightweight digital-rock super-resolution repository with two closely related projects that share most code:

1. `PoreBoostGAN` (main): EDSR / ESRGAN / SwinIR / SwinIR+GAN.
2. `DistillSR` (optional): ESRGAN-style student-teacher distillation without GAN, supervised by final deep RRDB features.

This repository does **not** target medical images, segmentation, denoising, video restoration, or generic BasicSR development.

3D Z-axis reconstruction is also **not** implemented here. This codebase only handles XY super-resolution and slice-wise inference. Z reconstruction can be completed later in ImageJ or another external tool.

## Scope

- Domain: carbonate digital rocks.
- Data type: grayscale images.
- Model pipeline: native single-channel super-resolution, with optional distillation.
- Packaging: local `poreboostgan` package, no external `basicsr` dependency.

## Current Data Layout

The repository is already wired to the current carbonate dataset under:

```text
data/mengyang/CARBONATES/3DSR/
```

Important folders:

- `High/train`, `High/validation`: high-resolution full slices.
- `Low/train`, `Low/validation`: low-resolution full slices.
- `High_sub/train`, `High_sub/validation`: paired high-resolution training patches.
- `Low_sub/train`, `Low_sub/validation`: paired low-resolution training patches.
- `meta_info/`: meta files for paired patch loading.

## Install

Install PyTorch and TorchVision first according to your CUDA environment, then install the repo dependencies:

```bash
pip install -e .
```

If you prefer a plain requirements install:

```bash
pip install -r requirements.txt
```

## Main Training

The default main configuration is the grayscale SwinIR + GAN setup:

```bash
python src/train.py -opt configs/train/poreboostgan_swinir_gan_x4_gray.yml
```

## DistillSR Training (Optional)

The distillation project is separated by model/config, while reusing the same package and data pipeline.

```bash
python src/train.py -opt configs/train/poreboostgan_distill_rrdb_x4_gray.yml
```

Core idea:

- teacher: pretrained RRDB SR network (frozen)
- student: lightweight RRDB SR network (trainable)
- objective: deep feature supervision on the last RRDB trunk feature (`return_feats=True`)
- GAN loss is not used in this project

## Main Inference

Update `dataroot_lq` and `pretrain_network_g` in:

```text
configs/infer/poreboostgan_swinir_gan_x4_gray.yml
```

Then run:

```bash
python src/infer.py -opt configs/infer/poreboostgan_swinir_gan_x4_gray.yml
```

`results/<run_name>/visualization/` will contain the restored slices.

## Inference Commands (4 Experiments)

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

DistillSR student inference:

```bash
python src/infer.py -opt configs/infer/poreboostgan_distill_rrdb_x4_gray.yml
```

## Quick Smoke Training (1~5 Steps)

The following commands are for quick smoke runs only.  
They set `total_iter=5`, disable validation, and effectively disable mid-run checkpoint saving to reduce disk writes.

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

## Active Layout

- `poreboostgan/`: the active lightweight package for digital-rock super-resolution.
- `configs/`: the active training and inference configs for the grayscale carbonate workflow.
- `tools/`: dataset preparation and downsampling utilities for grayscale slices.
- `src/`: lightweight entry scripts (`train.py`, `test.py`, `app.py`, `infer.py`) that call into `poreboostgan`.

## Extrapolation Workflow

Extrapolation is just repeated inference:

1. Run inference on the current low-resolution input.
2. Use the previous output folder as the next `dataroot_lq`.
3. Run inference again.

This is how the repository handles super-resolution beyond the original instrument resolution.

## Downsampling Study

For representation studies, use:

```bash
python tools/downsample_folder.py --input-dir <input> --output-dir <output> --scale 4
```

This is intended for experiments such as:

- ultra-high-resolution downsampled to high-resolution
- high-resolution downsampled to original low-resolution

## Dataset Utilities

Generate paired XY patches from aligned high/low folders:

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

Generate meta info:

```bash
python tools/generate_meta_info.py \
  --gt-dir data/mengyang/CARBONATES/3DSR/High_sub/train \
  --output data/mengyang/CARBONATES/3DSR/meta_info/meta_info_high_carbon.txt
```

## Notes

- The code now treats digital-rock data as grayscale by default through `img_flag: grayscale`.
- VGG perceptual loss still works: single-channel tensors are repeated to 3 channels only inside the perceptual extractor.
