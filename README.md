# PoreBoostGAN

PoreBoostGAN is a lightweight digital-rock super-resolution repository for carbonate cores.
It now supports both legacy XY 2D SR and stage-1 end-to-end 3D cubic SR.

This repository supports the SPE Journal paper:

- **Super-Resolution Carbonate Rock Image Beyond Instrument Limitations**
- Authors: Yang Meng, Kunning Tang, Heping Xie, Zhangxin Chen, Ying Teng, Yuntian Chen, Cunbao Li, and Senyou An
- Journal: **SPE Journal**, published online July 2026
- DOI: https://doi.org/10.2118/234678-PA
- PDF: https://onepetro.org/SJ/article-pdf/doi/10.2118/234678-PA/5390025/spe-234678-pa.pdf

The paper introduces **SwinIRGAN**, a sliding-window-attention super-resolution framework for carbonate digital rocks. Reported results include 99.48% Euler-number accuracy on the biogenic carbonate dataset, 97.05% accuracy for higher-resolution extrapolation, and a 15.59% Euler-number improvement over the baseline on the multiresolution complex carbonates micro-computed tomography (MRCCM) dataset.

This repo contains two related projects with shared code:

1. `PoreBoostGAN` (main): EDSR / ESRGAN / SwinIR / SwinIR+GAN
2. `DistillSR` (secondary): RRDB feature distillation without GAN

This repository does **not** target medical imaging, segmentation, denoising, video restoration, or generic BasicSR development.
Current focus is pore-scale super-resolution for carbonate digital rocks.

## Citation

If this repository or dataset helps your work, please cite:

```bibtex
@article{10.2118/234678-PA,
    author = {Meng, Yang and Tang, Kunning and Xie, Heping and Chen, Zhangxin and Teng, Ying and Chen, Yuntian and Li, Cunbao and An, Senyou},
    title = {Super-Resolution Carbonate Rock Image Beyond Instrument Limitations},
    journal = {SPE Journal},
    pages = {1-8},
    year = {2026},
    month = {07},
    abstract = {Carbonate rocks, as complex multiscale porous media, present major imaging challenges because of intricate structures and strong heterogeneity. To address the trade-off between field of view (FOV) and resolution, we introduce the Swin transformer for image restoration generative adversarial network (SwinIRGAN), a super-resolution (SR) framework based on sliding-window attention that captures long-range features efficiently. The model balances global consistency with high-frequency detail preservation and learns the mapping between low-resolution (LR) and high-resolution (HR) images. Using a biogenic carbonate data set, SwinIRGAN achieves 99.48\% accuracy in Euler's number and 97.05\% accuracy in higher-resolution extrapolation. For the multiresolution complex carbonates micro-computed tomography (micro-CT, MRCCM) data set, the proposed reconstruction and extrapolation workflow improves Euler's number by 15.59\% compared with the baseline. Results show that SwinIRGAN preserves mineralogical and topological characteristics across scales and provides more reliable digital rocks for pore-scale analysis and flow simulation.},
    issn = {1086-055X},
    doi = {10.2118/234678-PA},
    url = {https://doi.org/10.2118/234678-PA},
    eprint = {https://onepetro.org/SJ/article-pdf/doi/10.2118/234678-PA/5390025/spe-234678-pa.pdf},
}
```

## Scope

- Domain: carbonate digital rocks
- Data type: grayscale or multi-channel volume data
- Pipeline: 2D SR (legacy) + 3D cubic SR (stage-1)
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
  High_sub3d/
    train/
    validation/
  Low_sub3d/
    train/
    validation/
  meta_info/
```

## 3D End-to-End Dataset Prep (Stage-1)

Prepare aligned paired HR/LR volumes (`.npy/.npz/.tif/.tiff`) under:

- `High/train`, `High/validation`
- `Low/train`, `Low/validation`

Extract paired cubic patches (`subimage^3`):

```bash
python tools/extract_xyz_cubes.py \
  --hr-dir data/mengyang/CARBONATES/3DSR/High/train \
  --lr-dir data/mengyang/CARBONATES/3DSR/Low/train \
  --hr-out-dir data/mengyang/CARBONATES/3DSR/High_sub3d/train \
  --lr-out-dir data/mengyang/CARBONATES/3DSR/Low_sub3d/train \
  --scale 4 \
  --hr-cube-size 64 \
  --step 64 \
  --data-format auto
```

Optional meta info for 3D patches:

```bash
python tools/generate_meta_info_3d.py \
  --gt-dir data/mengyang/CARBONATES/3DSR/High_sub3d/train \
  --output data/mengyang/CARBONATES/3DSR/meta_info/meta_info_high_carbon_3d.txt
```

## Legacy XY Dataset Prep (2D)

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
conda env create -f environment_3d.yml
conda activate pore3d
pip install -e .
```

Or use your existing env (for example `conda activate pore`):

```bash
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

## Main Training

3D EDSR (stage-1):

```bash
python src/train.py -opt configs/train/poreboostgan_edsr3d_x4.yml
```

3D ESRGAN (x2, RRDB):

```bash
python src/train.py -opt configs/train/poreboostgan_esrgan3d_x2.yml
```

3D ESRGAN (x2, 4090-friendly):

```bash
python src/train.py -opt configs/train/poreboostgan_esrgan3d_x2_4090.yml
```

3D SwinIRGAN (x4, 4090-friendly):

```bash
python src/train.py -opt configs/train/poreboostgan_swinirgan3d_x4_4090.yml
```

For 3D validation with `save_img: true`, outputs include:

- SR volume: `*.npy`
- center slices: `*_x.png`, `*_y.png`, `*_z.png`
- SR/GT comparisons: `*_x_sr_gt.png`, `*_y_sr_gt.png`, `*_z_sr_gt.png`

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

3D EDSR:

```bash
python src/infer.py -opt configs/infer/poreboostgan_edsr3d_x4.yml
```

3D ESRGAN (x2):

```bash
python src/infer.py -opt configs/infer/poreboostgan_esrgan3d_x2.yml
```

3D SwinIRGAN (x4):

```bash
python src/infer.py -opt configs/infer/poreboostgan_swinirgan3d_x4_4090.yml
```

## Full 3D Reconstruction (Seam-Reduced)

Use tiled 3D inference for large LR volumes and stitch with center-trust strategy:

- each block is inferred with extra context
- only center core is written back
- boundary predictions are discarded to reduce seams

```bash
python tools/infer_3d_volume_reconstruct.py \
  --input data/c2_360_360_1024.tif \
  --output results/c2_sr_x2_reconstruct.tif \
  --model-path experiments/poreboostgan_esrgan3d_x2_4090/models/net_g_2000.pth \
  --config experiments/poreboostgan_esrgan3d_x2_4090/poreboostgan_esrgan3d_x2_4090.yml \
  --device cuda \
  --fp16 \
  --core-size 64 \
  --context 16 \
  --save-dtype uint8
```

Inference output for 3D is saved as `.tif` volume files in `results/<name>/visualization/`.

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
- 3D datasets use `PairedVolumeDataset` / `SingleVolumeDataset` and expect tensors in `C,D,H,W`.
