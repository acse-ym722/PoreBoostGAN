# Data Layout and Preparation

This repository only tracks data documentation, scripts, and directory skeletons.
Raw and generated dataset files are ignored by `.gitignore`.

## Open Dataset (Raw 3D Digital Rocks)

- Link: https://data.mendeley.com/datasets/6kvtfb5kts/1
- Raw format: `.tif` 3D volumes

Volumes:

- `Biogenic_16um.tif`
- `Biogenic_4um_filter.tif`
- `Biogenic_4um.tif`
- `MRCCM_10.72um.tif`
- `MRCCM_2.68um.tif`

## Expected Local Directory

```text
data/mengyang/CARBONATES/3DSR/
  High/train
  High/validation
  Low/train
  Low/validation
  High_sub/train
  High_sub/validation
  Low_sub/train
  Low_sub/validation
  meta_info
```

## Preparation Workflow

1. Slice raw 3D TIFF volumes into aligned XY slices (external tool such as ImageJ).
2. Place paired full slices into `High/*` and `Low/*`.
3. Build paired patches with `tools/extract_xy_patches.py`.
4. Build metadata with `tools/generate_meta_info.py`.
