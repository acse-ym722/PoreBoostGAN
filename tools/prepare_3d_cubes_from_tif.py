import argparse
import math
import random
from pathlib import Path

import numpy as np
import tifffile


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare 3D SR cubes from HR tif volumes. Example: HR 128^3 -> LR 64^3 (x2) or 32^3 (x4).')
    parser.add_argument('--input-dir', type=Path, default=Path('data'), help='Folder containing HR .tif/.tiff files.')
    parser.add_argument(
        '--high-out-dir',
        type=Path,
        default=Path('data/mengyang/CARBONATES/3DSR/High_sub'),
        help='Output root for HR cubes. train/validation will be created automatically.')
    parser.add_argument(
        '--low-out-dir',
        type=Path,
        default=Path('data/mengyang/CARBONATES/3DSR/Low_sub'),
        help='Output root for LR cubes. train/validation will be created automatically.')
    parser.add_argument('--hr-cube-size', type=int, default=128, help='HR cubic size (REV lower bound is 128).')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 4], help='Downsample scale for LR generation.')
    parser.add_argument('--step', type=int, default=128, help='Sliding step on HR grid.')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation split ratio.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for split.')
    parser.add_argument('--max-cubes', type=int, default=0, help='Optional hard cap for number of cubes. 0 means no cap.')
    parser.add_argument('--overwrite', action='store_true', help='Clear existing output .npy files before writing.')
    parser.add_argument(
        '--output-format',
        type=str,
        default='npy',
        choices=['npy', 'tif', 'both'],
        help='Output file format. npy for training speed, tif for easier inspection.')
    parser.add_argument(
        '--save-dtype',
        type=str,
        default='uint8',
        choices=['uint8', 'float32'],
        help='Storage dtype for saved cubes. uint8 is much smaller on disk.')
    return parser.parse_args()


def list_tif_files(input_dir: Path):
    tif_files = sorted(list(input_dir.glob('*.tif')) + list(input_dir.glob('*.tiff')))
    return [path for path in tif_files if path.is_file()]


def normalize_to_float32(volume: np.ndarray):
    if np.issubdtype(volume.dtype, np.integer):
        max_val = float(np.iinfo(volume.dtype).max)
        return volume.astype(np.float32) / max_val
    return volume.astype(np.float32)


def to_dhwc(volume: np.ndarray):
    if volume.ndim == 3:
        return volume[..., None]
    if volume.ndim != 4:
        raise ValueError(f'Expected 3D/4D volume, got shape={volume.shape}')

    # Keep it simple: prefer DHWC when last axis looks like channel count.
    if volume.shape[-1] <= 8:
        return volume
    if volume.shape[0] <= 8:
        return np.transpose(volume, (1, 2, 3, 0))
    # Fallback: treat as DHWC.
    return volume


def sliding_positions(length: int, cube_size: int, step: int):
    if length < cube_size:
        return []
    pos = list(range(0, length - cube_size + 1, step))
    if pos[-1] != length - cube_size:
        pos.append(length - cube_size)
    return pos


def _downsample_x2_mean(cube: np.ndarray):
    d, h, w, c = cube.shape
    if d % 2 != 0 or h % 2 != 0 or w % 2 != 0:
        raise ValueError(f'Cube shape {cube.shape} is not divisible by 2.')
    return cube.reshape(d // 2, 2, h // 2, 2, w // 2, 2, c).mean(axis=(1, 3, 5)).astype(np.float32)


def downsample_mean(hr_cube: np.ndarray, scale: int):
    if scale not in (2, 4):
        raise ValueError(f'Unsupported scale={scale}.')
    lr = hr_cube
    num_down = int(math.log2(scale))
    for _ in range(num_down):
        lr = _downsample_x2_mean(lr)
    return lr


def clear_old_files(root: Path, output_format: str):
    patterns = []
    if output_format in ('npy', 'both'):
        patterns.append('*.npy')
    if output_format in ('tif', 'both'):
        patterns.extend(['*.tif', '*.tiff'])

    for split in ('train', 'validation'):
        split_dir = root / split
        if split_dir.exists():
            for pat in patterns:
                for f in split_dir.glob(pat):
                    f.unlink()


def ensure_dirs(root: Path):
    (root / 'train').mkdir(parents=True, exist_ok=True)
    (root / 'validation').mkdir(parents=True, exist_ok=True)


def _to_save_dtype(cube: np.ndarray, save_dtype: str):
    if save_dtype == 'float32':
        return cube.astype(np.float32)
    # uint8 path
    cube = np.clip(cube, 0.0, 1.0)
    return np.round(cube * 255.0).astype(np.uint8)


def save_cube(cube: np.ndarray, out_path_no_ext: Path, output_format: str, save_dtype: str):
    cube_to_save = _to_save_dtype(cube, save_dtype)
    if output_format in ('npy', 'both'):
        np.save(out_path_no_ext.with_suffix('.npy'), cube_to_save)
    if output_format in ('tif', 'both'):
        tifffile.imwrite(str(out_path_no_ext.with_suffix('.tif')), cube_to_save)


def main():
    args = parse_args()

    if args.hr_cube_size < 128:
        raise ValueError('REV constraint: hr-cube-size must be >= 128.')
    if args.hr_cube_size % args.scale != 0:
        raise ValueError('hr-cube-size must be divisible by scale.')
    if args.step <= 0:
        raise ValueError('step must be positive.')
    if not (0 <= args.val_ratio < 1):
        raise ValueError('val-ratio must be in [0, 1).')

    tif_files = list_tif_files(args.input_dir)
    if not tif_files:
        raise ValueError(f'No .tif/.tiff files found in {args.input_dir}')

    ensure_dirs(args.high_out_dir)
    ensure_dirs(args.low_out_dir)
    if args.overwrite:
        clear_old_files(args.high_out_dir, args.output_format)
        clear_old_files(args.low_out_dir, args.output_format)

    all_indices = []
    volume_cache = {}

    for tif_path in tif_files:
        volume = tifffile.imread(str(tif_path))
        volume = normalize_to_float32(to_dhwc(np.asarray(volume)))
        d, h, w, _ = volume.shape
        z_pos = sliding_positions(d, args.hr_cube_size, args.step)
        y_pos = sliding_positions(h, args.hr_cube_size, args.step)
        x_pos = sliding_positions(w, args.hr_cube_size, args.step)
        if not z_pos or not y_pos or not x_pos:
            print(f'[skip] {tif_path.name}: shape={volume.shape}, smaller than cube_size={args.hr_cube_size}')
            continue

        volume_cache[tif_path.stem] = volume
        for z in z_pos:
            for y in y_pos:
                for x in x_pos:
                    all_indices.append((tif_path.stem, z, y, x))

    if not all_indices:
        raise ValueError('No valid cubes were generated from input tif files.')

    random.seed(args.seed)
    random.shuffle(all_indices)

    if args.max_cubes > 0:
        all_indices = all_indices[:args.max_cubes]

    total = len(all_indices)
    val_count = int(math.floor(total * args.val_ratio))
    if args.val_ratio > 0 and val_count == 0 and total > 1:
        val_count = 1

    val_indices = set(all_indices[:val_count])
    train_count = 0
    val_written = 0

    for stem, z, y, x in all_indices:
        hr_vol = volume_cache[stem]
        hr_cube = hr_vol[z:z + args.hr_cube_size, y:y + args.hr_cube_size, x:x + args.hr_cube_size, :]
        lr_cube = downsample_mean(hr_cube, args.scale)

        cube_stem = f'{stem}_z{z:04d}_y{y:04d}_x{x:04d}'
        split = 'validation' if (stem, z, y, x) in val_indices else 'train'
        save_cube(
            hr_cube,
            args.high_out_dir / split / cube_stem,
            output_format=args.output_format,
            save_dtype=args.save_dtype)
        save_cube(
            lr_cube,
            args.low_out_dir / split / cube_stem,
            output_format=args.output_format,
            save_dtype=args.save_dtype)

        if split == 'validation':
            val_written += 1
        else:
            train_count += 1

    print('Done.')
    print(f'Input tif files: {len(tif_files)}')
    print(f'Total cubes: {total}')
    print(f'Train cubes: {train_count}')
    print(f'Validation cubes: {val_written}')
    print(f'HR cube size: {args.hr_cube_size}^3, LR cube size: {args.hr_cube_size // args.scale}^3, scale: x{args.scale}')
    print(f'Output format: {args.output_format}, save dtype: {args.save_dtype}')
    print(f'Saved to: {args.high_out_dir} and {args.low_out_dir}')


if __name__ == '__main__':
    main()
