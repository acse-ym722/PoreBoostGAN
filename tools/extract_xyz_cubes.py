import argparse
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None


def collect_pairs(hr_dir: Path, lr_dir: Path):
    valid_suffixes = {'.npy', '.npz', '.tif', '.tiff'}
    hr_files = {
        path.stem: path
        for path in hr_dir.iterdir()
        if path.is_file() and (not path.name.startswith('.')) and path.suffix.lower() in valid_suffixes
    }
    lr_files = {
        path.stem: path
        for path in lr_dir.iterdir()
        if path.is_file() and (not path.name.startswith('.')) and path.suffix.lower() in valid_suffixes
    }
    common = sorted(hr_files.keys() & lr_files.keys())
    return [(hr_files[name], lr_files[name]) for name in common]


def resolve_layout(volume: np.ndarray, data_format: str):
    if volume.ndim == 3:
        return volume[..., None]
    if volume.ndim != 4:
        raise ValueError(f'Expected 3D/4D volume, got shape={volume.shape}.')

    fmt = data_format.lower()
    if fmt == 'dhwc':
        return volume
    if fmt == 'cdhw':
        return np.transpose(volume, (1, 2, 3, 0))
    if fmt != 'auto':
        raise ValueError(f'Unsupported data_format={data_format}.')

    if volume.shape[-1] <= 8 and volume.shape[0] > 8:
        return volume
    if volume.shape[0] <= 8 and volume.shape[-1] > 8:
        return np.transpose(volume, (1, 2, 3, 0))
    return volume


def load_volume(path: Path, data_format: str):
    suffix = path.suffix.lower()
    if suffix == '.npy':
        volume = np.load(path)
    elif suffix == '.npz':
        with np.load(path) as npz_data:
            if not npz_data.files:
                raise ValueError(f'No arrays in {path}.')
            volume = npz_data[npz_data.files[0]]
    elif suffix in ('.tif', '.tiff'):
        if tifffile is None:
            raise ImportError('tifffile is required to read .tif/.tiff files.')
        volume = tifffile.imread(path)
    else:
        raise ValueError(f'Unsupported volume extension: {suffix} ({path}).')

    volume = np.asarray(volume)
    volume = resolve_layout(volume, data_format)
    if np.issubdtype(volume.dtype, np.integer):
        volume = volume.astype(np.float32) / float(np.iinfo(volume.dtype).max)
    else:
        volume = volume.astype(np.float32)
    return volume


def sliding_positions(length: int, cube_size: int, step: int):
    if length < cube_size:
        return []
    positions = list(range(0, length - cube_size + 1, step))
    if positions[-1] != length - cube_size:
        positions.append(length - cube_size)
    return positions


def extract_pair(hr_path: Path, lr_path: Path, hr_out_dir: Path, lr_out_dir: Path, scale: int, hr_cube_size: int,
                 step: int, data_format: str):
    hr = load_volume(hr_path, data_format=data_format)
    lr = load_volume(lr_path, data_format=data_format)

    if hr_cube_size % scale != 0 or step % scale != 0:
        raise ValueError('hr-cube-size and step must be divisible by scale.')

    lr_cube_size = hr_cube_size // scale
    lr_step = step // scale

    if hr.shape[3] != lr.shape[3]:
        raise ValueError(f'Channel mismatch for {hr_path.name}: HR C={hr.shape[3]}, LR C={lr.shape[3]}')
    if hr.shape[0] != lr.shape[0] * scale or hr.shape[1] != lr.shape[1] * scale or hr.shape[2] != lr.shape[2] * scale:
        raise ValueError(f'Shape mismatch for {hr_path.name}: HR={hr.shape}, LR={lr.shape}, scale={scale}')

    z_positions = sliding_positions(hr.shape[0], hr_cube_size, step)
    y_positions = sliding_positions(hr.shape[1], hr_cube_size, step)
    x_positions = sliding_positions(hr.shape[2], hr_cube_size, step)
    if not z_positions or not y_positions or not x_positions:
        return 0

    count = 0
    for z in z_positions:
        for y in y_positions:
            for x in x_positions:
                count += 1
                hr_cube = hr[z:z + hr_cube_size, y:y + hr_cube_size, x:x + hr_cube_size, :]
                lr_cube = lr[z // scale:z // scale + lr_cube_size, y // scale:y // scale + lr_cube_size,
                             x // scale:x // scale + lr_cube_size, :]

                suffix = f'_z{z:04d}_y{y:04d}_x{x:04d}'
                np.save(hr_out_dir / f'{hr_path.stem}{suffix}.npy', hr_cube.astype(np.float32))
                np.save(lr_out_dir / f'{lr_path.stem}{suffix}.npy', lr_cube.astype(np.float32))
    return count


def parse_args():
    parser = argparse.ArgumentParser(description='Extract paired cubic patches from aligned 3D HR/LR volumes.')
    parser.add_argument('--hr-dir', required=True, type=Path)
    parser.add_argument('--lr-dir', required=True, type=Path)
    parser.add_argument('--hr-out-dir', required=True, type=Path)
    parser.add_argument('--lr-out-dir', required=True, type=Path)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--hr-cube-size', type=int, default=64)
    parser.add_argument('--step', type=int, default=64)
    parser.add_argument('--data-format', type=str, default='auto', choices=['auto', 'dhwc', 'cdhw'])
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.overwrite:
        for folder in [args.hr_out_dir, args.lr_out_dir]:
            if folder.exists():
                for path in folder.iterdir():
                    if path.is_file():
                        path.unlink()
    args.hr_out_dir.mkdir(parents=True, exist_ok=True)
    args.lr_out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.hr_dir, args.lr_dir)
    if not pairs:
        raise ValueError('No aligned HR/LR volume pairs found.')

    total = 0
    for hr_path, lr_path in pairs:
        total += extract_pair(
            hr_path=hr_path,
            lr_path=lr_path,
            hr_out_dir=args.hr_out_dir,
            lr_out_dir=args.lr_out_dir,
            scale=args.scale,
            hr_cube_size=args.hr_cube_size,
            step=args.step,
            data_format=args.data_format,
        )
    print(f'Extracted {total} paired 3D cubes from {len(pairs)} volume pairs.')


if __name__ == '__main__':
    main()
