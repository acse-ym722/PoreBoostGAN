import argparse
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None


def _load_shape(path: Path):
    suffix = path.suffix.lower()
    if suffix == '.npy':
        array = np.load(path, mmap_mode='r')
        shape = tuple(array.shape)
    elif suffix == '.npz':
        with np.load(path) as npz_data:
            if not npz_data.files:
                raise ValueError(f'No arrays in {path}.')
            shape = tuple(npz_data[npz_data.files[0]].shape)
    elif suffix in ('.tif', '.tiff'):
        if tifffile is None:
            raise ImportError('tifffile is required for .tif/.tiff files.')
        shape = tuple(tifffile.imread(path).shape)
    else:
        raise ValueError(f'Unsupported volume extension: {suffix} ({path})')
    return shape


def build_meta_info(gt_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    valid_suffixes = {'.npy', '.npz', '.tif', '.tiff'}
    volume_paths = sorted(
        path for path in gt_dir.iterdir()
        if path.is_file() and (not path.name.startswith('.')) and path.suffix.lower() in valid_suffixes)

    with output.open('w') as handle:
        for volume_path in volume_paths:
            shape = _load_shape(volume_path)
            shape_str = ','.join(str(dim) for dim in shape)
            handle.write(f'{volume_path.name} ({shape_str})\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate meta_info text for 3D volume patches.')
    parser.add_argument('--gt-dir', required=True, type=Path, help='Directory containing GT volume files.')
    parser.add_argument('--output', required=True, type=Path, help='Output meta_info txt path.')
    return parser.parse_args()


def main():
    args = parse_args()
    build_meta_info(args.gt_dir, args.output)


if __name__ == '__main__':
    main()
