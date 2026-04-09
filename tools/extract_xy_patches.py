import argparse
from pathlib import Path

import cv2


def collect_pairs(hr_dir: Path, lr_dir: Path):
    hr_files = {path.stem: path for path in hr_dir.iterdir() if path.is_file()}
    lr_files = {path.stem: path for path in lr_dir.iterdir() if path.is_file()}
    common = sorted(hr_files.keys() & lr_files.keys())
    return [(hr_files[name], lr_files[name]) for name in common]


def sliding_positions(length: int, crop_size: int, step: int):
    if length < crop_size:
        return []
    positions = list(range(0, length - crop_size + 1, step))
    if positions[-1] != length - crop_size:
        positions.append(length - crop_size)
    return positions


def extract_pair(hr_path: Path, lr_path: Path, hr_out_dir: Path, lr_out_dir: Path, scale: int, hr_crop_size: int, step: int):
    hr = cv2.imread(str(hr_path), cv2.IMREAD_GRAYSCALE)
    lr = cv2.imread(str(lr_path), cv2.IMREAD_GRAYSCALE)
    if hr is None or lr is None:
        raise ValueError(f'Failed to read pair: {hr_path}, {lr_path}')

    lr_crop_size = hr_crop_size // scale
    lr_step = step // scale

    if hr.shape[0] != lr.shape[0] * scale or hr.shape[1] != lr.shape[1] * scale:
        raise ValueError(f'Size mismatch for {hr_path.name}: HR {hr.shape}, LR {lr.shape}, scale={scale}')

    y_positions = sliding_positions(hr.shape[0], hr_crop_size, step)
    x_positions = sliding_positions(hr.shape[1], hr_crop_size, step)
    if not y_positions or not x_positions:
        return 0

    count = 0
    for y in y_positions:
        for x in x_positions:
            count += 1
            hr_patch = hr[y:y + hr_crop_size, x:x + hr_crop_size]
            lr_patch = lr[y // scale:y // scale + lr_crop_size, x // scale:x // scale + lr_crop_size]
            hr_out = hr_out_dir / f'{hr_path.stem}_s{count:03d}.png'
            lr_out = lr_out_dir / f'{lr_path.stem}_s{count:03d}.png'
            cv2.imwrite(str(hr_out), hr_patch)
            cv2.imwrite(str(lr_out), lr_patch)
    return count


def parse_args():
    parser = argparse.ArgumentParser(description='Extract paired XY training patches from aligned digital-rock slices.')
    parser.add_argument('--hr-dir', required=True, type=Path)
    parser.add_argument('--lr-dir', required=True, type=Path)
    parser.add_argument('--hr-out-dir', required=True, type=Path)
    parser.add_argument('--lr-out-dir', required=True, type=Path)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--hr-crop-size', type=int, default=384)
    parser.add_argument('--step', type=int, default=320)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hr_crop_size % args.scale != 0 or args.step % args.scale != 0:
        raise ValueError('hr-crop-size and step must be divisible by scale.')

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
        raise ValueError('No aligned HR/LR pairs were found.')

    total = 0
    for hr_path, lr_path in pairs:
        total += extract_pair(
            hr_path=hr_path,
            lr_path=lr_path,
            hr_out_dir=args.hr_out_dir,
            lr_out_dir=args.lr_out_dir,
            scale=args.scale,
            hr_crop_size=args.hr_crop_size,
            step=args.step,
        )
    print(f'Extracted {total} paired patches from {len(pairs)} image pairs.')


if __name__ == '__main__':
    main()
