import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Downsample a folder of grayscale digital-rock slices.')
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--scale', required=True, type=float, help='Downsampling factor. Example: 4 means 1/4 size.')
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in args.input_dir.iterdir() if path.is_file())
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        target_w = max(1, int(round(image.shape[1] / args.scale)))
        target_h = max(1, int(round(image.shape[0] / args.scale)))
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.output_dir / image_path.name), resized)


if __name__ == '__main__':
    main()
