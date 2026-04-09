import argparse
from pathlib import Path

from PIL import Image


def build_meta_info(gt_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in gt_dir.iterdir() if path.is_file())

    with output.open('w') as handle:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                width, height = image.size
                if image.mode == 'L':
                    channels = 1
                elif image.mode == 'RGB':
                    channels = 3
                else:
                    raise ValueError(f'Unsupported image mode: {image.mode} ({image_path})')
            handle.write(f'{image_path.name} ({height},{width},{channels})\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate meta_info text for paired digital-rock patches.')
    parser.add_argument('--gt-dir', required=True, type=Path, help='Directory containing GT patch images.')
    parser.add_argument('--output', required=True, type=Path, help='Output meta_info txt path.')
    return parser.parse_args()


def main():
    args = parse_args()
    build_meta_info(args.gt_dir, args.output)


if __name__ == '__main__':
    main()
