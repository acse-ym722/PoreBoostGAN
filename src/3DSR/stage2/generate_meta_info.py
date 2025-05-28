import os
from os import path as osp
from PIL import Image

from basicsr.utils import scandir


def generate_meta_info_carbonates(gt_folder, meta_info_txt):
    """Generate meta info for carbonates dataset.
    """

    # Create the 'meta_info' folder if it doesn't exist
    meta_info_folder = './meta_info'
    if not os.path.exists(meta_info_folder):
        os.makedirs(meta_info_folder)
    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


if __name__ == '__main__':
    gt_folder = './High_sub/train'
    meta_info_txt = './meta_info/meta_info_high_carbon.txt'
    generate_meta_info_carbonates(gt_folder, meta_info_txt)


    gt_folder = './High_sub/validation'
    meta_info_txt = './meta_info/meta_info_val_carbon.txt'
    generate_meta_info_carbonates(gt_folder, meta_info_txt)
