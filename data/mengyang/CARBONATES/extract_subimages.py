import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
from skimage import io

from basicsr.utils import scandir


def main():
    """3D CT超分辨率数据集子图切割工具。
    
    将3D CT图像切割成小块用于训练：
    - 输入：3×100×100 (连续3层低分辨率)
    - 输出：1×400×400 (中心层高分辨率)
    
    避免显存不足问题。
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    # HR images for training (1×400×400 - 目标输出)
    opt['input_folder'] = 'High/train'
    opt['save_folder'] = 'High_sub/train'
    opt['crop_size'] = 400     # XY平面的裁剪尺寸
    opt['step'] = 320          # XY平面的步长 (80%重叠)
    opt['thresh_size'] = 0
    opt['is_3d'] = True
    opt['is_hr'] = True
    extract_subimages_3d(opt)

    # HR images for validation
    opt['input_folder'] = 'High/validation'
    opt['save_folder'] = 'High_sub/validation'
    extract_subimages_3d(opt)

    # LR images for training (3×100×100 - 输入)
    opt['input_folder'] = 'Low/train'
    opt['save_folder'] = 'Low_sub/train'
    opt['crop_size'] = 100     # XY平面的裁剪尺寸
    opt['step'] = 80           # XY平面的步长 (80%重叠)
    opt['thresh_size'] = 0
    opt['is_hr'] = False
    extract_subimages_3d(opt)

    # LR images for validation
    opt['input_folder'] = 'Low/validation'
    opt['save_folder'] = 'Low_sub/validation'
    extract_subimages_3d(opt)


def extract_subimages_3d(opt):
    """切割3D图像为子图像块。

    Args:
        opt (dict): Configuration dict. It contains:
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        n_thread (int): Thread number.
        is_hr (bool): Whether processing high resolution images.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    # 获取所有.tif文件
    img_list = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    img_list = [osp.join(input_folder, f) for f in img_list]

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract 3D')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker_3d, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All 3D processes done.')


def worker_3d(path, opt):
    """3D图像处理工作函数。

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
        crop_size (int): Crop size for XY plane.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.
        save_folder (str): Path to save folder.
        is_hr (bool): Whether processing high resolution images.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    is_hr = opt['is_hr']
    
    img_name, extension = osp.splitext(osp.basename(path))

    # 读取3D图像
    img_3d = io.imread(path)  # 形状: (Z, H, W)
    
    if len(img_3d.shape) != 3:
        raise ValueError(f"Expected 3D image, got shape {img_3d.shape}")
    
    z_depth, h, w = img_3d.shape
    
    # 根据是否为高分辨率确定Z轴处理方式
    if is_hr:
        # 高分辨率：输出1×400×400，只取第一层（预处理时已选择中心层）
        if z_depth != 1:
            print(f"Warning: HR image has {z_depth} slices, expected 1. Taking first slice.")
            img_3d = img_3d[0:1, :, :]
        target_z_size = 1
    else:
        # 低分辨率：输入3×100×100，需要3层
        if z_depth < 3:
            raise ValueError(f"LR image needs at least 3 slices, got {z_depth}")
        target_z_size = 3

    # 计算XY平面的裁剪位置
    h_space = np.arange(0, h - crop_size + 1, step)
    if len(h_space) > 0 and h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    elif len(h_space) == 0:
        h_space = [0] if h >= crop_size else []
    
    w_space = np.arange(0, w - crop_size + 1, step)
    if len(w_space) > 0 and w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)
    elif len(w_space) == 0:
        w_space = [0] if w >= crop_size else []

    # 检查是否可以裁剪
    if len(h_space) == 0 or len(w_space) == 0:
        print(f"Warning: Image {img_name} too small for crop_size {crop_size}")
        return f'Skipped {img_name} (too small)'

    # Z轴位置
    if is_hr:
        z_positions = [0]  # 高分辨率只有一层
    else:
        # 低分辨率：如果有更多层，可以有多个起始位置
        z_positions = list(range(0, z_depth - target_z_size + 1))

    index = 0
    for z_start in z_positions:
        for x in h_space:
            for y in w_space:
                index += 1
                
                # 裁剪3D块
                cropped_3d = img_3d[z_start:z_start + target_z_size, 
                                   int(x):int(x) + crop_size, 
                                   int(y):int(y) + crop_size]
                
                # 确保数据连续性
                cropped_3d = np.ascontiguousarray(cropped_3d)
                
                # 保存为.tif文件
                output_path = osp.join(opt['save_folder'], f'{img_name}_s{index:03d}.tif')
                io.imsave(output_path, cropped_3d)

    process_info = f'Processing 3D {img_name} ({"HR" if is_hr else "LR"}, {index} patches) ...'
    return process_info


def extract_subimages_2d_backup(opt):
    """保留原始2D图像处理函数作为备份。"""
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker_2d_backup, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker_2d_backup(path, opt):
    """原始2D图像处理工作函数备份。"""
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()