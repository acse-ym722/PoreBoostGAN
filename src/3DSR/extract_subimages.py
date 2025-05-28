import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
from skimage import io
from pathlib import Path

try:
    from basicsr.utils import scandir
except ImportError:
    def scandir(dir_path, full_path=False):
        """Simple scandir replacement if basicsr is not available."""
        files = []
        for f in os.listdir(dir_path):
            if full_path:
                files.append(os.path.join(dir_path, f))
            else:
                files.append(f)
        return files


def main():
    """3D CT XY-plane paired cropping for training and validation datasets.
    This script processes paired high-resolution (HR) and low-resolution (LR) images,
    ensuring that sub-images are cropped correctly and saved in corresponding directories.
    It creates subdirectories for training and validation sets, and handles the cropping
    of images based on specified crop sizes and overlap.
    The script also checks for paired files and ensures that the HR images are 4 times larger
    than the LR images in both dimensions.
    The output is organized into 'High_sub' and 'Low_sub' directories for both training and validation sets.
    The script uses multiprocessing to speed up the processing of images.
    It also includes checks for file existence and compatibility, ensuring that the cropping
    process does not fail due to mismatched dimensions or missing files.
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    for folder in ['High_sub/train', 'High_sub/validation', 'Low_sub/train', 'Low_sub/validation']:
        os.makedirs(folder, exist_ok=True)

    print("Processing training pairs...")
    process_paired_images('High/train', 'Low/train', 'High_sub/train', 'Low_sub/train', opt)

    print("Processing validation pairs...")
    process_paired_images('High/validation', 'Low/validation', 'High_sub/validation', 'Low_sub/validation', opt)


def process_paired_images(hr_folder, lr_folder, hr_save_folder, lr_save_folder, opt):
    """handles the processing of paired HR and LR images,"""
    
    if not (os.path.exists(hr_folder) and os.path.exists(lr_folder)):
        print(f"Warning: Input folders {hr_folder} or {lr_folder} do not exist. Skipping...")
        return
    
    for folder in [hr_save_folder, lr_save_folder]:
        if os.path.exists(folder):
            print(f'Warning: Folder {folder} already exists. Removing and recreating...')
            import shutil
            shutil.rmtree(folder)
        os.makedirs(folder)
        print(f'Created folder {folder}')


    hr_files = [f for f in os.listdir(hr_folder) if f.lower().endswith(('.tif', '.tiff'))]
    lr_files = [f for f in os.listdir(lr_folder) if f.lower().endswith(('.tif', '.tiff'))]
    
    paired_files = []
    for hr_file in hr_files:
        hr_name = os.path.splitext(hr_file)[0]
        matching_lr = [lr for lr in lr_files if os.path.splitext(lr)[0] == hr_name]
        if matching_lr:
            paired_files.append((
                os.path.join(hr_folder, hr_file),
                os.path.join(lr_folder, matching_lr[0])
            ))
        else:
            print(f"Warning: No matching LR file found for HR file {hr_file}")
    
    print(f"Found {len(paired_files)} paired images")
    
    if len(paired_files) == 0:
        print("No paired files found! Checking if file naming conventions differ...")

        if len(hr_files) == len(lr_files):
            print(f"Found equal number of files ({len(hr_files)}), pairing by index")
            hr_files.sort()
            lr_files.sort()
            paired_files = [
                (os.path.join(hr_folder, hr), os.path.join(lr_folder, lr))
                for hr, lr in zip(hr_files, lr_files)
            ]
    
    if len(paired_files) == 0:
        print("Error: Could not find paired files!")
        return

    # crop parameters
    hr_crop_size = 384
    lr_crop_size = 96
    hr_step = 320  # 80% overlap
    lr_step = 80   # 80% overlap
    
    # make sure the crop sizes are valid
    scale_factor = hr_crop_size // lr_crop_size  # equal to 4 for 3D CT images
    
    # handle multiprocessing
    for hr_path, lr_path in tqdm(paired_files, desc="Processing pairs"):
        hr_name = os.path.splitext(os.path.basename(hr_path))[0]
        lr_name = os.path.splitext(os.path.basename(lr_path))[0]
        
        hr_img = io.imread(hr_path)
        lr_img = io.imread(lr_path)
        
        if len(hr_img.shape) == 3:
            if hr_img.shape[2] == 1:  # (H, W, 1)
                hr_img = np.transpose(hr_img, (2, 0, 1))  # -> (1, H, W)
            elif hr_img.shape[0] <= hr_img.shape[1] and hr_img.shape[0] <= hr_img.shape[2]:
                # (Z, H, W)
                pass
            else:
                # (H, W, Z)
                hr_img = np.transpose(hr_img, (2, 0, 1))
        elif len(hr_img.shape) == 2:
            hr_img = hr_img[np.newaxis, :, :]  # single channel image, add Z dimension
        
        if len(lr_img.shape) == 3:
            if lr_img.shape[2] == 3:  # (H, W, 3)
                lr_img = np.transpose(lr_img, (2, 0, 1))  # -> (3, H, W)
            elif lr_img.shape[0] <= lr_img.shape[1] and lr_img.shape[0] <= lr_img.shape[2]:
                # (Z, H, W)
                pass
            else:
                # (H, W, Z)
                lr_img = np.transpose(lr_img, (2, 0, 1))
        elif len(lr_img.shape) == 2:
            lr_img = lr_img[np.newaxis, :, :]  # single channel image, add Z dimension
        
        hr_z, hr_h, hr_w = hr_img.shape
        lr_z, lr_h, lr_w = lr_img.shape
        
        if not (hr_h == lr_h * scale_factor and hr_w == lr_w * scale_factor):
            print(f"Warning: HR size ({hr_h}x{hr_w}) is not {scale_factor}x of LR size ({lr_h}x{lr_w})")
            # match sizes if possible
            if hr_h > lr_h * scale_factor and hr_w > lr_w * scale_factor:
                hr_img = hr_img[:, :lr_h * scale_factor, :lr_w * scale_factor]
                hr_h, hr_w = lr_h * scale_factor, lr_w * scale_factor
                print(f"Adjusted HR size to {hr_h}x{hr_w}")
        
        # validate dimensions z
        if hr_z != 1:
            print(f"Warning: HR image {hr_name} has {hr_z} slices, expected 1")
        if lr_z != 3:
            print(f"Warning: LR image {lr_name} has {lr_z} slices, expected 3")
        
        if hr_h < hr_crop_size or hr_w < hr_crop_size:
            print(f"Warning: HR image {hr_name} ({hr_h}x{hr_w}) is smaller than crop_size {hr_crop_size}")
            continue
        
        if lr_h < lr_crop_size or lr_w < lr_crop_size:
            print(f"Warning: LR image {lr_name} ({lr_h}x{lr_w}) is smaller than crop_size {lr_crop_size}")
            continue
        
        # calculate cropping spaces
        h_space = np.arange(0, hr_h - hr_crop_size + 1, hr_step)
        if h_space.size > 0 and hr_h - (h_space[-1] + hr_crop_size) > 0:
            h_space = np.append(h_space, hr_h - hr_crop_size)
        elif h_space.size == 0:
            h_space = np.array([0])
        
        w_space = np.arange(0, hr_w - hr_crop_size + 1, hr_step)
        if w_space.size > 0 and hr_w - (w_space[-1] + hr_crop_size) > 0:
            w_space = np.append(w_space, hr_w - hr_crop_size)
        elif w_space.size == 0:
            w_space = np.array([0])
        
        # save paired patches
        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                
                # calculate corresponding HR coordinates
                hr_x, hr_y = int(x), int(y)
                lr_x, lr_y = hr_x // scale_factor, hr_y // scale_factor
                
                # crop HR and LR images
                hr_crop = hr_img[:, hr_x:hr_x + hr_crop_size, hr_y:hr_y + hr_crop_size]
                hr_crop = np.ascontiguousarray(hr_crop)
                
                lr_crop = lr_img[:, lr_x:lr_x + lr_crop_size, lr_y:lr_y + lr_crop_size]
                lr_crop = np.ascontiguousarray(lr_crop)
                
                # validate crop shapes
                if hr_crop.shape != (hr_z, hr_crop_size, hr_crop_size):
                    print(f"Warning: HR crop shape {hr_crop.shape} != expected {(hr_z, hr_crop_size, hr_crop_size)}")
                    continue
                    
                if lr_crop.shape != (lr_z, lr_crop_size, lr_crop_size):
                    print(f"Warning: LR crop shape {lr_crop.shape} != expected {(lr_z, lr_crop_size, lr_crop_size)}")
                    continue
                
                # save cropped images
                hr_output_path = osp.join(hr_save_folder, f'{hr_name}_s{index:03d}.png')
                lr_output_path = osp.join(lr_save_folder, f'{lr_name}_s{index:03d}.png')
                
                io.imsave(hr_output_path, hr_crop)
                io.imsave(lr_output_path, lr_crop)
        
        print(f"Created {index} paired patches from {hr_name} and {lr_name}")


def count_files_pathlib(folder_path):
    path = Path(folder_path)
    file_count = sum(1 for item in path.iterdir() if item.is_file())
    return file_count


if __name__ == '__main__':
    main()
    print("3D CT XY-plane paired cropping completed!")
    
    for folder_path in ['High_sub/train', 'High_sub/validation', 'Low_sub/train', 'Low_sub/validation']:
        file_count = count_files_pathlib(folder_path)
        print(f"Number of files in {folder_path}: {file_count}")
    
    hr_train_files = set(f.stem.rsplit('_s', 1)[0] for f in Path('High_sub/train').glob('*.tif'))
    lr_train_files = set(f.stem.rsplit('_s', 1)[0] for f in Path('Low_sub/train').glob('*.tif'))
    print(f"Train pairs check - Common base names: {len(hr_train_files.intersection(lr_train_files))}")
    
    hr_val_files = set(f.stem.rsplit('_s', 1)[0] for f in Path('High_sub/validation').glob('*.tif'))
    lr_val_files = set(f.stem.rsplit('_s', 1)[0] for f in Path('Low_sub/validation').glob('*.tif'))
    print(f"Validation pairs check - Common base names: {len(hr_val_files.intersection(lr_val_files))}")