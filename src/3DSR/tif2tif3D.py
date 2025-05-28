
from tqdm import tqdm
import os
import numpy as np
from skimage import io
import shutil

def main():
    # load low resolution 3D CT images
    LR = io.imread('../c2_lo_resized_512_376_376_8bits.tif')  # 形状: (Z, H, W)
    if not os.path.exists("Low"):
        os.mkdir("Low")
    
    # Split LR images into train and test sets
    num_images = LR.shape[0] - 2  # minus 2 to avoid index out of bounds when taking 3 slices
    num_validation_images = 10
    train_indices = list(range(num_images))
    validation_indices = np.random.choice(train_indices, size=num_validation_images, replace=False)
    train_indices = [idx for idx in train_indices if idx not in validation_indices]

    # create low resolution 3D images (output: 3×100×100)
    for i in tqdm(train_indices):
        # conturning slices: i, i+1, i+2
        lr_block = LR[i:i+3, :, :]  # (3, H, W)
        io.imsave(f"Low/train/{i}.png", lr_block)
    
    for i in tqdm(validation_indices):
        lr_block = LR[i:i+3, :, :]
        io.imsave(f"Low/validation/{i}.png", lr_block)

    print('Low resolution 3D images generated')

    # read high resolution 3D CT images
    HR = io.imread('../c2_hi_resized_1024_3072_8bits.tif')  # 形状: (Z, H, W)
    if not os.path.exists("High"):
        os.mkdir("High")
    
    # create high resolution 3D images
    # median slice for 3D super-resolution
    for i in tqdm(train_indices):
        hr_center_idx = (i + 1) * 4  # 4 times upsampling
        hr_slice = HR[hr_center_idx, :, :]  # single slice at the center of the block
        # add a new axis to make it 3D (1, H, W)
        hr_block = hr_slice[np.newaxis, :, :]
        io.imsave(f"High/train/{i}.png", hr_block)
    
    for i in tqdm(validation_indices):
        hr_center_idx = (i + 1) * 4
        hr_slice = HR[hr_center_idx, :, :]
        hr_block = hr_slice[np.newaxis, :, :]
        io.imsave(f"High/validation/{i}.png", hr_block)

    print('High resolution images generated (center slice)')

if __name__ == '__main__':
    # Create train/validation directories
    os.makedirs('Low/train', exist_ok=True)
    os.makedirs('Low/validation', exist_ok=True)
    os.makedirs('High/train', exist_ok=True)
    os.makedirs('High/validation', exist_ok=True)

    main()


