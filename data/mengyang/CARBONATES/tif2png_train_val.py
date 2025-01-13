from tqdm import tqdm
import os
import numpy as np
from skimage import io
import shutil

def main():
    LR = io.imread('c2_lo_resized_256_768_8bits.tif')
    if not os.path.exists("Low"):
        os.mkdir("Low")
    
    # Split LR images into train and test sets
    num_images = LR.shape[0]
    num_validation_images = 10
    train_indices = list(range(num_images))
    validation_indices = np.random.choice(train_indices, size=num_validation_images, replace=False)
    train_indices = [idx for idx in train_indices if idx not in validation_indices]

    for i in tqdm(train_indices):
        io.imsave(f"Low/train/{i}.png", LR[i, :, :])
    
    for i in tqdm(validation_indices):
        io.imsave(f"Low/validation/{i}.png", LR[i, :, :])

    print('Low resolution images generated')

    HR = io.imread('c2_hi_resized_1024_3072_8bits.tif')
    if not os.path.exists("High"):
        os.mkdir("High")
    
    # Split HR images into train and test sets using the same indices
    for i in tqdm(train_indices):
        io.imsave(f"High/train/{i}.png", HR[i * 4 + 1, :, :])
    
    for i in tqdm(validation_indices):
        io.imsave(f"High/validation/{i}.png", HR[i * 4 + 1, :, :])

    print('High resolution images generated')
    
    for i in tqdm(range(num_images)):
        if i not in train_indices and i not in validation_indices:
            shutil.copyfile(f"Low/{i}.png", f"Low/test/{i}.png")
            shutil.copyfile(f"High/{i}.png", f"High/test/{i}.png")

    print('Test images generated')

if __name__ == '__main__':
    # Create train/validation/test directories for both low and high-resolution images
    os.makedirs('Low/train', exist_ok=True)
    os.makedirs('Low/validation', exist_ok=True)
    os.makedirs('High/train', exist_ok=True)
    os.makedirs('High/validation', exist_ok=True)

    main()
