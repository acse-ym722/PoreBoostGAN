import tifffile
import numpy as np
from PIL import Image
import os

low_path = 'volume-bicubic.tif'
save_dir = 'slice_images'
os.makedirs(save_dir, exist_ok=True)


low = tifffile.imread(low_path)
Z, Y, X = low.shape
print("shape:", low.shape)

for x in range(X):
    yz_slice = low[:, :, x]
    # standardize the crop to 0-255 range
    crop_norm = ((yz_slice - yz_slice.min()) / (yz_slice.ptp() + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(crop_norm)
    img.save(os.path.join(save_dir, f'{x}.png'))

print(f"save {X} images to {save_dir}")