import tifffile
import numpy as np
from PIL import Image
import os
import random
from tqdm import tqdm

high_path = 'gt_hr.tif'
low_path = 'volume-bicubic.tif'
crop_size = 240
total_pairs = 10000
train_ratio = 0.999

high_save_dir = 'High_sub'
low_save_dir = 'Low_sub'

for sub in ['train', 'validation']:
    os.makedirs(os.path.join(high_save_dir, sub), exist_ok=True)
    os.makedirs(os.path.join(low_save_dir, sub), exist_ok=True)

high = tifffile.imread(high_path)
low = tifffile.imread(low_path)

print("HR shape:", high.shape)
print("LR shape:", low.shape)

Z, Y, X = high.shape

# make sure the crop size is valid
valid_x = [i for i in range(X) if Y >= crop_size and Z >= crop_size]
num_x = len(valid_x)
pairs_per_x = total_pairs // num_x + 1

all_pairs = []
for x in valid_x:
    for _ in range(pairs_per_x):
        # randomly choose y_start and z_start
        y_start = random.randint(0, Y - crop_size)
        z_start = random.randint(0, Z - crop_size)
        all_pairs.append((x, y_start, z_start))
        if len(all_pairs) >= total_pairs:
            break
    if len(all_pairs) >= total_pairs:
        break

random.shuffle(all_pairs)
split_idx = int(len(all_pairs) * train_ratio)
train_pairs = all_pairs[:split_idx]
val_pairs = all_pairs[split_idx:]

def save_crop(data, x, y_start, z_start, crop_size, save_dir, idx):
    # crop along the z-axis
    crop = data[z_start:z_start+crop_size, y_start:y_start+crop_size, x]
    # standardize the crop to 0-255 range
    crop_norm = ((crop - crop.min()) / (crop.ptp() + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(crop_norm)
    img.save(os.path.join(save_dir, f'{idx:05d}.png'))

print("save training data...")
for idx, (x, y, z) in tqdm(enumerate(train_pairs), total=len(train_pairs)):
    save_crop(high, x, y, z, crop_size, os.path.join(high_save_dir, 'train'), idx)
    save_crop(low, x, y, z, crop_size, os.path.join(low_save_dir, 'train'), idx)

print("save test data...")
for idx, (x, y, z) in tqdm(enumerate(val_pairs), total=len(val_pairs)):
    save_crop(high, x, y, z, crop_size, os.path.join(high_save_dir, 'validation'), idx)
    save_crop(low, x, y, z, crop_size, os.path.join(low_save_dir, 'validation'), idx)
