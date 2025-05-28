import tifffile
import numpy as np
from PIL import Image
import os

# 读取3D TIFF文件
filename = 'dual2D-ds.tif'
data = tifffile.imread(filename)

# 打印xyz轴尺寸信息
print("Shape (Z, Y, X):", data.shape)

# 固定X轴，取出YZ平面
x = 25  # 你可以根据实际情况调整
yz_plane = data[:, :, x]  # 所有Z和Y，对应固定X

# 切割120*120区域
y_start = 0
z_start = 0
crop_size = 120
crop_yz = yz_plane[z_start:z_start+crop_size, y_start:y_start+crop_size]

print("Cropped YZ plane shape:", crop_yz.shape)

# 归一化到uint8范围，便于保存为图片
if crop_yz.dtype != np.uint8:
    crop_yz_norm = (255 * (crop_yz - crop_yz.min()) / (crop_yz.ptp() + 1e-8)).astype(np.uint8)
else:
    crop_yz_norm = crop_yz

# 生成保存文件名
basename = os.path.splitext(os.path.basename(filename))[0]
save_name = f"{basename}_YZ_x{x}_crop.png"

# 保存图片
img = Image.fromarray(crop_yz_norm)
img.save(save_name)
print(f"Saved cropped image as {save_name}")