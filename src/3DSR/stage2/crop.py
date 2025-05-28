import tifffile

filename = 'gt_hr.tif'

data = tifffile.imread(filename)

print("original shape:", data.shape)  # (2048, 1504, 1504)

# remove the last 8 slices along the first dimension
new_data = data[:-8, :, :]

print("new shape:", new_data.shape)  # (2040, 1504, 1504)

# 保存
tifffile.imwrite('gt_hr_2040.tif', new_data)
print("save as gt_hr_2040.tif")