# 数据预处理说明（3D Cubic SR, x2）

本项目当前第一阶段目标是训练“真 3D 超分”：

- 高分辨率块：`128^3`
- 低分辨率块：`64^3`
- 放大倍率：`x2`
- 数据组织：仍使用 `High_sub` / `Low_sub`，但文件是 3D 小块（`.npy`）

## 1. REV 约束

我们在当前数据上采用 REV（Representative Elementary Volume）下限：

- `HR cube size >= 128`
- 小于 128 时，很容易丢失大尺度轮廓与细节对应关系，训练会明显变难。

脚本里已经强制此约束：`--hr-cube-size < 128` 会直接报错。

## 2. 输入与输出目录

### 输入

把原始高分辨率 3D 体数据（`.tif/.tiff`）放在 `data/` 下，例如：

- `data/c2_hi_resized_1024_3072_8bits.tif`

### 输出

```text
data/mengyang/CARBONATES/3DSR/
  High_sub/
    train/
    validation/
  Low_sub/
    train/
    validation/
```

每个 HR/LR 对应小块同名保存：

- `High_sub/.../xxx_z0000_y0000_x0000.npy`  -> `(128,128,128,C)`
- `Low_sub/.../xxx_z0000_y0000_x0000.npy`   -> `(64,64,64,C)`

## 3. 一条命令完成预处理

脚本：`tools/prepare_3d_cubes_from_tif.py`

```bash
python tools/prepare_3d_cubes_from_tif.py \
  --input-dir data \
  --high-out-dir data/mengyang/CARBONATES/3DSR/High_sub \
  --low-out-dir data/mengyang/CARBONATES/3DSR/Low_sub \
  --hr-cube-size 128 \
  --step 128 \
  --val-ratio 0.1 \
  --overwrite


python tools/prepare_3d_cubes_from_tif.py \
  --input-dir data \
  --high-out-dir data/mengyang/CARBONATES/3DSR/High_sub \
  --low-out-dir data/mengyang/CARBONATES/3DSR/Low_sub_x4 \
  --hr-cube-size 128 \
  --scale 4 \
  --step 128 \
  --val-ratio 0.1 \
  --seed 0 \
  --output-format npy \
  --save-dtype uint8 \
  --overwrite
```

## 4. 关键参数说明

- `--input-dir`：扫描该目录下所有 `.tif/.tiff` 作为 HR 体数据输入。
- `--hr-cube-size`：HR 切块边长，默认 128（且必须 >=128）。
- `--step`：HR 空间滑窗步长，默认 128。
- `--val-ratio`：验证集比例（0~1），默认 0.1。
- `--max-cubes`：最多导出多少个 cube，0 表示不限（用于调试很有用）。
- `--overwrite`：覆盖模式，先清理已有 `.npy` 再重建。
- `--output-format`：输出格式，`npy` / `tif` / `both`。
- `--save-dtype`：保存数据类型，`uint8`（推荐，省空间）或 `float32`。

## 5. 代码逻辑（脚本内部流程）

`tools/prepare_3d_cubes_from_tif.py` 逻辑如下：

1. 扫描输入目录下的 `*.tif/*.tiff`。
2. 读取体数据并统一为 `float32`，范围归一化到 `[0,1]`（整数类型会自动除以 dtype 最大值）。
3. 统一布局为 `DHWC`：
   - 3D 输入 `DHW` 自动扩成 `DHWC`（单通道）。
   - 4D 输入自动判断通道轴（简单规则）。
4. 在 HR 体上按 `hr-cube-size` 和 `step` 生成 `z,y,x` 滑窗坐标。
5. 汇总所有 cube 索引，随机打乱后按 `val-ratio` 划分 train/validation。
6. 裁剪 HR cube：`(128,128,128,C)`。
7. 对 HR cube 做 `x2` 三维均值下采样，得到 LR cube：`(64,64,64,C)`。
   - 实现方式：`reshape(...,2,...,2,...,2,...)` 后在三个 `2` 维度做 mean。
8. HR/LR 分别保存到 `High_sub/<split>` 与 `Low_sub/<split>`，文件名保持一致。

## 6. 输出数据检查建议

预处理后建议抽样检查：

- HR 与 LR 文件名是否一一对应；
- HR shape 是否为 `(128,128,128,C)`；
- LR shape 是否为 `(64,64,64,C)`；
- 数值范围是否在 `[0,1]`；
- `train/validation` 数量是否符合 `val-ratio` 预期。

补充：

- `.npy` 是二进制数组文件，VSCode 文本模式打开可能显示空白或不可读，这是正常现象。
- 如果想“肉眼可见”浏览，可以用 `--output-format tif` 或 `--output-format both` 同时导出 `.tif`。

## 7. 兼容说明

- 当前这套预处理是“从 HR tif 自动生成 LR”的最简起步流程，适合先跑通 3D x2 SR。
- 后续如果改为“真实成像配对 HR/LR”，可保留相同目录结构，只替换数据生成方式即可。

## 8. 命令速查

### 8.1 正式生成（全量）

```bash
python tools/prepare_3d_cubes_from_tif.py \
  --input-dir data \
  --high-out-dir data/mengyang/CARBONATES/3DSR/High_sub \
  --low-out-dir data/mengyang/CARBONATES/3DSR/Low_sub \
  --hr-cube-size 128 \
  --step 128 \
  --val-ratio 0.1 \
  --output-format npy \
  --save-dtype uint8 \
  --overwrite
```

说明：`npy + uint8` 是当前推荐组合，训练读取快、空间占用小。

### 8.1b 可视化友好（同时导出 tif）

```bash
python tools/prepare_3d_cubes_from_tif.py \
  --input-dir data \
  --high-out-dir data/mengyang/CARBONATES/3DSR/High_sub \
  --low-out-dir data/mengyang/CARBONATES/3DSR/Low_sub \
  --hr-cube-size 128 \
  --step 128 \
  --val-ratio 0.1 \
  --output-format both \
  --save-dtype uint8 \
  --overwrite
```

### 8.2 调试生成（只切少量块）

```bash
python tools/prepare_3d_cubes_from_tif.py \
  --input-dir data \
  --high-out-dir data/mengyang/CARBONATES/3DSR/High_sub \
  --low-out-dir data/mengyang/CARBONATES/3DSR/Low_sub \
  --hr-cube-size 128 \
  --step 128 \
  --val-ratio 0.1 \
  --max-cubes 20 \
  --overwrite
```

### 8.3 统计 train/validation 文件数

```bash
find data/mengyang/CARBONATES/3DSR/High_sub/train -maxdepth 1 -type f -name '*.npy' | wc -l
find data/mengyang/CARBONATES/3DSR/High_sub/validation -maxdepth 1 -type f -name '*.npy' | wc -l
find data/mengyang/CARBONATES/3DSR/Low_sub/train -maxdepth 1 -type f -name '*.npy' | wc -l
find data/mengyang/CARBONATES/3DSR/Low_sub/validation -maxdepth 1 -type f -name '*.npy' | wc -l
```

### 8.4 抽样检查 cube 的 shape

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
h = next(Path('data/mengyang/CARBONATES/3DSR/High_sub/train').glob('*.npy'))
l = Path('data/mengyang/CARBONATES/3DSR/Low_sub/train') / h.name
print('HR', np.load(h).shape)
print('LR', np.load(l).shape)
PY
```
