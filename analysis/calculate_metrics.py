import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy.stats import kurtosis, skew

# ===== 手写 NIQE 和 PIQE =====
def fake_niqe(img):
    """极简无参考质量分: 统计特征偏离自然图像经验值的绝对和"""
    img = img.astype(np.float64) / 255.
    mu = np.mean(img)
    sigma = np.std(img)
    skw = skew(img.ravel())
    krt = kurtosis(img.ravel())
    # 经验：自然图像均值0.5，方差0.15，偏度0，峰度3
    score = abs(mu - 0.5) + abs(sigma - 0.15) + abs(skw - 0) + abs(krt - 3)
    return score

def fake_piqe(img, block_size=16):
    """块方差过低/过高为有缺陷块，PIQE约等于坏块比例×100"""
    img = img.astype(np.float64)
    h, w = img.shape
    blocks = []
    for i in range(0, h-block_size+1, block_size):
        for j in range(0, w-block_size+1, block_size):
            block = img[i:i+block_size, j:j+block_size]
            blocks.append(block)
    if not blocks:
        return 0.0
    blocks = np.array(blocks)
    variances = [np.var(b) for b in blocks]
    bad_blocks = [(v < 5) or (v > 500) for v in variances]
    return 100. * np.mean(bad_blocks)

# ====== 主流程 ======

# 图片路径
image_files = {
    'GTLR': 'gtlr.png',
    'Dual-2D-DS': 'dual2dds.png',
    '3D SR-DS': '3dsrds.png',
    'GTHR': 'gthr.png',
    'Dual-2D': 'dual2d.png',
    '3D SR': '3dsr.png'
}

groups = {
    'lowres': {
        'gt': 'GTLR',
        'methods': ['Dual-2D-DS', '3D SR-DS']
    },
    'highres': {
        'gt': 'GTHR',
        'methods': ['Dual-2D', '3D SR']
    }
}

def compute_metrics(pred, gt):
    pred_f = pred.astype(np.float32) / 255.
    gt_f = gt.astype(np.float32) / 255.
    ssim_score = ssim(gt_f, pred_f, data_range=1.0)
    psnr_score = psnr(gt_f, pred_f, data_range=1.0)
    niqe_score = fake_niqe(pred)
    piqe_score = fake_piqe(pred)
    return ssim_score, psnr_score, niqe_score, piqe_score

print(f"{'Method':<12} {'GT':<8} {'SSIM':>7} {'PSNR':>7} {'NIQE':>7} {'PIQE':>7}")

for group, info in groups.items():
    gt_img = np.array(Image.open(image_files[info['gt']]).convert('L'))
    # 先对GT自身
    ssim_score, psnr_score, niqe_score, piqe_score = compute_metrics(gt_img, gt_img)
    print(f"{info['gt']:<12} {info['gt']:<8} {ssim_score:7.4f} {psnr_score:7.2f} {niqe_score:7.2f} {piqe_score:7.2f}")
    # 再对各方法
    for method in info['methods']:
        pred_img = np.array(Image.open(image_files[method]).convert('L'))
        # 若分辨率不一致，自动resize到GT
        if pred_img.shape != gt_img.shape:
            pred_img = np.array(Image.fromarray(pred_img).resize(gt_img.shape[::-1], Image.BICUBIC))
        ssim_score, psnr_score, niqe_score, piqe_score = compute_metrics(pred_img, gt_img)
        print(f"{method:<12} {info['gt']:<8} {ssim_score:7.4f} {psnr_score:7.2f} {niqe_score:7.2f} {piqe_score:7.2f}")