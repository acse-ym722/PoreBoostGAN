import os
import random

import numpy as np
import torch
from torch.utils import data as data

from poreboostgan.data.data_util import paired_paths_from_folder, paired_paths_from_meta_info_file
from poreboostgan.utils import scandir
from poreboostgan.utils.registry import DATASET_REGISTRY

try:
    import tifffile
except ImportError:
    tifffile = None


def _resolve_volume_layout(volume, data_format):
    if volume.ndim == 3:
        return volume[..., None]
    if volume.ndim != 4:
        raise ValueError(f'Expected 3D/4D volume, but got shape={volume.shape}.')

    fmt = data_format.lower()
    if fmt == 'dhwc':
        return volume
    if fmt == 'cdhw':
        return np.transpose(volume, (1, 2, 3, 0))
    if fmt != 'auto':
        raise ValueError(f'Unsupported data_format={data_format}.')

    # Auto-guess channel axis for 4D volumes.
    if volume.shape[-1] <= 8 and volume.shape[0] > 8:
        return volume
    if volume.shape[0] <= 8 and volume.shape[-1] > 8:
        return np.transpose(volume, (1, 2, 3, 0))
    # Ambiguous case: default to DHWC.
    return volume


def _load_volume(path, data_format='auto'):
    suffix = os.path.splitext(path)[1].lower()
    if suffix == '.npy':
        volume = np.load(path)
    elif suffix == '.npz':
        with np.load(path) as npz_data:
            if not npz_data.files:
                raise ValueError(f'No arrays in {path}.')
            volume = npz_data[npz_data.files[0]]
    elif suffix in ('.tif', '.tiff'):
        if tifffile is None:
            raise ImportError('tifffile is required for .tif/.tiff volume loading.')
        volume = tifffile.imread(path)
    else:
        raise ValueError(f'Unsupported volume extension: {suffix} ({path}).')

    volume = np.asarray(volume)
    volume = _resolve_volume_layout(volume, data_format)
    if np.issubdtype(volume.dtype, np.integer):
        volume = volume.astype(np.float32) / float(np.iinfo(volume.dtype).max)
    else:
        volume = volume.astype(np.float32)
    return volume


def _paired_random_crop_3d(img_gt, img_lq, gt_size, scale, gt_path):
    if gt_size % scale != 0:
        raise ValueError(f'gt_size ({gt_size}) must be divisible by scale ({scale}).')
    lq_size = gt_size // scale

    gt_d, gt_h, gt_w = img_gt.shape[:3]
    lq_d, lq_h, lq_w = img_lq.shape[:3]

    if gt_d != lq_d * scale or gt_h != lq_h * scale or gt_w != lq_w * scale:
        raise ValueError(f'Spatial mismatch for {gt_path}: GT={img_gt.shape}, LQ={img_lq.shape}, scale={scale}.')

    if gt_d < gt_size or gt_h < gt_size or gt_w < gt_size:
        raise ValueError(f'GT volume {img_gt.shape} is smaller than gt_size={gt_size}.')
    if lq_d < lq_size or lq_h < lq_size or lq_w < lq_size:
        raise ValueError(f'LQ volume {img_lq.shape} is smaller than lq_size={lq_size}.')

    lq_z = random.randint(0, lq_d - lq_size)
    lq_y = random.randint(0, lq_h - lq_size)
    lq_x = random.randint(0, lq_w - lq_size)

    gt_z, gt_y, gt_x = lq_z * scale, lq_y * scale, lq_x * scale
    cropped_lq = img_lq[lq_z:lq_z + lq_size, lq_y:lq_y + lq_size, lq_x:lq_x + lq_size, :]
    cropped_gt = img_gt[gt_z:gt_z + gt_size, gt_y:gt_y + gt_size, gt_x:gt_x + gt_size, :]
    return cropped_gt, cropped_lq


def _augment_3d(img_gt, img_lq, use_hflip, use_rot):
    if use_hflip:
        if random.random() < 0.5:
            img_gt = np.flip(img_gt, axis=0)
            img_lq = np.flip(img_lq, axis=0)
        if random.random() < 0.5:
            img_gt = np.flip(img_gt, axis=1)
            img_lq = np.flip(img_lq, axis=1)
        if random.random() < 0.5:
            img_gt = np.flip(img_gt, axis=2)
            img_lq = np.flip(img_lq, axis=2)

    if use_rot and random.random() < 0.5:
        perm = np.random.permutation(3).tolist()
        img_gt = np.transpose(img_gt, (perm[0], perm[1], perm[2], 3))
        img_lq = np.transpose(img_lq, (perm[0], perm[1], perm[2], 3))

    return np.ascontiguousarray(img_gt), np.ascontiguousarray(img_lq)


def _to_tensor(volume):
    # DHWC -> CDHW
    volume = np.ascontiguousarray(np.transpose(volume, (3, 0, 1, 2)))
    return torch.from_numpy(volume).float()


def _normalize_tensor(volume_tensor, mean, std):
    if mean is None and std is None:
        return volume_tensor
    if mean is None or std is None:
        raise ValueError('Both mean and std should be provided for normalization.')
    mean_tensor = torch.tensor(mean, dtype=volume_tensor.dtype).view(-1, 1, 1, 1)
    std_tensor = torch.tensor(std, dtype=volume_tensor.dtype).view(-1, 1, 1, 1)
    return (volume_tensor - mean_tensor) / std_tensor


@DATASET_REGISTRY.register()
class PairedVolumeDataset(data.Dataset):
    """Paired 3D volume dataset for cubic super-resolution."""

    def __init__(self, opt):
        super(PairedVolumeDataset, self).__init__()
        self.opt = opt
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.data_format = opt.get('data_format', 'auto')

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if opt.get('io_backend', {}).get('type', 'disk') != 'disk':
            raise ValueError('PairedVolumeDataset currently supports only disk io_backend.')

        if opt.get('meta_info_file') is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        img_gt = _load_volume(gt_path, self.data_format)
        img_lq = _load_volume(lq_path, self.data_format)

        if self.opt['phase'] == 'train':
            gt_size = self.opt.get('gt_size')
            if gt_size is not None:
                img_gt, img_lq = _paired_random_crop_3d(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = _augment_3d(img_gt, img_lq, self.opt.get('use_hflip', False),
                                         self.opt.get('use_rot', False))
        else:
            img_gt = img_gt[:img_lq.shape[0] * scale, :img_lq.shape[1] * scale, :img_lq.shape[2] * scale, :]

        img_gt = _to_tensor(img_gt)
        img_lq = _to_tensor(img_lq)

        img_lq = _normalize_tensor(img_lq, self.mean, self.std)
        img_gt = _normalize_tensor(img_gt, self.mean, self.std)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class SingleVolumeDataset(data.Dataset):
    """Single 3D volume dataset for inference."""

    def __init__(self, opt):
        super(SingleVolumeDataset, self).__init__()
        self.opt = opt
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.data_format = opt.get('data_format', 'auto')
        self.lq_folder = opt['dataroot_lq']

        if opt.get('io_backend', {}).get('type', 'disk') != 'disk':
            raise ValueError('SingleVolumeDataset currently supports only disk io_backend.')

        if opt.get('meta_info_file'):
            with open(opt['meta_info_file'], 'r') as fin:
                self.paths = [os.path.join(self.lq_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        lq_path = self.paths[index]
        img_lq = _load_volume(lq_path, self.data_format)
        img_lq = _to_tensor(img_lq)
        img_lq = _normalize_tensor(img_lq, self.mean, self.std)
        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
