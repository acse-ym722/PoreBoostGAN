import logging
import numpy as np
import sys
import torch
from tqdm import tqdm
from os import path as osp

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from poreboostgan.data import build_dataloader, build_dataset
from poreboostgan.models import build_model
from poreboostgan.utils.options import parse_options
from poreboostgan.utils import imwrite, tensor2img

try:
    import tifffile
except ImportError:
    tifffile = None


def _tensor_to_volume(tensor):
    # Accept (1, C, D, H, W) or (C, D, H, W)
    if tensor.dim() == 5:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 4:
        raise ValueError(f'Expected 4D/5D tensor for volume output, got shape={tuple(tensor.shape)}.')
    volume = tensor.detach().float().cpu().clamp_(0, 1).permute(1, 2, 3, 0).numpy()
    if volume.shape[-1] == 1:
        volume = np.squeeze(volume, axis=-1)
    return volume.astype(np.float32)


def _save_volume_tif(volume, save_path):
    if tifffile is None:
        raise ImportError('tifffile is required for 3D tif output. Please install tifffile.')
    # Save as uint8 for better compatibility with ImageJ/common TIFF viewers.
    volume_u8 = np.clip(volume, 0.0, 1.0)
    volume_u8 = np.round(volume_u8 * 255.0).astype(np.uint8)
    if volume_u8.ndim == 4 and volume_u8.shape[-1] == 1:
        volume_u8 = volume_u8[..., 0]

    if volume_u8.ndim == 3:
        # ImageJ stack: Z, Y, X
        volume_to_write = volume_u8
        axes = 'ZYX'
    elif volume_u8.ndim == 4:
        # Convert Z, Y, X, C -> Z, C, Y, X for ImageJ hyperstack axes order.
        volume_to_write = np.moveaxis(volume_u8, -1, 1)
        axes = 'ZCYX'
    else:
        raise ValueError(f'Expected 3D/4D volume, got shape={volume_u8.shape}.')

    need_bigtiff = volume_to_write.nbytes >= (4 * 1024**3 - 32 * 1024**2)
    tifffile.imwrite(
        str(save_path),
        np.ascontiguousarray(volume_to_write),
        imagej=True,
        metadata={'axes': axes},
        compression=None,
        bigtiff=need_bigtiff)


def application_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        print(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    metric_data = dict()
    
    for test_loader in tqdm(test_loaders):
        test_set_name = test_loader.dataset.opt['name']
        print(f'Testing {test_set_name}...')
        for idx, val_data in enumerate(tqdm(test_loader)):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            model.feed_data(val_data)
            model.test()
            visuals = model.get_current_visuals()
            result_tensor = visuals['result']

            if result_tensor.dim() == 5:
                sr_volume = _tensor_to_volume(result_tensor)
                metric_data['img'] = sr_volume
                save_img_path = osp.join(model.opt['path']['visualization'], f'{img_name}.tif')
                osp_dir = osp.dirname(save_img_path)
                if osp_dir:
                    import os
                    os.makedirs(osp_dir, exist_ok=True)
                _save_volume_tif(sr_volume, save_img_path)
            else:
                sr_img = tensor2img(result_tensor)
                metric_data['img'] = sr_img
                save_img_path = osp.join(model.opt['path']['visualization'], f'{img_name}.png')
                imwrite(sr_img, save_img_path)

if __name__ == '__main__':
    root_path = PROJECT_ROOT
    application_pipeline(root_path)
