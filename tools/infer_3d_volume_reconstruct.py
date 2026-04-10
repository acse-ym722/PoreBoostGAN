import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from poreboostgan.archs import build_network  # noqa: E402

try:
    import tifffile
except ImportError:
    tifffile = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reconstruct full 3D SR volume with seam-reduced tiled inference (center-trust stitching).')
    parser.add_argument('--input', required=True, type=Path, help='Input LR volume path (.tif/.tiff/.npy/.npz).')
    parser.add_argument('--output', required=True, type=Path, help='Output SR volume path (.tif/.tiff/.npy).')
    parser.add_argument('--model-path', required=True, type=Path, help='Generator checkpoint path, e.g. net_g_2000.pth.')
    parser.add_argument('--config', required=True, type=Path, help='YAML containing network_g and scale.')
    parser.add_argument('--param-key', type=str, default='auto', help='Checkpoint key: auto|params|params_ema.')

    parser.add_argument('--data-format', type=str, default='auto', choices=['auto', 'dhwc', 'cdhw'],
                        help='Volume channel layout for 4D input.')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--fp16', action='store_true', help='Use autocast fp16 on CUDA.')

    parser.add_argument('--core-size', type=int, default=64,
                        help='Core LR block size written to output. Full volume is covered by these non-overlap cores.')
    parser.add_argument('--context', type=int, default=16,
                        help='Extra LR context around each core for inference. Only core center is written.')
    parser.add_argument('--max-blocks', type=int, default=0, help='Debug option: only run first N blocks. 0 means all.')

    parser.add_argument('--save-dtype', type=str, default='uint8', choices=['uint8', 'float32'],
                        help='Output storage dtype.')
    parser.add_argument('--tif-compression', type=str, default='none', choices=['none', 'zlib', 'lzma'],
                        help='Compression for tif output.')
    parser.add_argument('--tif-layout', type=str, default='imagej', choices=['imagej', 'plain'],
                        help='TIFF metadata layout. imagej is preferred for ImageJ/Fiji compatibility.')
    return parser.parse_args()


def _load_yaml(path: Path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_data_layout(volume: np.ndarray, data_format: str):
    # Return DHWC float32 in [0,1]
    if volume.ndim == 3:
        volume = volume[..., None]  # DHW -> DHWC
    elif volume.ndim == 4:
        fmt = data_format.lower()
        if fmt == 'dhwc':
            pass
        elif fmt == 'cdhw':
            volume = np.transpose(volume, (1, 2, 3, 0))
        elif fmt == 'auto':
            if volume.shape[-1] <= 8 and volume.shape[0] > 8:
                pass
            elif volume.shape[0] <= 8 and volume.shape[-1] > 8:
                volume = np.transpose(volume, (1, 2, 3, 0))
            else:
                # Ambiguous: default to DHWC.
                pass
        else:
            raise ValueError(f'Unsupported data_format={data_format}.')
    else:
        raise ValueError(f'Expected 3D/4D volume, got shape={volume.shape}.')

    if np.issubdtype(volume.dtype, np.integer):
        volume = volume.astype(np.float32) / float(np.iinfo(volume.dtype).max)
    else:
        volume = volume.astype(np.float32)
    return volume


def _load_volume(path: Path, data_format: str):
    suffix = path.suffix.lower()
    if suffix in ['.tif', '.tiff']:
        if tifffile is None:
            raise ImportError('tifffile is required to read tif/tiff.')
        volume = tifffile.imread(str(path))
    elif suffix == '.npy':
        volume = np.load(path)
    elif suffix == '.npz':
        with np.load(path) as z:
            if not z.files:
                raise ValueError(f'No arrays in {path}.')
            volume = z[z.files[0]]
    else:
        raise ValueError(f'Unsupported input extension: {suffix}')
    return _resolve_data_layout(np.asarray(volume), data_format)


def _pick_param_key(ckpt, param_key):
    if not isinstance(ckpt, dict):
        return None
    if param_key != 'auto':
        return param_key
    if 'params_ema' in ckpt:
        return 'params_ema'
    if 'params' in ckpt:
        return 'params'
    # Fallback: treat full dict as state dict if tensor-like values.
    return None


def _load_generator(model_path: Path, config_path: Path, device: torch.device, param_key: str):
    cfg = _load_yaml(config_path)
    if 'network_g' not in cfg:
        raise ValueError(f'network_g not found in {config_path}')
    if 'scale' not in cfg:
        raise ValueError(f'scale not found in {config_path}')

    net_g = build_network(cfg['network_g'])
    ckpt = torch.load(model_path, map_location='cpu')
    key = _pick_param_key(ckpt, param_key)
    state_dict = ckpt[key] if key is not None else ckpt
    net_g.load_state_dict(state_dict, strict=True)
    net_g.to(device)
    net_g.eval()
    return net_g, int(cfg['scale'])


def _ranges(length: int, core_size: int):
    starts = list(range(0, length, core_size))
    return [(s, min(s + core_size, length)) for s in starts]


def _to_save_dtype(chunk: np.ndarray, save_dtype: str):
    chunk = np.clip(chunk, 0.0, 1.0)
    if save_dtype == 'float32':
        return chunk.astype(np.float32)
    return np.round(chunk * 255.0).astype(np.uint8)


def _prepare_output_array(output_path: Path, shape_dhwc, save_dtype: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if save_dtype == 'float32' else np.uint8
    if output_path.suffix.lower() == '.npy':
        out = np.lib.format.open_memmap(output_path, mode='w+', dtype=dtype, shape=shape_dhwc)
        temp_npy = None
    elif output_path.suffix.lower() in ['.tif', '.tiff']:
        fd, tmp_name = tempfile.mkstemp(prefix='sr3d_tmp_', suffix='.npy', dir=str(output_path.parent))
        os.close(fd)
        temp_npy = Path(tmp_name)
        out = np.lib.format.open_memmap(temp_npy, mode='w+', dtype=dtype, shape=shape_dhwc)
    else:
        raise ValueError('Output must end with .npy/.tif/.tiff')
    return out, temp_npy


def _finalize_output(output_array, output_path: Path, temp_npy: Path, save_dtype: str, compression: str, tif_layout: str):
    output_array.flush()
    if output_path.suffix.lower() == '.npy':
        return

    if tifffile is None:
        raise ImportError('tifffile is required to write tif/tiff.')

    arr = np.asarray(output_array)
    # Use DHW for single-channel to keep standard volume style.
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    comp = None if compression == 'none' else compression
    need_bigtiff = arr.nbytes >= (4 * 1024**3 - 32 * 1024**2)
    if tif_layout == 'imagej':
        arr_to_write = arr
        if arr.ndim == 3:
            axes = 'ZYX'
        elif arr.ndim == 4:
            # Convert DHWC (ZYXC) to ZCYX for ImageJ hyperstack-compatible axes order.
            arr_to_write = np.moveaxis(arr, -1, 1)
            axes = 'ZCYX'
        else:
            axes = None
        if axes is not None:
            tifffile.imwrite(
                str(output_path),
                np.ascontiguousarray(arr_to_write),
                compression=comp,
                bigtiff=need_bigtiff,
                imagej=True,
                metadata={'axes': axes})
        else:
            tifffile.imwrite(str(output_path), np.ascontiguousarray(arr), compression=comp, bigtiff=need_bigtiff)
    else:
        tifffile.imwrite(str(output_path), np.ascontiguousarray(arr), compression=comp, bigtiff=need_bigtiff)
    if temp_npy is not None and temp_npy.exists():
        temp_npy.unlink()


def infer_volume(args):
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    net_g, scale = _load_generator(args.model_path, args.config, device, args.param_key)
    volume = _load_volume(args.input, args.data_format)  # DHWC in [0,1]
    d, h, w, c = volume.shape
    out_shape = (d * scale, h * scale, w * scale, c)

    print(f'Input volume: {args.input}')
    print(f'Input shape (D,H,W,C): {volume.shape}')
    print(f'Scale: x{scale}')
    print(f'Core size: {args.core_size}, context: {args.context}')
    print(f'Output shape (D,H,W,C): {out_shape}')
    print(f'Device: {device}, fp16: {args.fp16}')

    out_arr, temp_npy = _prepare_output_array(args.output, out_shape, args.save_dtype)

    z_ranges = _ranges(d, args.core_size)
    y_ranges = _ranges(h, args.core_size)
    x_ranges = _ranges(w, args.core_size)
    total_blocks = len(z_ranges) * len(y_ranges) * len(x_ranges)
    if args.max_blocks > 0:
        total_blocks = min(total_blocks, args.max_blocks)

    # DHWC -> CDHW
    volume_cdhw = np.transpose(volume, (3, 0, 1, 2))

    block_idx = 0
    pbar = tqdm(total=total_blocks, desc='3D reconstruct', unit='block')
    with torch.no_grad():
        for z0, z1 in z_ranges:
            for y0, y1 in y_ranges:
                for x0, x1 in x_ranges:
                    block_idx += 1
                    if args.max_blocks > 0 and block_idx > args.max_blocks:
                        break

                    iz0 = max(0, z0 - args.context)
                    iy0 = max(0, y0 - args.context)
                    ix0 = max(0, x0 - args.context)
                    iz1 = min(d, z1 + args.context)
                    iy1 = min(h, y1 + args.context)
                    ix1 = min(w, x1 + args.context)

                    lr_patch = volume_cdhw[:, iz0:iz1, iy0:iy1, ix0:ix1]
                    lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0).to(device=device, dtype=torch.float32)

                    if device.type == 'cuda' and args.fp16:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            sr_patch = net_g(lr_tensor)
                    else:
                        sr_patch = net_g(lr_tensor)

                    # Trust center region: only write corresponding core region.
                    cz0 = (z0 - iz0) * scale
                    cy0 = (y0 - iy0) * scale
                    cx0 = (x0 - ix0) * scale
                    cz1 = cz0 + (z1 - z0) * scale
                    cy1 = cy0 + (y1 - y0) * scale
                    cx1 = cx0 + (x1 - x0) * scale

                    sr_core = sr_patch[:, :, cz0:cz1, cy0:cy1, cx0:cx1]
                    sr_core = sr_core.squeeze(0).detach().cpu().numpy()  # CDHW
                    sr_core = np.transpose(sr_core, (1, 2, 3, 0))  # DHWC
                    sr_core = _to_save_dtype(sr_core, args.save_dtype)

                    oz0, oy0, ox0 = z0 * scale, y0 * scale, x0 * scale
                    oz1, oy1, ox1 = z1 * scale, y1 * scale, x1 * scale
                    out_arr[oz0:oz1, oy0:oy1, ox0:ox1, :] = sr_core

                    pbar.update(1)
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                if args.max_blocks > 0 and block_idx >= args.max_blocks:
                    break
            if args.max_blocks > 0 and block_idx >= args.max_blocks:
                break
    pbar.close()

    _finalize_output(out_arr, args.output, temp_npy, args.save_dtype, args.tif_compression, args.tif_layout)
    print(f'Done. Output saved to: {args.output}')


if __name__ == '__main__':
    infer_volume(parse_args())
