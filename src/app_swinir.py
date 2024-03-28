import logging
import torch
from torch.nn import functional as F
from tqdm import tqdm
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
from basicsr.utils import get_root_logger, imwrite, tensor2img

def application_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    
    # log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    # logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # logger.info(get_env_info())
    # logger.info(dict2str(opt))

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

            # model.test
            # pad to multiplication of window_size
            window_size = model.opt['network_g']['window_size']
            scale = model.opt.get('scale', 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = model.lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = F.pad(model.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            if hasattr(model, 'net_g_ema'):
                model.net_g_ema.eval()
                with torch.no_grad():
                    model.output = model.net_g_ema(img)
            else:
                model.net_g.eval()
                with torch.no_grad():
                    model.output = model.net_g(img)
                model.net_g.train()

            _, _, h, w = model.output.size()
            model.output = model.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


            visuals = model.get_current_visuals()
            sr_img = tensor2img(visuals['result'])
            metric_data['img'] = sr_img

            
            save_img_path = osp.join(model.opt['path']['visualization'], 
                                                 f'{img_name}.png')
            imwrite(sr_img, save_img_path)



if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    application_pipeline(root_path)