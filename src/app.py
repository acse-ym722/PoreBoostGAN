import logging
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
            sr_img = tensor2img(visuals['result'])
            metric_data['img'] = sr_img
            
            save_img_path = osp.join(model.opt['path']['visualization'], 
                                                 f'{img_name}.png')
            imwrite(sr_img, save_img_path)

if __name__ == '__main__':
    root_path = PROJECT_ROOT
    application_pipeline(root_path)
