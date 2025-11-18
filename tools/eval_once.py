import argparse
import yaml
import sys, os

# ensure repo root is in sys.path so local modules can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate


def build_model(cfg):
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth', map_location='cpu')
    model.backbone.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg)
    model.to(device)
    model.eval()

    # Make distributed all_reduce a no-op for single-process evaluation
    try:
        dist.all_reduce
        dist.all_reduce = lambda x: x
    except Exception:
        pass

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)

    print('Short eval result: mIoU={:.4f}'.format(mIoU))
    for i, iou in enumerate(iou_class):
        print(f'Class {i}: {iou:.4f}')
