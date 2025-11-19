import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.semi import SemiDataset
from model.semseg.dpt import DPT

cfg = yaml.load(open('configs/pascal.yaml','r'), Loader=yaml.Loader)

model_configs = {
    'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth', map_location='cpu')
model.backbone.load_state_dict(state_dict)
model.eval().cuda()

valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

multiplier = 14
for img, mask, id in valloader:
    print('orig img.shape', img.shape, 'mask.shape', mask.shape)
    img = img.cuda()
    if multiplier is not None:
        ori_h, ori_w = img.shape[-2:]
        if multiplier == 512:
            new_h, new_w = 512, 512
        else:
            new_h = int(ori_h / multiplier + 0.5) * multiplier
            new_w = int(ori_w / multiplier + 0.5) * multiplier
        print('resizing to', (new_h, new_w))
        img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)
    print('input to model shape:', img.shape)
    out = model(img, comp_drop=False)
    print('raw out shape:', out.shape)
    if multiplier is not None:
        out = F.interpolate(out, (ori_h, ori_w), mode='bilinear', align_corners=True)
        print('resized out shape:', out.shape)
    arg = out.argmax(dim=1)
    print('argmax shape:', arg.shape)
    break
