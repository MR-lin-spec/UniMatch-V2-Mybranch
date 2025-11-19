import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import yaml
import torch
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

for i, (img, mask, id) in enumerate(valloader):
    print('batch idx', i)
    print('img type, device, shape:', type(img), getattr(img,'device',None), getattr(img,'shape',None))
    print('mask type, device, shape:', type(mask), getattr(mask,'device',None), getattr(mask,'shape',None))
    try:
        img_cuda = img.cuda()
        out = model(img_cuda)
        print('model output shape:', out.shape)
        arg = out.argmax(dim=1)
        print('argmax shape:', arg.shape)
    except Exception as e:
        print('model forward failed:', e)
    break
