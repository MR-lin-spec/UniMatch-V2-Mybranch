# test_evaluate.py
import torch
import sys
import os

sys.path.append(".")  # 确保能 import 你的模块

from model.semseg.dpt import DPT  # 替换为你的实际路径
from supervised import evaluate  # 假设 evaluate 在 supervised.py 中
from util.utils import color_map
import yaml

import argparse
from copy import deepcopy
import logging
import os
import pprint
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
import random
import numpy as np
import torch.nn.functional as F  # 提前导入 F
import torch.distributed as dist


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg['ngpus'] = 1
    cfg['local_rank'] = 0
    return cfg

def main():
    # === 1. 加载配置 ===
    cfg = load_config("configs/pascal.yaml")
    cfg['dataset'] = 'pascal'
    cfg['nclass'] = 21
    cfg['crop_size'] = 473  # Pascal val 常用尺寸
    
    cudnn.enabled = True
    cudnn.benchmark = True
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    'resnet50': {
        'backbone': 'resnet50',
        'dilations': [6, 12, 18],
        'replace_stride_with_dilation': [False, True, True],
    },
    'resnet101': {
        'backbone': 'resnet101',
        'dilations': [6, 12, 18],
        'replace_stride_with_dilation': [False, True, True],
    },
    'xception': {
        'backbone': 'xception',
        'dilations': [6, 12, 18],
        'replace_stride_with_dilation': None,  # xception 不需要这个
    }
    }
    use_feature_aware_dropout_cfg = cfg.get('use_feature_aware_dropout', True)
    use_augmix_cfg = cfg.get('use_augmix', True)
    conf_thresh_cfg=cfg.get('conf_thresh', 0.95)
        #获取参数
    use_dropout_cfg = cfg.get('use_dropout', True)
    use_augment_cfg = cfg.get('use_augment', True)
    cutmix_ratio_cfg = cfg.get('cutmix_ratio', 0.5)
    # === 2. 构建模型 ===
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass'],
                 'use_feature_aware_dropout': use_feature_aware_dropout_cfg,"feature_dropout_prob":cfg['feature_dropout_prob']})
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    model.to(device)

    # === 3. 构造 fake validation data (or load real one) ===
    # 方式 A: Fake data (最快)
    img = torch.randn(1, 3, 473, 473).to(device)
    gt = torch.randint(0, cfg['nclass'], (1, 473, 473), dtype=torch.long).to(device)

    # 方式 B: 加载真实数据（更真实，但需路径）
    # from datasets.pascal import PascalVOC
    # valset = PascalVOC(root=cfg['data_root'], split='val', crop_size=None, augment=False)
    # img, gt = valset[0]
    # img = img.unsqueeze(0).to(device)
    # gt = gt.unsqueeze(0).to(device)

    # === 4. 手动构造 dataloader-like 输入 ===
    class FakeLoader:
        def __iter__(self):
            yield img, gt, ["test_img"]
        def __len__(self):
            return 1

    valloader = FakeLoader()

    # === 5. 测试 evaluate 函数 ===
    print("🧪 Testing evaluate function...")
    try:
        result = evaluate(
            model,
            valloader,
            mode=eval_mode,  # or 'sliding_window'
            cfg=cfg,
            multiplier=14,
            return_sample=False,
            dataset_name=cfg['dataset'],
            step=0,
            writer=None,  # 不需要 TensorBoard
        )
        print("✅ evaluate() succeeded!")
        print("Result:", result)
    except Exception as e:
        print("❌ evaluate() failed with error:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()