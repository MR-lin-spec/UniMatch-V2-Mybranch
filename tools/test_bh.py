import sys
import os
# 添加项目根目录到Python路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.semseg.dpt import DPT

model = DPT(encoder_size='small', features=64, out_channels=[48,96,192,384], nclass=21).cuda()
x = torch.randn(16, 3, 476, 476).cuda()  # try 16, 12, 8...
with torch.no_grad():
    y = model(x)
print("OK with batch_size =", x.shape[0])