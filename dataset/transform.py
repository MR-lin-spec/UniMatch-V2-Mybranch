import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
from torch import nn


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
class GridMask(nn.Module):
    def __init__(self, r=0.6, d_min=96, d_max=224, p=0.8):
        """
        Args:
            r (float): 保留比例，控制掩码中未被删除的区域比例（默认0.6）。
            d_min, d_max (int): 单元大小d的随机范围（默认96-224）。
            p (float): 应用GridMask的概率（默认0.8）。
        """
        super().__init__()
        self.r = r
        self.d_min = d_min
        self.d_max = d_max
        self.p = p

    def forward(self, x):
        """
        x: 输入图像，可以是PIL.Image.Image对象或tensor，形状为 (C, H, W)
        返回: 增强后的图像
        """
        if torch.rand(1) > self.p:  # 按概率p决定是否应用GridMask
            return x
        
        # 区分PIL Image和Tensor输入
        is_pil = isinstance(x, Image.Image)
        
        # 获取图像尺寸
        if is_pil:
            W, H = x.size  # PIL Image的size是(width, height)
        else:
            H, W = x.shape[1], x.shape[2]  # Tensor的shape是(C, H, W)
            
        d = torch.randint(self.d_min, self.d_max, (1,)).item()  # 随机选择单元大小d
        l = int(self.r * d)  # 计算每个删除方块的边长
        delta_x = torch.randint(0, d, (1,)).item()  # 随机偏移量δ_x
        delta_y = torch.randint(0, d, (1,)).item()  # 随机偏移量δ_y
        
        # 生成网格掩码
        mask = torch.ones(H, W)
        for i in range(0, H + d, d):
            for j in range(0, W + d, d):
                i_start = i + delta_x
                j_start = j + delta_y
                i_end = min(i_start + l, H)
                j_end = min(j_start + l, W)
                if i_end > i_start and j_end > j_start:
                    mask[i_start:i_end, j_start:j_end] = 0  # 将区域置为0（删除）
        
        # 根据输入类型应用掩码
        if is_pil:
            # 对于PIL Image，转换为tensor进行计算，然后转回Image
            x_tensor = transforms.ToTensor()(x)
            result_tensor = x_tensor * mask.unsqueeze(0)
            return transforms.ToPILImage()(result_tensor)
        else:
            # 对于Tensor，直接应用掩码
            return x * mask.unsqueeze(0)  # 应用掩码
