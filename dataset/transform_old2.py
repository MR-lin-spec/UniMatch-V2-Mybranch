import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
from torch import nn
from torchvision.transforms import functional as TF

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
    return (img, mask) if mask is not None else img

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
        super().__init__()
        self.r = r
        self.d_min = d_min
        self.d_max = d_max
        self.p = p

    def forward(self, x):
        if torch.rand(1) > self.p:
            return x
        is_pil = isinstance(x, Image.Image)
        if is_pil:
            W, H = x.size
        else:
            H, W = x.shape[1], x.shape[2]
        d = torch.randint(self.d_min, self.d_max, (1,)).item()
        l = int(self.r * d)
        delta_x = torch.randint(0, d, (1,)).item()
        delta_y = torch.randint(0, d, (1,)).item()
        mask = torch.ones(H, W)
        for i in range(0, H + d, d):
            for j in range(0, W + d, d):
                i_start = i + delta_x
                j_start = j + delta_y
                i_end = min(i_start + l, H)
                j_end = min(j_start + l, W)
                if i_end > i_start and j_end > j_start:
                    mask[i_start:i_end, j_start:j_end] = 0
        if is_pil:
            x_tensor = transforms.ToTensor()(x)
            result_tensor = x_tensor * mask.unsqueeze(0)
            return transforms.ToPILImage()(result_tensor)
        else:
            return x * mask.unsqueeze(0)

# ========================
# ✅ 新增：仅增强链，无混合
# ========================
class AugMixOperations:
    def __init__(self):
        # 移除可能破坏结构的操作（如 solarize），保留安全操作
        self.operations = [
            self.rotate,
            self.color,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y,
        ]

    def rotate(self, x, magnitude):
        return TF.rotate(x, magnitude * 10)  # 限制旋转角度

    def color(self, x, magnitude):
        return TF.adjust_saturation(x, 1 + magnitude * 0.9)

    def contrast(self, x, magnitude):
        return TF.adjust_contrast(x, 1 + magnitude * 0.9)

    def brightness(self, x, magnitude):
        return TF.adjust_brightness(x, 1 + magnitude * 0.9)

    def sharpness(self, x, magnitude):
        return TF.adjust_sharpness(x, 1 + magnitude * 0.9)

    def shear_x(self, x, magnitude):
        return TF.affine(x, angle=0, translate=(0, 0), scale=1, shear=(magnitude * 10, 0))

    def shear_y(self, x, magnitude):
        return TF.affine(x, angle=0, translate=(0, 0), scale=1, shear=(0, magnitude * 10))

    def translate_x(self, x, magnitude):
        return TF.affine(x, angle=0, translate=(int(magnitude * 50), 0), scale=1, shear=0)

    def translate_y(self, x, magnitude):
        return TF.affine(x, angle=0, translate=(0, int(magnitude * 50)), scale=1, shear=0)

def augmix_chain(img, k=3, p_apply=1.0):
    """
    生成一条 AugMix 风格的增强链（无图像混合），适用于分割任务。
    Args:
        img: PIL.Image
        k: 增强操作数量
        p_apply: 应用概率
    Returns:
        增强后的 PIL.Image，尺寸不变
    """
    if random.random() > p_apply:
        return img
    original_size = img.size
    operations = AugMixOperations()
    x_aug = img.copy()
    for _ in range(k):
        op = random.choice(operations.operations)
        magnitude = random.random()
        x_aug = op(x_aug, magnitude)
    # 确保尺寸一致
    if x_aug.size != original_size:
        x_aug = x_aug.resize(original_size, Image.BILINEAR)
    return x_aug

# 兼容旧接口（可选）
def augmix(img, k=3, alpha=1.0, p=0.5):
    """兼容旧调用，实际使用 augmix_chain"""
    return augmix_chain(img, k=k, p_apply=p)