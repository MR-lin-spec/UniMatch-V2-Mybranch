import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
from torch import nn
from torch.distributions import Dirichlet, Beta
import torchvision.transforms as T
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
# 定义常用的图像增强操作集合
class AugMixOperations:
    def __init__(self):
        self.operations = [
            self.rotate,
            self.solarize,
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
        return TF.rotate(x, magnitude * 30)
    
    def solarize(self, x, magnitude):
        return TF.solarize(x, magnitude * 256)
    
    def color(self, x, magnitude):
        return TF.adjust_saturation(x, 1 + magnitude * 0.9)
    
    def contrast(self, x, magnitude):
        return TF.adjust_contrast(x, 1 + magnitude * 0.9)
    
    def brightness(self, x, magnitude):
        return TF.adjust_brightness(x, 1 + magnitude * 0.9)
    
    def sharpness(self, x, magnitude):
        return TF.adjust_sharpness(x, 1 + magnitude * 0.9)
    
    def shear_x(self, x, magnitude):
        return TF.affine(x, angle=0, translate=(0, 0), scale=1, shear=(magnitude * 45, 0))
    
    def shear_y(self, x, magnitude):
        return TF.affine(x, angle=0, translate=(0, 0), scale=1, shear=(0, magnitude * 45))
    
    def translate_x(self, x, magnitude):
        return TF.affine(x, angle=0, translate=(int(magnitude * 150), 0), scale=1, shear=0)
    
    def translate_y(self, x, magnitude):
        return TF.affine(x, angle=0, translate=(0, int(magnitude * 150)), scale=1, shear=0)

# 图像预处理转换函数
def preprocess_for_augmix(img):
    """将PIL图像转换为tensor并标准化"""
    if isinstance(img, Image.Image):
        # 转换为tensor
        img_tensor = transforms.ToTensor()(img)
        # 标准化
        img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
        return img_tensor
    return img

def postprocess_from_augmix(img_tensor):
    """将tensor转换回PIL图像"""
    if isinstance(img_tensor, torch.Tensor):
        # 反标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = img_tensor * std + mean
        # 限制到[0,1]范围
        img_tensor = torch.clamp(img_tensor, 0, 1)
        # 转换为PIL图像
        img_pil = transforms.ToPILImage()(img_tensor)
        return img_pil
    return img_tensor

# AugMix 增强函数（保持尺寸不变）
def augment_and_mix(x, operations, k=3, alpha=1.0):
    """
    x: 输入图像，可以是PIL.Image或tensor
    返回: 增强后的图像，尺寸与输入相同
    """
    # 记录原始类型和尺寸
    is_pil = isinstance(x, Image.Image)
    original_size = x.size if is_pil else x.shape[-2:]
    
    # 转换为tensor进行AugMix处理
    if is_pil:
        x_tensor = preprocess_for_augmix(x)
    else:
        x_tensor = x
    
    aug = torch.zeros_like(x_tensor)
    
    # 采样混合权重
    dirichlet = Dirichlet(torch.ones(k) * alpha)
    weights = dirichlet.sample()
    
    for i in range(k):
        # 临时转换为PIL进行空间变换（如果需要）
        temp_img = x_tensor.clone()
        if is_pil:
            temp_img_pil = postprocess_from_augmix(temp_img)
        else:
            # 对于tensor，直接使用torchvision的functional
            temp_img_pil = None
        
        # 采样操作和强度
        op = random.choice(operations.operations)
        magnitude = random.random()
        
        # 应用操作
        if is_pil:
            x_aug_pil = op(temp_img_pil, magnitude)
            x_aug = preprocess_for_augmix(x_aug_pil)
        else:
            # 对于tensor，直接应用操作
            x_aug = op(temp_img, magnitude)
        
        aug = aug + weights[i] * x_aug
    
    # 采样插值权重
    beta = Beta(alpha, alpha)
    m = beta.sample()
    
    # 与原始图像插值
    augmix_tensor = m * x_tensor + (1 - m) * aug
    
    # 转换回原始格式
    if is_pil:
        result = postprocess_from_augmix(augmix_tensor)
        # 确保尺寸一致
        if result.size != original_size:
            result = result.resize(original_size, Image.BILINEAR)
    else:
        result = augmix_tensor
        # 确保尺寸一致
        if result.shape[-2:] != original_size:
            result = F.interpolate(result.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
    
    return result

# 简化的AugMix增强函数，便于在数据加载器中使用
def augmix(img, k=3, alpha=1.0, p=0.5):
    """
    即插即用的AugMix增强
    Args:
        img: 输入图像(PIL.Image或tensor)
        k: 操作链数量
        alpha: Dirichlet分布参数
        p: 应用概率
    Returns:
        增强后的图像，尺寸不变
    """
    if random.random() > p:
        return img
    
    operations = AugMixOperations()
    return augment_and_mix(img, operations, k, alpha)

# JS 散度计算
def jensen_shannon_divergence(p, q, r):
    m = 0.5 * (p + q)
    js_p = 0.5 * (F.kl_div(m.log(), p, reduction='batchmean') + F.kl_div(m.log(), q, reduction='batchmean'))
    m = 0.5 * (p + r)
    js_r = 0.5 * (F.kl_div(m.log(), p, reduction='batchmean') + F.kl_div(m.log(), r, reduction='batchmean'))
    return (js_p + js_r) / 2

# 完整的 AugMix 训练损失
def augmix_loss(model, x, y, operations, k=3, alpha=1.0, lambda_js=1.0):
    # 原始图像的预测
    logits_orig = model(x)
    loss_orig = F.cross_entropy(logits_orig, y)
    # 生成两个 AugMix 样本
    x_aug1 = augment_and_mix(x, operations, k, alpha)
    x_aug2 = augment_and_mix(x, operations, k, alpha)
    # 增强样本的预测
    logits_aug1 = model(x_aug1)
    logits_aug2 = model(x_aug2)
    # 计算 JS 散度正则项
    p = F.softmax(logits_orig, dim=1)
    q = F.softmax(logits_aug1, dim=1)
    r = F.softmax(logits_aug2, dim=1)
    js_loss = jensen_shannon_divergence(p, q, r)
    # 总损失
    total_loss = loss_orig + lambda_js * js_loss
    return total_loss
