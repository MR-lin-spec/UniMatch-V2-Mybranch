import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch
from model.util.moex import MoEx


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        return out


class LearnableDualDropout(nn.Module):
    """
    可学习的双流 dropout。训练阶段用 Gumbel-Softmax 采样，推理阶段用概率值。
    两个分支门控均值互补，无需额外缩放。
    """
    def __init__(self, dim: int, temp: float = 1.0):
        super().__init__()
        self.dim = dim
        self.temp = temp
        # 两个分支各一组 logit，初始化接近 0（即初始 0.5 概率）
        self.logit1 = nn.Parameter(torch.zeros(dim))
        self.logit2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor):
        """
        x: [B, N, D] 或 [B, D, H, W]，在最后一个维度(D)上做门控
        返回: 与 x 同形的张量，已乘对应掩码
        """
        if self.training:
            # 用 Gumbel-Softmax 采样 0/1 掩码
            logits1 = torch.stack([self.logit1, -self.logit1], dim=-1)  # [D, 2]
            gate1 = F.gumbel_softmax(logits1, tau=self.temp, hard=True)[..., 0]  # [D]
            gate2 = 1.0 - gate1  # 严格互补
        else:
            gate1 = torch.sigmoid(self.logit1)
            gate2 = 1.0 - gate1

        # reshape 到 [1,...,D] 方便广播
        shape = [1] * x.ndim
        shape[-1] = self.dim
        gate1 = gate1.view(shape)
        gate2 = gate2.view(shape)

        # 双流：前半批分支 1，后半批分支 2
        half = x.size(0) // 2
        if half == 0:
            return x
        x1 = x[:half] * gate1
        x2 = x[half:half*2] * gate2
        return torch.cat([x1, x2, x[half*2:]], dim=0)

class DPT(nn.Module):
    def __init__(
        self,
        encoder_size='base',
        nclass=21,
        features=128,
        out_channels=[96, 192, 384, 768],
        use_bn=False,
        use_moex=False,
        moex_norm_type='in',
        moex_swap_prob=0.5,
        use_feature_aware_dropout=False,
        feature_dropout_prob=0.5,
        feature_importance_method='variance',
        # ===== 新增两个默认参数 =====
        learnable_comp_drop=True,
        comp_drop_temp=1.0,
    ):
        super(DPT, self).__init__()

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
            'giant': [9, 19, 29, 39]
        }

        self.encoder_size = encoder_size
        self.backbone = DINOv2(model_name=encoder_size)

        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)

        # 补偿 dropout 相关
        self._dropout_prob = 0.5
        self.learnable_comp_drop = learnable_comp_drop
        if learnable_comp_drop:
            self.learnable_dual_drop = LearnableDualDropout(self.backbone.embed_dim, temp=comp_drop_temp)

        # MoEx 相关
        self.use_moex = use_moex
        self.moex_norm_type = moex_norm_type
        self.moex_swap_prob = moex_swap_prob

        # 特征感知 dropout 相关
        self.use_feature_aware_dropout = use_feature_aware_dropout
        self.feature_dropout_prob = feature_dropout_prob
        self.feature_importance_method = feature_importance_method

        self._moex_swap_index = None

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def set_moex_swap_index(self, swap_index):
        self._moex_swap_index = swap_index

    # —— 以下 feature_aware_dropout / apply_feature_aware_dropout 与旧版完全一致 —— #
    def feature_aware_dropout(self, features, dropout_prob=0.5):
        if not self.training or not self.use_feature_aware_dropout:
            return None
        device = features.device
        if self.feature_importance_method == 'variance':
            importance = features.var(dim=1, keepdim=True)
        elif self.feature_importance_method == 'mean_abs':
            importance = features.abs().mean(dim=1, keepdim=True)
        elif self.feature_importance_method == 'max_abs':
            importance = features.abs().max(dim=1, keepdim=True)[0]
        else:
            importance = features.var(dim=1, keepdim=True)
        # 归一化
        imp_min = importance.amin((2, 3), keepdim=True)
        imp_max = importance.amax((2, 3), keepdim=True)
        importance = (importance - imp_min) / (imp_max - imp_min + 1e-8)
        adjusted_prob = dropout_prob * (1 - importance)
        mask = torch.bernoulli(1 - adjusted_prob).to(device)
        return mask

    def apply_feature_aware_dropout(self, features, mask):
        if mask is None:
            return features
        if mask.dim() == 4 and features.dim() == 4:
            return features * mask
        if mask.dim() == 4 and features.dim() == 3:
            B, N, D = features.shape
            H = W = int(N ** 0.5)
            features2d = features.transpose(1, 2).reshape(B, D, H, W)
            features2d = features2d * mask
            return features2d.reshape(B, D, H * W).transpose(1, 2)
        return features

    # —— 主入口 —— #
    def forward(self, x, comp_drop=False):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )

        # MoEx 交换
        if self.use_moex and self.training:
            bsz = x.size(0)
            idx = self._moex_swap_index if self._moex_swap_index is not None \
                else MoEx.create_swap_index(bsz, self.moex_swap_prob)
            self._moex_swap_index = None
            features = [MoEx.apply(f, idx, norm_type=self.moex_norm_type,
                                 epsilon=1e-5, positive_only=False)[0]
                       for f in features]

        # 特征感知 dropout
        if self.training and self.use_feature_aware_dropout:
            proc = []
            for f in features:
                B, N, D = f.shape
                H = W = int(N ** 0.5)
                f2d = f.transpose(1, 2).reshape(B, D, H, W).to(x.device)
                mask = self.feature_aware_dropout(f2d, self.feature_dropout_prob)
                f2d = self.apply_feature_aware_dropout(f2d, mask)
                proc.append(f2d.reshape(B, D, H * W).transpose(1, 2))
            features = proc

        # ===== 改进后的双流 dropout =====
        if comp_drop:
            if self.learnable_comp_drop:
                # 走可学习门控
                features = [self.learnable_dual_drop(f) for f in features]
            else:
                # 完全保留旧版逻辑
                bs, dim = features[0].shape[0], features[0].shape[-1]
                device = x.device
                binomial = torch.distributions.binomial.Binomial(probs=0.5)
                mask1 = binomial.sample((bs // 2, dim)).to(device) * 2.0
                mask2 = 2.0 - mask1
                dropout_prob = 0.5
                num_kept = int(bs // 2 * (1 - dropout_prob))
                kept = torch.randperm(bs // 2, device=device)[:num_kept]
                mask1[kept, :] = 1.0
                mask2[kept, :] = 1.0
                mask = torch.cat([mask1, mask2], dim=0)
                features = [f * mask.unsqueeze(1) for f in features]

        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14),
                          mode='bilinear', align_corners=True)
        
        return out