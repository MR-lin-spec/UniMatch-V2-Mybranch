import model.backbone.resnet as resnet
from model.backbone.xception import xception
import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        backbone='resnet50',
        nclass=21,
        dilations=[6, 12, 18],
        replace_stride_with_dilation=[False, False, True],
        use_feature_aware_dropout=False,
        feature_dropout_prob=0.2,
        feature_importance_method='variance',
        use_comp_drop=False,  # 可选：是否默认启用 comp_drop（通常由 forward 控制，但保留扩展性）
    ):
        super(DeepLabV3Plus, self).__init__()
        
        # Backbone setup
        if 'resnet' in backbone:
            self.backbone = resnet.__dict__[backbone](
                pretrained=True,
                replace_stride_with_dilation=replace_stride_with_dilation
            )
        else:
            assert backbone == 'xception', f"Unsupported backbone: {backbone}"
            self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, dilations)
        self.reduce = nn.Sequential(
            nn.Conv2d(low_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)

        # ===== Dropout & Augmentation Control (aligned with DPT style) =====
        self.use_feature_aware_dropout = use_feature_aware_dropout
        self.feature_dropout_prob = feature_dropout_prob
        self.feature_importance_method = feature_importance_method
        self.use_comp_drop = use_comp_drop  # reserved for future; comp_drop usually triggered via forward arg

    def feature_aware_dropout_mask(self, features, drop_prob=None, method=None):
        """
        Generate feature-aware dropout mask.
        Input: [B, C, H, W]
        Output: binary mask of same shape
        """
        if not self.training:
            return None

        drop_prob = drop_prob if drop_prob is not None else self.feature_dropout_prob
        method = method if method is not None else self.feature_importance_method

        if not self.use_feature_aware_dropout or drop_prob <= 0:
            return None

        B, C, H, W = features.shape
        device = features.device

        if method == 'variance':
            imp = features.var(dim=1, keepdim=True)  # (B,1,H,W)
        elif method == 'mean_abs':
            imp = torch.abs(features).mean(dim=1, keepdim=True)
        elif method == 'max_abs':
            imp = torch.abs(features).max(dim=1, keepdim=True)[0]
        else:
            imp = features.var(dim=1, keepdim=True)

        # Per-sample min-max normalization to [0, 1]
        imp_flat = imp.view(B, -1)
        min_val = imp_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_val = imp_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        eps = 1e-8
        imp_norm = (imp - min_val) / (max_val - min_val + eps)

        # Important regions → lower dropout probability
        adjusted_prob = drop_prob * (1 - imp_norm)
        mask = torch.bernoulli(1 - adjusted_prob).to(device)
        return mask

    def _decode(self, c1, c4, h, w):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c1 = self.reduce(c1)
        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)
        out = self.classifier(feature)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out

    def forward(self, x, comp_drop=False):
        h, w = x.shape[-2:]
        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        # Apply feature-aware dropout during training (if enabled)
        if self.use_feature_aware_dropout and self.training:
            mask_c1 = self.feature_aware_dropout_mask(c1)
            mask_c4 = self.feature_aware_dropout_mask(c4)
            if mask_c1 is not None:
                c1 = c1 * mask_c1
            if mask_c4 is not None:
                c4 = c4 * mask_c4

        if comp_drop:
            assert x.size(0) % 2 == 0, "Batch size must be even when comp_drop=True"
            bs_half = x.size(0) // 2
            c1_1, c1_2 = c1[:bs_half], c1[bs_half:]
            c4_1, c4_2 = c4[:bs_half], c4[bs_half:]

            device = x.device
            C = c4.size(1)
            binomial = torch.distributions.binomial.Binomial(probs=0.5)
            mask1 = binomial.sample((bs_half, C)).to(device)  # [B/2, C]
            mask1 = mask1 * 2.0
            mask2 = (2.0 - mask1)

            c4_2 = c4_2 * mask2.unsqueeze(-1).unsqueeze(-1)
            low_mask = mask2.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            c1_2 = c1_2 * low_mask

            out_clean = self._decode(c1_1, c4_1, h, w)
            out_perturbed = self._decode(c1_2, c4_2, h, w)

            return torch.cat([out_clean, out_perturbed], dim=0)

        else:
            # ✅ 必须有这个 else 分支并返回结果！
            out = self._decode(c1, c4, h, w)
            return out
        
       
           


# --- Helper Modules (unchanged) ---

def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)