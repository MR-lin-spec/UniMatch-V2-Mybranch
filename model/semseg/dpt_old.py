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
        use_feature_aware_dropout=True,
        feature_dropout_prob=0.5,
        feature_importance_method='variance',
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
        
        # 修复：移除硬编码的binomial分布，改为在forward中动态创建
        self._dropout_prob = 0.5
        
        # MoEx相关参数
        self.use_moex = use_moex
        self.moex_norm_type = moex_norm_type
        self.moex_swap_prob = moex_swap_prob
        
        # 特征感知dropout相关参数
        self.use_feature_aware_dropout = use_feature_aware_dropout
        self.feature_dropout_prob = feature_dropout_prob
        self.feature_importance_method = feature_importance_method
        
        # 用于存储交换索引
        self._moex_swap_index = None
    
    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
            
    def set_moex_swap_index(self, swap_index):
        """设置MoEx交换索引"""
        self._moex_swap_index = swap_index
    
    def feature_aware_dropout(self, features, dropout_prob=0.5):
        """基于特征重要性进行有选择的dropout"""
        if not self.training or not self.use_feature_aware_dropout:
            return None
            
        # 获取设备信息
        device = features.device
        
        # 计算特征重要性
        if self.feature_importance_method == 'variance':
            feature_importance = features.var(dim=1, keepdim=True)
        elif self.feature_importance_method == 'mean_abs':
            feature_importance = torch.abs(features).mean(dim=1, keepdim=True)
        elif self.feature_importance_method == 'max_abs':
            feature_importance = torch.abs(features).max(dim=1, keepdim=True)[0]
        else:
            feature_importance = features.var(dim=1, keepdim=True)
        
        # 修复：确保所有操作在同一设备上
        # 归一化重要性
        importance_min = feature_importance.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        importance_max = feature_importance.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        importance_range = importance_max - importance_min + 1e-8
        importance_norm = (feature_importance - importance_min) / importance_range
        
        # 基于重要性的dropout概率调整
        adjusted_prob = dropout_prob * (1 - importance_norm)
        
        # 生成重要性感知的掩码 - 确保在同一设备上
        dropout_mask = torch.bernoulli(1 - adjusted_prob, generator=None).to(device)
        
        return dropout_mask
    
    def apply_feature_aware_dropout(self, features, dropout_mask):
        """应用特征感知dropout"""
        if dropout_mask is not None:
            if dropout_mask.dim() == 4 and features.dim() == 4:
                features = features * dropout_mask
            elif dropout_mask.dim() == 4 and features.dim() == 3:
                B, N, D = features.shape
                H = W = int(N ** 0.5)
                features_2d = features.transpose(1, 2).reshape(B, D, H, W)
                features_2d = features_2d * dropout_mask
                features = features_2d.reshape(B, D, H * W).transpose(1, 2)
        
        return features

    def forward(self, x, comp_drop=False):
        """
        Args:
            x: 输入图像
            comp_drop: 是否使用补偿dropout
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )
        
        # 应用MoEx交换（如果启用）
        if self.use_moex and self.training:
            batch_size = x.size(0)
            
            if self._moex_swap_index is not None:
                moex_swap_index = self._moex_swap_index
                self._moex_swap_index = None
            else:
                moex_swap_index = MoEx.create_swap_index(batch_size, self.moex_swap_prob)
            
            moex_features = []
            for feature in features:
                feature_moex, mean, std = MoEx.apply(
                    feature, 
                    moex_swap_index, 
                    norm_type=self.moex_norm_type,
                    epsilon=1e-5,
                    positive_only=False
                )
                moex_features.append(feature_moex)
            
            features = moex_features
        
        # 应用特征感知dropout
        if self.training and self.use_feature_aware_dropout:
            processed_features = []
            for i, feature in enumerate(features):
                B, N, D = feature.shape
                H = W = int(N ** 0.5)
                feature_2d = feature.transpose(1, 2).reshape(B, D, H, W)
                
                # 确保设备一致
                feature_2d = feature_2d.to(x.device)
                
                dropout_mask = self.feature_aware_dropout(
                    feature_2d, 
                    dropout_prob=self.feature_dropout_prob
                )
                
                if dropout_mask is not None:
                    feature_2d = self.apply_feature_aware_dropout(feature_2d, dropout_mask)
                
                feature_processed = feature_2d.reshape(B, D, H * W).transpose(1, 2)
                processed_features.append(feature_processed)
            
            features = processed_features
        
        if comp_drop:
            bs, dim = features[0].shape[0], features[0].shape[-1]
            
            # 修复：动态获取设备，避免硬编码
            device = x.device
            
            # 修复补偿dropout的设备问题
            binomial = torch.distributions.binomial.Binomial(probs=0.5)
            dropout_mask1 = binomial.sample((bs // 2, dim)).to(device) * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2, device=device)[:num_kept]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0
            
            dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
            
            features = [feature * dropout_mask.unsqueeze(1) for feature in features]
            
            out = self.head(features, patch_h, patch_w)
            
            out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
            
            return out
        
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        
        return out