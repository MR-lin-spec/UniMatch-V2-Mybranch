import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEx:
    """MoEx操作：交换特征的均值和标准差"""
    
    @staticmethod
    def apply(x, swap_index, norm_type='in', epsilon=1e-5, positive_only=False):
        '''
        MoEx操作：交换特征的均值和标准差
        
        Args:
            x: 输入特征 [B, C, H, W] 或 [B, N, D]
            swap_index: 交换索引 [B]
            norm_type: 归一化类型
            epsilon: 数值稳定性常数
            positive_only: 是否仅对正特征计算矩
        '''
        dtype = x.dtype
        x = x.float()
        
        original_shape = x.shape
        original_ndim = x.dim()
        
        # 首先判断是否为序列格式，并决定是否转换
        need_shape_conversion = False
        if original_ndim == 3:
            B, N, D = x.shape
            H = W = int(N ** 0.5)
            
            # 如果是空间特征（如1156=34x34），转换为图像格式
            if H * W == N and norm_type not in ['in', 'bn', 'ln']:
                # 转换为图像格式以获得更好的归一化效果
                x = x.transpose(1, 2).reshape(B, D, H, W)
                need_shape_conversion = True
                is_spatial = True
            else:
                # 保持序列格式，添加空间维度 [B, D, N, 1]
                x = x.transpose(1, 2).unsqueeze(-1)
                need_shape_conversion = True
                is_spatial = False
        else:
            # 已经是图像格式
            B, C, H, W = x.shape
            D = C
            is_spatial = True
            need_shape_conversion = False
        
        # 获取当前形状
        if x.dim() == 4:
            B, C, H, W = x.shape
        else:  # 序列格式保持为 [B, D, N, 1]
            B, D, N, _ = x.shape
            C, H, W = D, N, 1
        
        # 根据归一化类型确定计算维度（统一处理）
        need_reshape = False
        if norm_type == 'bn':
            norm_dims = [0, 2, 3] if x.dim() == 4 else [0, 2]  # 适应不同维度
        elif norm_type == 'in':
            norm_dims = [2, 3] if x.dim() == 4 else [2]  # 空间维度或序列维度
        elif norm_type == 'ln':
            norm_dims = [1, 2, 3] if x.dim() == 4 else [1, 2]  # 通道和空间维度
        elif norm_type == 'pono':
            norm_dims = [1]  # 仅跨通道维度
        elif norm_type.startswith('gn-d'):
            G_dim = int(norm_type[4:])
            G = C // G_dim
            if x.dim() == 4:
                x = x.view(B, G, G_dim, H, W)
                norm_dims = [2, 3, 4]
            else:
                x = x.view(B, G, G_dim, N, 1)
                norm_dims = [2, 3]
            need_reshape = True
            
        elif norm_type.startswith('gn'):
            G = int(norm_type[2:])
            G_dim = C // G
            if x.dim() == 4:
                x = x.view(B, G, G_dim, H, W)
                norm_dims = [2, 3, 4]
            else:
                x = x.view(B, G, G_dim, N, 1)
                norm_dims = [2, 3]
            need_reshape = True
        elif norm_type.startswith('gpono-d'):
            G_dim = int(norm_type[7:])
            G = C // G_dim
            if x.dim() == 4:
                x = x.view(B, G, G_dim, H, W)
            else:
                x = x.view(B, G, G_dim, N, 1)
            norm_dims = [2]  # 仅跨通道维度
            need_reshape = True
        elif norm_type.startswith('gpono'):
            G = int(norm_type[5:])
            G_dim = C // G
            if x.dim() == 4:
                x = x.view(B, G, G_dim, H, W)
            else:
                x = x.view(B, G, G_dim, N, 1)
            norm_dims = [2]  # 仅跨通道维度
            need_reshape = True
        else:
            norm_dims = [2, 3] if x.dim() == 4 else [2]  # 默认

        # 计算均值和标准差
        if positive_only:
            x_pos = F.relu(x)
            s1 = x_pos.sum(dim=norm_dims, keepdim=True)
            s2 = x_pos.pow(2).sum(dim=norm_dims, keepdim=True)
            count = x_pos.gt(0).sum(dim=norm_dims, keepdim=True)
            count = count.clamp(min=1)
            mean = s1 / count
            var = s2 / count - mean.pow(2)
            std = var.add(epsilon).sqrt()
        else:
            mean = x.mean(dim=norm_dims, keepdim=True)
            std = x.var(dim=norm_dims, unbiased=False, keepdim=True).add(epsilon).sqrt()

        # 交换矩
        swap_mean = mean[swap_index]
        swap_std = std[swap_index]

        # 应用MoEx变换
        scale = swap_std / std
        shift = swap_mean - mean * scale
        output = x * scale + shift

        # 恢复分组形状
        if need_reshape:
            if x.dim() == 5:  # 图像格式
                output = output.reshape(B, C, H, W)
            else:  # 序列格式
                output = output.reshape(B, C, N, 1)
        
        # 恢复原始格式
        if need_shape_conversion:
            if is_spatial:
                # 从图像格式恢复为序列格式 [B, N, D]
                output = output.reshape(B, D, H * W).transpose(1, 2)
            else:
                # 从 [B, D, N, 1] 恢复为 [B, N, D]
                output = output.squeeze(-1).transpose(1, 2)
        
        # 确保形状一致
        output = output.reshape(original_shape)
        
        return output.to(dtype), mean, std

    @staticmethod
    def create_swap_index(batch_size, swap_prob=0.5):
        """创建交换索引"""
        if batch_size < 2:
            return torch.arange(batch_size)
            
        indices = torch.randperm(batch_size)
        swap_mask = torch.rand(batch_size) < swap_prob
        swap_index = torch.arange(batch_size)
        swap_index[swap_mask] = indices[swap_mask]
        
        return swap_index

    @staticmethod
    def get_supported_norm_types():
        """获取支持的归一化类型"""
        return ['in', 'bn', 'ln', 'pono', 'gn4', 'gn8', 'gn16', 'gn32',
                'gn-d16', 'gn-d32', 'gn-d64', 'gpono4', 'gpono8', 'gpono16',
                'gpono-d16', 'gpono-d32', 'gpono-d64']