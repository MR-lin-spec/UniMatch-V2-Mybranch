import torch
import torch.nn.functional as F

class MoEx:
    """
    兼容版 MoEx：
    1. y=None  （默认）→ 返回3个值，与原接口完全一致
    2. y=Tensor               → 返回4个值 (output, mean, std, y_mix)
    其余全部不变
    """

    @staticmethod
    def apply(x,
              swap_index,
              norm_type='in',
              epsilon=1e-5,
              positive_only=False,
              y=None,
              lam=0.9):
        dtype = x.dtype
        x = x.float()
        original_shape = x.shape
        original_ndim = x.dim()

        # ----------- 形状预处理（与旧版完全一致）-----------
        need_shape_conversion = False
        if original_ndim == 3:
            B, N, D = x.shape
            H = W = int(N ** 0.5)
            if H * W == N and norm_type not in ['in', 'bn', 'ln']:
                x = x.transpose(1, 2).reshape(B, D, H, W)
                need_shape_conversion = True
                is_spatial = True
            else:
                x = x.transpose(1, 2).unsqueeze(-1)
                need_shape_conversion = True
                is_spatial = False
        else:
            B, C, H, W = x.shape
            D = C
            is_spatial = True
            need_shape_conversion = False

        if x.dim() == 4:
            B, C, H, W = x.shape
        else:
            B, D, N, _ = x.shape
            C, H, W = D, N, 1

        # ----------- 归一化维度 -----------
        need_reshape = False
        if norm_type == 'bn':
            norm_dims = [0, 2, 3] if x.dim() == 4 else [0, 2]
        elif norm_type == 'in':
            norm_dims = [2, 3] if x.dim() == 4 else [2]
        elif norm_type == 'ln':
            norm_dims = [1, 2, 3] if x.dim() == 4 else [1, 2]
        elif norm_type == 'pono':
            norm_dims = [1]
        elif norm_type.startswith('gn-d'):
            G_dim = int(norm_type[4:])
            G = C // G_dim
            if x.dim() == 4:
                x = x.view(B, G, G_dim, H, W); norm_dims = [2, 3, 4]
            else:
                x = x.view(B, G, G_dim, N, 1); norm_dims = [2, 3]
            need_reshape = True
        elif norm_type.startswith('gn'):
            G = int(norm_type[2:]); G_dim = C // G
            if x.dim() == 4:
                x = x.view(B, G, G_dim, H, W); norm_dims = [2, 3, 4]
            else:
                x = x.view(B, G, G_dim, N, 1); norm_dims = [2, 3]
            need_reshape = True
        elif norm_type.startswith('gpono-d'):
            G_dim = int(norm_type[7:]); G = C // G_dim
            x = x.view(B, G, G_dim, H, W) if x.dim() == 4 else x.view(B, G, G_dim, N, 1)
            norm_dims = [2]; need_reshape = True
        elif norm_type.startswith('gpono'):
            G = int(norm_type[5:]); G_dim = C // G
            x = x.view(B, G, G_dim, H, W) if x.dim() == 4 else x.view(B, G, G_dim, N, 1)
            norm_dims = [2]; need_reshape = True
        else:
            norm_dims = [2, 3] if x.dim() == 4 else [2]

        # ----------- 矩计算 -----------
        if positive_only:
            x_pos = F.relu(x)
            s1 = x_pos.sum(dim=norm_dims, keepdim=True)
            s2 = x_pos.pow(2).sum(dim=norm_dims, keepdim=True)
            count = x_pos.gt(0).sum(dim=norm_dims, keepdim=True).clamp(min=1)
            mean = s1 / count
            var = s2 / count - mean.pow(2)
            std = var.add(epsilon).sqrt()
        else:
            mean = x.mean(dim=norm_dims, keepdim=True)
            std = x.var(dim=norm_dims, unbiased=False, keepdim=True).add(epsilon).sqrt()

        swap_mean = mean[swap_index]
        swap_std = std[swap_index]
        scale = swap_std / std
        shift = swap_mean - mean * scale
        output = x * scale + shift

        # ----------- 恢复形状 -----------
        if need_reshape:
            output = output.reshape(B, C, H, W) if x.dim() == 5 else output.reshape(B, C, N, 1)
        if need_shape_conversion:
            if is_spatial:
                output = output.reshape(B, D, H * W).transpose(1, 2)
            else:
                output = output.squeeze(-1).transpose(1, 2)
        output = output.reshape(original_shape).to(dtype)

        # ----------- 标签分支 -----------
        return_more = y is not None
        if return_more:
            y_b = y[swap_index]
            y_mix = lam * y + (1 - lam) * y_b
            return output, mean, std, y_mix
        else:
            return output, mean, std

    # ============= 工具函数保持不动 =============
    @staticmethod
    def create_swap_index(batch_size, swap_prob=0.5):
        if batch_size < 2:
            return torch.arange(batch_size)
        indices = torch.randperm(batch_size)
        swap_mask = torch.rand(batch_size) < swap_prob
        swap_index = torch.arange(batch_size)
        swap_index[swap_mask] = indices[swap_mask]
        return swap_index

    @staticmethod
    def get_supported_norm_types():
        return ['in', 'bn', 'ln', 'pono', 'gn4', 'gn8', 'gn16', 'gn32',
                'gn-d16', 'gn-d32', 'gn-d64',
                'gpono4', 'gpono8', 'gpono16',
                'gpono-d16', 'gpono-d32', 'gpono-d64']