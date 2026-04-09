# 动态HOG感知自注意力模块（Dynamic HOG-aware Self-Attention, DHOGSA）
# 核心设计：融合方向梯度直方图（HOG）特征与双路径自注意力，通过“HOG特征提取→动态排序→双路径注意力→特征还原”的流程，
# 基于图像边缘与纹理的方向信息引导注意力计算，强化局部结构特征的同时捕捉全局依赖，提升模型对目标轮廓与细节的辨识度
# 参考: https://github.com/Fire-friend/HOGformer (AAAI 2026)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

Conv2d = nn.Conv2d


class Attention_DHOGSA(nn.Module):
    """
    动态HOG感知自注意力模块（Dynamic HOG-aware Self-Attention, DHOGSA）
    
    功能：HOG特征引导的双路径自注意力，强化结构特征与全局依赖捕捉
    
    核心设计：
        - HOG特征融合：提取局部patch的方向梯度直方图，引导特征排序与注意力权重分配
        - 双路径注意力：两种维度重排方式的自注意力并行计算，互补捕捉不同维度依赖
        - 动态特征排序：基于HOG梯度幅值与方向，对特征像素动态排序，强化关键结构
        - 深度卷积增强：QKV生成后经深度卷积提升局部特征表达，适配注意力计算
    
    Args:
        dim (int): 输入/输出通道数
        num_heads (int): 注意力头数
        bias (bool): 卷积层是否带偏置（默认False）
        ifBox (bool): 第一路径注意力的维度重排模式开关（默认True）
        patch_size (int): HOG特征提取的patch尺寸（默认8）
        clip_limit (float): 预留参数（当前未启用，用于梯度裁剪，默认1.0）
        n_bins (int): HOG特征的方向直方图bins数（默认9，覆盖0~180°）
    
    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`
    
    注意事项：
        - 使用数值稳定的softmax计算避免溢出
        - HOG特征计算采用Sobel算子
        - 双路径注意力分别采用不同的维度重排策略
    """

    def __init__(self, dim, num_heads=8, bias=False, ifBox=True, patch_size=8, clip_limit=1.0, n_bins=9):
        super(Attention_DHOGSA, self).__init__()
        self.factor = num_heads  # 注意力头数（用于特征填充适配）
        self.ifBox = ifBox  # 第一路径注意力重排模式（True/False对应不同维度拆分）
        self.num_heads = num_heads  # 注意力头数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 注意力温度因子（可学习）
        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)  # QKV生成（输出5路：q1/k1/q2/k2/v）
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1,
                                 groups=dim * 5, bias=bias)  # QKV深度卷积增强（局部依赖）
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)  # 注意力输出投影
        self.bin_proj = Conv2d(n_bins, dim // 2, kernel_size=1, bias=bias)  # HOG特征通道投影（适配输入维度）
        self.patch_size = patch_size  # HOG提取的patch尺寸
        self.n_bins = n_bins  # HOG方向直方图的bin数量（0~180°分9组，每组20°）

        # 固定Sobel算子（用于计算梯度，生成HOG特征）
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)  # 水平梯度
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)  # 垂直梯度
        self.register_buffer('sobel_x', sobel_x.repeat(dim, 1, 1, 1))  # 扩展至输入通道数
        self.register_buffer('sobel_y', sobel_y.repeat(dim, 1, 1, 1))

    def pad(self, x, factor):
        """
        特征填充：确保特征尺寸能被factor整除（适配注意力维度重排）
        
        Args:
            x: 输入特征，shape=[b, c, l]（l为展平后的像素数）
            factor: 整除因子（通常为注意力头数）
        
        Returns:
            x_padded: 填充后特征
            t_pad: 填充量 [0, pad_right]（用于后续反填充）
        """
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        """
        特征反填充：移除pad函数添加的填充部分，恢复原始尺寸
        
        Args:
            x: 已填充的特征
            t_pad: pad返回的填充量
        
        Returns:
            原始尺寸的特征
        """
        *_, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        """数值稳定的Softmax：使用PyTorch内置F.softmax避免溢出风险"""
        # 使用F.softmax的数值稳定实现 (内部使用log-sum-exp技巧)
        return F.softmax(x, dim=dim)

    def normalize(self, x):
        """特征归一化：基于最后两维的均值方差，提升数值稳定性"""
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        # 使用更大的epsilon来增强稳定性，避免除以极小数
        return (x - mu) / torch.sqrt(sigma + 1e-6)

    def reshape_attn(self, q, k, v, ifBox):
        """
        注意力维度重排与计算：适配双路径不同的维度拆分模式
        
        两种维度重排模式对应不同的特征分解方式，交替使用能够从不同角度捕捉全局依赖。
        
        Args:
            q: 查询特征，shape=[b, c, l]
            k: 键特征，shape=[b, c, l]
            v: 值特征，shape=[b, c, l]
            ifBox: 维度重排模式开关（True/False对应两种不同的重排方式）
        
        Returns:
            out: 注意力加权后的特征，shape=[b, c, l]
        """
        b, c = q.shape[:2]
        # 填充特征以适配维度拆分
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor  # 拆分后的特征长度

        # 维度重排公式（两种模式）
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        # 重排为注意力适配格式：[b, head, c×factor, hw]
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)

        # 注意力计算：使用数值稳定的方法
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 缩放点积注意力 + 温度因子调制
        attn = self.softmax_1(attn, dim=-1)  # 数值稳定的softmax（权重归一化）
        out = (attn @ v)  # 价值加权（注意力输出）

        # 维度还原：恢复原始特征形状
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)  # 移除填充部分
        return out

    def split_into_patches(self, x):
        """
        特征拆分为patch：适配HOG特征提取（按patch_size拆分）
        
        Args:
            x: 输入特征，shape=[b, c, h, w]
        
        Returns:
            patches: 拆分后的patch特征，shape=[b, n_patches, c, patch_pixels]
            shape_info: 原始形状信息元组 (b, c, h, w, pad_h, pad_w, n_h, n_w)（用于后续patch合并）
        """
        b, c, h, w = x.shape
        # 填充特征以适配patch_size拆分
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        # 维度重排拆分patch：[b, c, h, w]→[b, n_patches, c, patch_pixels]
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        n_h, n_w = (h + pad_h) // self.patch_size, (w + pad_w) // self.patch_size  # patch的行数/列数
        return patches, (b, c, h, w, pad_h, pad_w, n_h, n_w)

    def merge_patches(self, patches, shape_info):
        """
        patch合并：将拆分后的patch还原为原始特征尺寸
        
        Args:
            patches: 拆分后的patch特征，shape=[b, n_patches, c, patch_pixels]
            shape_info: split_into_patches返回的形状信息
        
        Returns:
            合并后的特征，shape=[b, c, h, w]（原始尺寸）
        """
        b, c, h, w, pad_h, pad_w, n_h, n_w = shape_info
        # 维度重排合并：[b, n_patches, c, patch_pixels]→[b, c, h, w]
        patches = rearrange(patches, 'b (h w) c (p1 p2) -> b c (h p1) (w p2)', h=n_h, w=n_w, p1=self.patch_size,
                            p2=self.patch_size)
        if pad_h > 0 or pad_w > 0:
            patches = patches[:, :, :h, :w]  # 移除填充
        return patches

    def apply_hog_to_patch(self, x_half):
        """
        对输入特征的前半通道提取HOG特征，并引导特征排序
        
        Args:
            x_half: 输入特征的前半通道，shape=[b, c, h, w]
        
        Returns:
            x_half_processed: HOG引导排序后的特征，shape=[b, c, h, w]
            sort_indices: 排序索引用于后续还原，shape=[b, c, 1, h*w]
            hog_features: HOG特征（每个patch的方向直方图），shape=[b, n_patches, n_bins]
            shape_info: patch拆分的形状信息，tuple
        
        注意：该方法使用Sobel算子计算梯度，通过方向直方图引导像素排序
        """
        b, c, h, w = x_half.shape
        device = x_half.device
        dtype = x_half.dtype
        
        # 1. 计算梯度幅值与方向（HOG核心）- 使用Sobel算子
        gx = F.conv2d(x_half, self.sobel_x[:c], padding=1, groups=c)  # 水平梯度
        gy = F.conv2d(x_half, self.sobel_y[:c], padding=1, groups=c)  # 垂直梯度
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)  # 梯度幅值（增加epsilon提升稳定性）
        orientation = torch.atan2(gy, gx)  # 梯度方向（[-pi, pi]）

        # 2. 方向映射为bin索引（0~n_bins-1）
        orientation_bin = ((orientation + torch.pi) / (2 * torch.pi) * self.n_bins).long() % self.n_bins

        # 3. 特征拆分为patch，按patch提取HOG
        patches_x, shape_info = self.split_into_patches(x_half)
        patches_mag, _ = self.split_into_patches(magnitude)
        patches_ori, _ = self.split_into_patches(orientation_bin.float())
        b, n_patches, c_patch, patch_pixels = patches_x.shape

        # 4. 计算每个patch的HOG特征（方向直方图）- 优化版本
        # 预分配张量避免每次迭代重新分配
        sort_values = torch.zeros_like(patches_x)
        hog_features = torch.zeros(b, n_patches, self.n_bins, device=device, dtype=dtype)
        
        # 按bin循环计算HOG直方图，同时累积排序值
        for i in range(self.n_bins):
            bin_mask = (patches_ori == i).float()  # 第i个方向bin的掩码
            bin_magnitude = patches_mag * bin_mask  # 该方向的梯度幅值
            sort_values += bin_magnitude * (i + 1)  # 累积排序值（用于后续像素排序）
            # 计算每个patch在该bin的平均幅值（HOG直方图分量）
            hog_features[..., i] = bin_magnitude.mean(dim=[-1, -2])  # shape: [b, n_patches]

        # 5. HOG特征归一化 - 提升数值稳定性
        hog_norm = hog_features.sum(dim=-1, keepdim=True) + 1e-8  # 分母加epsilon避免除零
        hog_features = hog_features / hog_norm  # 每个patch的HOG特征归一化

        # 6. 基于HOG梯度幅值排序patch内像素，强化结构特征
        # 按通道和patch的累积排序值对像素排序
        _, sort_indices = sort_values.sum(dim=2, keepdim=True).expand_as(patches_x).sort(dim=-1)
        patches_x_sorted = torch.gather(patches_x, -1, sort_indices)

        # 7. 合并patch，还原特征尺寸
        x_half_processed = self.merge_patches(patches_x_sorted, shape_info)
        return x_half_processed, sort_indices, hog_features, shape_info

    def forward(self, x):
        """
        前向传播：HOG特征引导→双路径注意力→特征还原
        
        Args:
            x: 输入特征，shape=[b, dim, h, w]
        
        Returns:
            增强后特征，shape=[b, dim, h, w]（与输入维度一致）
        
        计算流程：
            1. HOG特征提取与动态排序（前半通道）
            2. HOG特征投影与融合
            3. QKV生成与深度卷积增强
            4. 基于V梯度的动态排序
            5. 双路径自注意力计算
            6. 注意力输出融合
            7. 特征还原
        """
        b, c, h, w = x.shape
        half_c = c // 2  # 通道拆分：前半通道用于HOG引导，后半通道保留原始信息

        # 1. HOG特征提取与引导排序（仅作用于前半通道）
        x_half = x[:, :half_c]
        x_half_processed, idx_patch, hog_features, shape_info = self.apply_hog_to_patch(x_half)

        # 2. HOG特征投影与融合（将HOG直方图映射为特征图，与原始特征融合）
        b, n_patches, n_bins = hog_features.shape
        n_h, n_w = shape_info[-2], shape_info[-1]
        hog_map = rearrange(hog_features, 'b (nh nw) bins -> b bins nh nw', nh=n_h, nw=n_w).contiguous()  # HOG特征图
        hog_map = self.bin_proj(hog_map)  # 通道投影至half_c
        hog_map = F.interpolate(hog_map, size=(h, w), mode='bilinear')  # 上采样至原始尺寸
        x = torch.cat((x_half_processed + hog_map, x[:, half_c:]), dim=1)  # 融合HOG特征与后半通道

        # 3. QKV生成与深度卷积增强
        qkv = self.qkv_dwconv(self.qkv(x))  # 1×1卷积生成5路→3×3深度卷积增强
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # 拆分5路：q1/k1（路径1）、q2/k2（路径2）、v（共享价值）

        # 4. 基于V的梯度信息，动态排序Q/K/V（强化结构相关特征）
        gx = F.conv2d(v, self.sobel_x[:c], padding=1, groups=c)
        gy = F.conv2d(v, self.sobel_y[:c], padding=1, groups=c)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).view(b, c, -1)  # 梯度幅值（展平，epsilon增强稳定性）
        orientation = torch.atan2(gy, gx).view(b, c, -1)  # 梯度方向（展平）
        orientation_norm = ((orientation + torch.pi) / (2 * torch.pi))  # 方向归一化至[0,1]
        weighted_magnitude = magnitude * orientation_norm  # 幅值×方向加权，整合局部结构信息
        _, idx = weighted_magnitude.sum(dim=1).sort(dim=-1)  # 按加权值排序，得到索引
        idx = idx.unsqueeze(1).expand(b, c, -1)  # 扩展索引至通道维度

        # 5. 按排序索引重排Q1/K1/Q2/K2/V
        v = torch.gather(v.view(b, c, -1), dim=2, index=idx)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        # 6. 双路径注意力计算（不同维度重排模式）
        out1 = self.reshape_attn(q1, k1, v, True)  # 路径1：ifBox=True
        out2 = self.reshape_attn(q2, k2, v, False)  # 路径2：ifBox=False

        # 7. 注意力输出还原（按索引恢复原始顺序）
        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2  # 双路径特征融合（元素乘法）

        # 8. 输出投影与前半通道还原（恢复patch内像素原始顺序）
        out = self.project_out(out)
        out_replace = out[:, :half_c]
        patches_out, shape_info = self.split_into_patches(out_replace)
        
        # 使用scatter操作反演HOG排序索引，恢复patch内像素的原始顺序
        patches_out_restored = torch.zeros_like(patches_out)
        patches_out_restored.scatter_(-1, idx_patch, patches_out)
        patches_out = patches_out_restored
        
        out_replace = self.merge_patches(patches_out, shape_info)
        out[:, :half_c] = out_replace  # 替换前半通道

        return out


# 向后兼容性别名
DHOGSA = Attention_DHOGSA