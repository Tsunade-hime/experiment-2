# 多头线性注意力模块（Multi-Head Linear Attention, MHLA）
# 核心设计：整合“块距离卷积+多头线性投影+轻量化注意力”的流程，
# 通过块距离引导的1×1卷积捕捉空间依赖，多头并行提升特征表达多样性，
# 线性复杂度注意力计算（O(N)）替代传统自注意力（O(N²)），在保证全局空间关联捕捉能力的同时，
# 大幅降低计算成本，适配高分辨率特征处理


import math
from typing import Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from torch import nn


class BlockDistanceConv(nn.Module):
    """
    块距离卷积模块：基于空间块距离的1×1卷积，捕捉全局空间依赖
    核心设计：
        - 块划分策略：将特征图按patch分组为块，计算块中心欧氏距离，生成距离矩阵
        - 距离变换函数：支持线性/余弦/指数/高斯/局部等多种距离-权重映射，适配不同空间依赖模式
        - 固定权重卷积：卷积权重由距离矩阵生成，无需训练，降低过拟合风险
        - 轻量化设计：1×1卷积结构，计算成本低，可快速捕捉全局空间关联
    Args:
        num_patches_per_side: 特征图每边的patch数量（如16表示16×16个patch）
        patch_group_size: 每个块包含的patch数量（默认16，对应4×4个patch组成一个块）
        transform: 距离变换函数（可选'linear'/'cos'/'exp'/'gaussian'/'local'）
        local_thres: 局部模式下的距离阈值（默认1.5，小于等于该值的块视为局部关联）
        exp_sigma: 指数变换的衰减系数（默认3，控制衰减速度）
    """
    def __init__(
            self, num_patches_per_side=16, patch_group_size=16, transform="linear", local_thres=1.5, exp_sigma=3
    ):
        """
        Args:
            num_patches_per_side: Number of patches per side (e.g., 16 for 16x16 patches)
            patch_group_size: Number of patches in each block (default: 16)
            transform: Transform function to apply to distances ('linear', 'cos', 'exp', 'gaussian')
        """
        super().__init__()

        self.num_patches_per_side = num_patches_per_side
        self.patch_group_size = patch_group_size
        self.transform = transform
        self.local_thres = local_thres  # Threshold for local connections, can be adjusted
        self.exp_sigma = exp_sigma

        # Calculate number of blocks per side
        patches_per_block_side = int(np.sqrt(patch_group_size))  # 4 for 16 patches
        self.blocks_per_side = (
                num_patches_per_side // patches_per_block_side
        )  # 4 for 16x16 patches
        self.total_blocks = self.blocks_per_side ** 2  # 16 blocks

        # Create distance matrix
        distance_matrix = self._compute_block_distances()

        # Apply transformation
        weight_matrix = self._apply_transform(distance_matrix)

        # Create 1x1 conv layer
        self.conv = nn.Conv2d(
            in_channels=self.total_blocks,
            out_channels=self.total_blocks,
            kernel_size=1,
            bias=False,
        )

        # Set the weights as fixed (no gradient)
        with torch.no_grad():
            # Weight shape for Conv2d: (out_channels, in_channels, kernel_h, kernel_w)
            # For 1x1 conv: (16, 16, 1, 1)
            self.conv.weight.data = weight_matrix.unsqueeze(-1).unsqueeze(-1)

        # Freeze the weights
        # self.conv.weight.requires_grad = False

    def _compute_block_distances(self):
        """Compute Euclidean distances between all block centers."""
        # Get block center coordinates
        block_centers = []
        for i in range(self.blocks_per_side):
            for j in range(self.blocks_per_side):
                # Center of block in grid coordinates
                center_x = i + 0.5
                center_y = j + 0.5
                block_centers.append([center_x, center_y])

        block_centers = torch.tensor(block_centers, dtype=torch.float32)

        # Compute pairwise distances
        # distance_matrix[i, j] = distance from block i to block j
        distance_matrix = torch.zeros(self.total_blocks, self.total_blocks)

        for i in range(self.total_blocks):
            for j in range(self.total_blocks):
                dist = torch.norm(block_centers[i] - block_centers[j], p=2)
                distance_matrix[i, j] = dist

        return distance_matrix

    def _apply_transform(self, distance_matrix):
        """Apply transformation function to distance matrix."""
        if self.transform == "linear":
            # Normalize to [0, 1] and invert (closer blocks have higher weights)
            max_dist = distance_matrix.max()
            mat = 1.0 - (distance_matrix / max_dist)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "cos":
            # Cosine transformation
            max_dist = distance_matrix.max()
            normalized_dist = distance_matrix / max_dist * math.pi / 4
            mat = torch.cos(normalized_dist)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "exp":
            # Exponential decay
            sigma = distance_matrix.max() / 3  # Adjust decay rate
            mat = torch.exp(-distance_matrix / self.exp_sigma)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "gaussian":
            # Gaussian kernel
            sigma = distance_matrix.max() / 3
            return torch.exp(-(distance_matrix ** 2) / (2 * sigma ** 2))

        elif self.transform == "local":
            thres = getattr(
                self, "local_thres", 1.5
            )  # 可通过 self.local_thres 控制阈值
            mat = (distance_matrix <= thres).float()
            mat = mat / mat.sum(dim=0, keepdim=True)
            return mat

        else:
            raise ValueError(f"Unknown transform: {self.transform}")

    def forward(self, x):
        """
        Forward pass through the distance-based convolution.

        Args:
            x: Input tensor of shape (B, 16, H, W) where 16 is the number of blocks

        Returns:
            Output tensor of shape (B, 16, H, W)
        """
        return self.conv(x)

    def get_weight_matrix(self):
        """Return the weight matrix for inspection."""
        return self.conv.weight.data.squeeze(-1).squeeze(-1)


class MHLA_Normed_Torch(nn.Module):
    """
    多头线性注意力模块（Multi-Head Linear Attention, MHLA）
    功能：块距离卷积+多头线性投影+轻量化注意力，高效捕捉全局-局部特征依赖
    核心设计：
        - 块距离空间引导：BlockDistanceConv捕捉全局空间依赖，替代传统自注意力的空间关联计算
        - 多头并行增强：多组线性投影并行处理，提升特征表达多样性
        - 线性复杂度计算：无高维矩阵乘法，注意力计算复杂度为O(N)，适配高分辨率
        - 残差融合：保留原始特征信息，避免过度增强导致的细节丢失
    Args:
        dim: 输入/输出通道数
        num_heads: 注意力头数（默认8，需整除dim）
        num_patches_per_side: 每边patch数量（默认16，传递给BlockDistanceConv）
        patch_group_size: 每个块的patch数（默认16，传递给BlockDistanceConv）
        transform: 距离变换函数（默认'linear'，传递给BlockDistanceConv）
        bias: 线性层是否带偏置（默认False）
        proj_drop: 输出dropout概率（默认0.）
    """
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        dropout: float = 0.1,
        fixed_weight_value: Optional[float] = None,
        qk_norm: bool = False,
        transform: str = "cos",
        **kwargs,
    ):
        super(MHLA_Normed_Torch, self).__init__()

        if dim_head is None:
            assert (
                dim % heads == 0
            ), f"dim ({dim}) must be divisible by heads ({heads}) for MHLA_Normed_Torch."
            dim_head = dim // heads
        inner_dim = dim_head * heads
        self.num_heads = heads
        self.head_dim = dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        is_bias = kwargs.get("qkv_bias", False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=is_bias)

        self.q_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()

        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)

        # These are compatibility defaults. Real geometry is passed per-forward from wrapper.
        self.window_size = kwargs.get("window_size", 49)
        self.embed_len = kwargs.get("embed_len", 196)
        self.num_pieces = max(1, self.embed_len // max(self.window_size, 1))
        local_thres = kwargs.get("local_thres", 1.5)
        exp_sigma = kwargs.get("exp_sigma", 3)

        # cache for dynamically computed block distance weights
        self.transform = transform
        self.local_thres = local_thres
        self.exp_sigma = exp_sigma
        self._block_weight_cache = {}

        self.eps = kwargs.get("eps", 1e-6)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # # 如果指定了固定权重值，则初始化所有权重为该值
        if fixed_weight_value is not None:
            self._init_weights_with_fixed_value(fixed_weight_value)

        # print("✅ Piece Attention已编译")
        # print("✅ QKV处理和reshape已编译")

    def _init_weights_with_fixed_value(self, value):
        """将模型中的所有权重初始化为固定值"""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)

        # 特别处理一些层
        nn.init.constant_(self.to_qkv.weight, value)

        # 如果to_out是Sequential，需要单独处理其中的Linear层
        for module in self.to_out:
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, value)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def init_to_value(model, value=1.0):
        """静态方法，用于将现有模型的权重初始化为固定值"""
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)
        return model

    # @torch.compile
    def _process_qkv_impl(self, q, k, v, B, N, H, D):

        q = self.q_norm(q)  # [B, H, N, D]
        k = self.k_norm(k)  # [B, H, N, D]

        k = torch.relu(k) + self.eps
        q = torch.relu(q) + self.eps

        q, k, v = map(
            lambda t: rearrange(
                t, "b n w (h d) -> (b h) n w d", h=H, d=D
            ),
            (q, k, v)
        )

        k = k.transpose(-2, -1)

        return q, k, v

    # @torch.compile
    def _mlp_lepe(self, x, grid_hw: Tuple[int, int], window_hw: Tuple[int, int]):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        gh, gw = grid_hw
        wh, ww = window_hw
        lepe = self.lepe(
            rearrange(
                v,
                "b (h w) (p1 p2) d -> b d (h p1) (w p2)",
                h=gh,
                w=gw,
                p1=wh,
                p2=ww,
            )
        )
        lepe = rearrange(
            lepe,
            "b d (h p1) (w p2) -> b (h w) (p1 p2) d",
            h=gh,
            w=gw,
            p1=wh,
            p2=ww,
        )
        return q, k, v, lepe

    def _compute_block_weights(self, grid_hw: Tuple[int, int], device, dtype):
        """
        Compute (or fetch from cache) a [num_pieces, num_pieces] distance-based weight matrix
        over the window blocks. This is a purely functional operator and deliberately does not
        register any parameters or buffers, so EMA state remains shape-stable.
        """
        gh, gw = grid_hw
        num_pieces = gh * gw
        key = (gh, gw, self.transform, self.local_thres, self.exp_sigma, device, dtype)
        if key in self._block_weight_cache:
            return self._block_weight_cache[key]

        # block centers on a pieces_len x pieces_len grid
        coords = []
        for i in range(gh):
            for j in range(gw):
                coords.append([i + 0.5, j + 0.5])
        block_centers = torch.tensor(coords, dtype=torch.float32, device=device)

        # pairwise distances
        diff = block_centers.unsqueeze(1) - block_centers.unsqueeze(0)
        distance_matrix = torch.norm(diff, dim=-1)  # [num_pieces, num_pieces]

        if self.transform == "linear":
            max_dist = distance_matrix.max()
            mat = 1.0 - (distance_matrix / max_dist)
            mat = mat / mat.sum(dim=0, keepdim=True)
        elif self.transform == "cos":
            max_dist = distance_matrix.max()
            normalized_dist = distance_matrix / max_dist * math.pi / 4
            mat = torch.cos(normalized_dist)
            mat = mat / mat.sum(dim=0, keepdim=True)
        elif self.transform == "exp":
            mat = torch.exp(-distance_matrix / self.exp_sigma)
            mat = mat / mat.sum(dim=0, keepdim=True)
        elif self.transform == "gaussian":
            sigma = distance_matrix.max() / 3
            mat = torch.exp(-(distance_matrix ** 2) / (2 * sigma ** 2))
        elif self.transform == "local":
            thres = getattr(self, "local_thres", 1.5)
            mat = (distance_matrix <= thres).float()
            mat = mat / mat.sum(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown transform: {self.transform}")

        mat = mat.to(device=device, dtype=dtype)
        self._block_weight_cache[key] = mat
        return mat

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Optional[Tuple[int, int]] = None,
        window_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        x = self.norm(x)
        B, N, W, C = x.shape
        H = self.num_heads
        D = self.head_dim
        if grid_hw is None:
            side = int(N ** 0.5)
            if side * side != N:
                raise ValueError("MHLA_Normed_Torch requires grid_hw for non-square window grids.")
            grid_hw = (side, side)
        if window_hw is None:
            side = int(W ** 0.5)
            if side * side != W:
                raise ValueError("MHLA_Normed_Torch requires window_hw for non-square windows.")
            window_hw = (side, side)

        q, k, v, lepe = self._mlp_lepe(x, grid_hw=grid_hw, window_hw=window_hw)

        q, k, v = self._process_qkv_impl(q, k, v, B, N, H, D)

        kv = torch.matmul(k, v)  # [B*H, num_pieces, D, D]

        # apply distance-based mixing over the block dimension (num_pieces)
        weight = self._compute_block_weights(grid_hw=grid_hw, device=kv.device, dtype=kv.dtype)  # [N, N]
        kv = torch.einsum("ij, b j x y -> b i x y", weight, kv)  # [B*H, num_pieces, D, D]

        k_sum = k.sum(dim=-1, keepdim=True)  # [B*H, num_pieces, D, 1]
        qk_sum = torch.matmul(q, k_sum)  # [B*H, num_pieces, window_size, 1]
        normalizer = torch.einsum("ij, b j x y -> b i x y", weight, qk_sum) + self.eps  # [B*H, num_pieces, window_size, 1]

        out = torch.matmul(q, kv) / normalizer  # [B*H, num_pieces, window_size, D]
        # out = torch.matmul(q, kv) * self.scale
        # out = rearrange(out, "b n w d -> b (n w) d")
        out = rearrange(out, "(b h) n w d -> b n w (h d)", b=B, h=self.num_heads)
        # out = out * self.scale
        out = out + lepe

        return self.to_out(out)


class MHLA(nn.Module):
    """
    Ultralytics-friendly MHLA wrapper.

    This module accepts/returns BCHW tensors and internally reshapes the feature map
    into (num_windows, window_tokens) for MHLA_Normed_Torch. The core is initialized
    once in __init__ and is spatial-shape agnostic.
    """

    def __init__(
        self,
        c: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        dropout: float = 0.1,
        qk_norm: bool = False,
        transform: str = "cos",
        local_thres: float = 1.5,
        exp_sigma: float = 3.0,
        eps: float = 1e-6,
        qkv_bias: bool = False,
    ):
        """
        Args:
            c: Input/output channels.
            heads: Number of attention heads.
            dim_head: Per-head dimensionality. If None, uses c // heads.
            dropout: Dropout probability on output projection.
            qk_norm: Whether to apply RMSNorm to q and k.
            transform: Distance transform for BlockDistanceConv.
            local_thres, exp_sigma, eps: MHLA_Normed_Torch numerical hyperparameters.
            qkv_bias: Whether qkv projection uses bias.
        """
        super().__init__()
        self.c = c
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.transform = transform
        self.local_thres = local_thres
        self.exp_sigma = exp_sigma
        self.eps = eps
        self.qkv_bias = qkv_bias

        # Core attention is built once and reused for any spatial shape.
        self.core = MHLA_Normed_Torch(
            dim=self.c,
            heads=self.heads,
            dim_head=self.dim_head,
            dropout=self.dropout,
            qk_norm=self.qk_norm,
            transform=self.transform,
            window_size=49,  # compatibility default; real window size is per-forward
            embed_len=196,   # compatibility default; real embed geometry is per-forward
            local_thres=self.local_thres,
            exp_sigma=self.exp_sigma,
            eps=self.eps,
            qkv_bias=self.qkv_bias,
        )
        self.target_window = 8

    @staticmethod
    def _choose_window_dim(size: int, target: int = 8) -> int:
        """Choose a divisor of size close to target (>=4 when possible)."""
        candidates = [d for d in range(1, size + 1) if size % d == 0]
        preferred = [d for d in candidates if d >= 4]
        pool = preferred if preferred else candidates
        return min(pool, key=lambda d: (abs(d - target), -d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).
        Returns:
            Tensor of shape (B, C, H, W).
        """
        b, c, h, w = x.shape
        if c != self.c:
            raise ValueError(
                f"MHLA channel mismatch: init c={self.c}, but got input with c={c}."
            )

        # Keep core parameter dtype stable.
        # Under AMP, input tensor dtype may be fp16/bf16, but Ultralytics' GradScaler expects
        # model parameters (and therefore their gradients) to remain fp32 for unscaling.
        core_device = next(self.core.parameters()).device
        if core_device != x.device:
            self.core = self.core.to(device=x.device)

        wh = self._choose_window_dim(h, self.target_window)
        ww = self._choose_window_dim(w, self.target_window)
        assert h % wh == 0 and w % ww == 0, "H and W must be divisible by selected MHLA windows."

        gh, gw = h // wh, w // ww
        window_size = wh * ww

        # BCHW -> B (gh*gw) (wh*ww) C
        x_reshaped = x.view(b, c, gh, wh, gw, ww)
        x_reshaped = x_reshaped.permute(0, 2, 4, 3, 5, 1).contiguous()
        x_tokens = x_reshaped.reshape(b, gh * gw, window_size, c)

        out_tokens = self.core(x_tokens, grid_hw=(gh, gw), window_hw=(wh, ww))

        # B (gh*gw) (wh*ww) C -> BCHW
        out = out_tokens.view(b, gh, gw, wh, ww, c)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(b, c, h, w)

        return out