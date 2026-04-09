# 波传播算子模块（Wave Propagation Operator, WPO）
# 核心设计：基于二维波动方程的物理启发模型，通过“特征增强→DCT频域转换→波动方程调制→IDCT时域还原”的流程，
# 模拟波的传播与衰减特性，在频域中自适应调制特征的频率成分，强化特征的全局传播与局部细节一致性，
# 实现物理规律引导的特征增强，提升模型对全局依赖与结构信息的捕捉能力

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import DropPath from timm, with fallback implementation
try:
    from timm.models.layers import DropPath  # type: ignore
except ImportError:
    # Fallback implementation of DropPath if timm is not installed
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample for residual connections."""
        def __init__(self, drop_prob=0., training=False):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob
            self.training = training

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.training or self.drop_prob == 0.:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
            if keep_prob > 0.0:
                # Use safer division than div_ for numerical stability
                random_tensor = random_tensor / keep_prob
            return x * random_tensor

        def extra_repr(self):
            return f"drop_prob={self.drop_prob}"

        def __repr__(self):
            return f"DropPath({self.drop_prob})"

# Set repr for imported or fallback DropPath
if hasattr(DropPath, '__repr__'):
    # Only override if not already set properly
    DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})" if hasattr(self, 'drop_prob') else "DropPath()"


class Wave2D(nn.Module):
    """2D Wave Equation Operator for physics-inspired feature transformation.
    
    A frequency-domain operator implementing the damped 2D wave equation:
        ∂²u/∂t² - c²(∂²u/∂x² + ∂²u/∂y²) + α∂u/∂t = 0
    
    With Dirichlet boundary conditions and closed-form solution via DCT.
    
    This operator transforms spatial features through wave propagation,
    capturing both global relationships and local details through frequency domain modulation.
    
    Mathematical Foundation:
        The solution in frequency domain is:
        u(x,y,t) = IDCT[A(n,m) * decay(n,m) * time_mod(n,m,t)]
        
        where:
        - A(n,m) = DCT coefficients of input features
        - decay(n,m) = exp(-[(nπ/a)² + (mπ/b)²]) (high-freq damping)
        - time_mod(n,m,t) = cos(ct) + sin(ct)/c * velocity_init (wave evolution)
    
    Design Features:
        - 物理启发调制：基于波动方程的频域衰减/增强核，模拟波传播特性
        - 自适应参数：可学习波速c与衰减系数α，适配不同任务特征分布
        - 特征预处理：深度卷积+线性变换强化输入特征表达
        - 推理优化：支持推理模式预计算调制核，提升推理效率
        - 分辨率自适应：自动缓存DCT基函数，支持多分辨率输入
        
    Args:
        dim: Input channel dimension. Model expects input of shape [B, dim, H, W]
        hidden_dim: Internal/output channel dimension after processing
        infer_mode: Whether to enable inference optimization mode (default: False).
                   When True, precomputes wave propagation kernels for fixed resolution.
        res: Reference resolution for caching DCT bases (default: 14).
             Used to determine initial cache size; actual resolution is inferred from input.
        **kwargs: Additional keyword arguments (for framework compatibility)
        
    Attributes:
        c: Learnable wave speed parameter (initialized to 1.0, learns rate of wave propagation)
        alpha: Learnable damping coefficient (initialized to 0.1, controls frequency decay rate)
        
    Caching Strategy:
        To avoid redundant computation, DCT/IDCT basis functions are cached per resolution:
        - __WEIGHT_COSN__, __WEIGHT_COSM__, __WEIGHT_EXP__: Forward DCT transforms
        - __WEIGHT_COSN_IDCT__, __WEIGHT_COSM_IDCT__: Inverse IDCT transforms
        - __RES__, __RES_IDCT__: Cached resolution tuples for validity checks
        - __device_cache__: Previous device to invalidate cache on device transfers
        
        Cache is automatically invalidated on:
        - Resolution change (H or W mismatch)
        - Device transfer (e.g., CPU ↔ GPU)
        
    Shape:
        Input:  [B, dim, H, W]
        Output: [B, hidden_dim, H, W]
    """

    def __init__(self, dim, hidden_dim=None, infer_mode=False, res=14, **kwargs):
        """Initialize Wave2D module with YOLO11 integration.
        
        Args:
            dim: Input channel dimension (required). For YOLO11, automatically injected by parse_model.
            
            hidden_dim: Output channel dimension. If None, defaults to dim value.
            
            infer_mode: Whether to enable inference optimization (precomputes wave kernels)
            
            res: Reference spatial resolution for DCT kernel caching (default: 14)
            
            **kwargs: Additional arguments (for framework compatibility)
            
        YOLO11 Integration:
            This module works with YOLO11's attention module auto-injection.
            In YAML, use empty args to let parser inject channels:
            - [-1, 2, Wave2D, []]  # Parser auto-injects input channels
            
            The parse_model function automatically:
            1. Gets input channels c2 from previous layer
            2. Calls Wave2D(c2) as m(*args) where args=[c2]
            3. Wave2D outputs same channels as input, enabling seamless integration
        """
        super().__init__()
        
        # If hidden_dim not specified, match it to dim (standard for feature processors)
        if hidden_dim is None:
            hidden_dim = dim
        
        self.res = res
        # Use pointwise convolution (groups=1) when channels differ, depthwise when they match
        # This avoids channel divisibility issues
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=1 if dim != hidden_dim else dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        self.c = nn.Parameter(torch.ones(1) * 1.0)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        self.save_attention = False
        # Apply parameter initialization
        self._init_weights()
        # Initialize cache tracking
        self._device_cache = None

    def _clear_cache(self) -> None:
        """Clear resolution-dependent caches (e.g., after device transfer or resolution change).
        
        This method should be called when:
        - Input resolution changes (H or W mismatch)
        - Model is transferred to a different device
        """
        for attr in ['__WEIGHT_COSN__', '__WEIGHT_COSM__', '__WEIGHT_EXP__',
                     '__WEIGHT_COSN_IDCT__', '__WEIGHT_COSM_IDCT__',
                     '__RES__', '__RES_IDCT__']:
            if hasattr(self, attr):
                delattr(self, attr)
        self._device_cache = None

    def _init_weights(self) -> None:
        """Initialize weights following WaveFormer conventions for stable training.
        
        Initialization Strategy:
            - Linear layers: Truncated normal (std=0.02) to maintain stable signal propagation
            - Conv1d/Conv2d: Kaiming normal (fan_out, relu) for DCT kernel operations
            - LayerNorm: Constant (weight=1.0, bias=0) for identity initialization
            
        This initialization scheme is critical for physical models to ensure:
            1. Stable wave propagation without exploding/vanishing gradients
            2. Proper frequency response in DCT-based operations
            3. Consistent multi-layer behavior in deep backbones
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use truncated normal for Linear layers
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                # Use Kaiming for convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _validate_input_shape(self, x: torch.Tensor, freq_embed: Optional[torch.Tensor] = None) -> None:
        """Validate input tensor shapes for dimensional compatibility and provide clear error messages.
        
        This method performs early validation to catch shape mismatches before expensive
        computations, providing descriptive error messages including actual vs expected shapes.
        
        Args:
            x: Input feature tensor expected to have shape [B, C, H, W]
            freq_embed: Optional frequency embedding, expected shape [B, H, W, C], [1, H, W, C],
                       or None. Supports batch dimension 1 for broadcasting.
            
        Raises:
            ValueError: With detailed message if tensor shapes are incompatible, including:
                - Actual vs expected shapes
                - Dimension counts
                - Broadcast requirements for freq_embed
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor [B, C, H, W], got shape {tuple(x.shape)} "
                f"with {x.ndim} dimensions"
            )
        B, C, H, W = x.shape
        
        if freq_embed is not None:
            if freq_embed.ndim == 4:
                B_f, H_f, W_f, C_f = freq_embed.shape
                if (B_f != B or H_f != H or W_f != W) and B_f != 1:
                    raise ValueError(
                        f"freq_embed shape {tuple(freq_embed.shape)} incompatible with input "
                        f"[B={B}, H={H}, W={W}, C={C}]. Expected [B, H, W, C] or [1, H, W, C]"
                    )
                if C_f != self.hidden_dim:
                    raise ValueError(
                        f"freq_embed channel dimension {C_f} must match hidden_dim {self.hidden_dim}. "
                        f"Got freq_embed shape {tuple(freq_embed.shape)}, expected [*, *, *, {self.hidden_dim}]"
                    )
            else:
                raise ValueError(
                    f"freq_embed must be 4D tensor [B, H, W, C], got shape {tuple(freq_embed.shape)} "
                    f"with {freq_embed.ndim} dimensions"
                )

    def infer_init_wave2d(self, freq: torch.Tensor) -> None:
        """Initialize inference-optimized wave operator with precomputed kernels.
        
        In inference mode, this precomputes frequency-dependent modulation kernels
        to avoid redundant computation when doing multiple forward passes at fixed resolution.
        
        Args:
            freq: Frequency embedding tensor of shape [H, W, hidden_dim] used to 
                  parametrize wave propagation kernels
                  
        Side Effects:
            - Stores precomputed k_exp as non-trainable parameter  
            - Deletes self.to_k to save memory (no longer needed)
        """
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N: int = 224, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float) -> torch.Tensor:
        """Generate DCT/IDCT cosine mapping matrices (core discretization for physical model).
        
        Implements the DCT-II basis functions: cos((x + 0.5) / N * n * π)
        
        The cosine map is used in both forward (DCT) and inverse (IDCT) transforms:
        - DCT:  F(n) = sum( c(n) * cos((x + 0.5) / N * n * π) * f(x) )
        - IDCT: f(x) = sum( c(n) * cos((x + 0.5) / N * n * π) * F(n) )
        
        where c(n) = sqrt(2/N) for n > 0, else sqrt(1/N)
        
        Args:
            N: Size of the cosine map (default 224)
            device: Torch device for tensor allocation (default CPU)
            dtype: Tensor data type (default float32)
            
        Returns:
            Orthonormal cosine basis matrix of shape [N, N]
            where first dimension represents frequency components and second spatial positions
        """
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution: Tuple[int, int] = (224, 224), device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float) -> torch.Tensor:
        """Generate frequency-dependent decay mapping for wave propagation.
        
        Implements the exponential decay term from the damped wave equation:
        decay = exp(-[(nπ/a)² + (mπ/b)²])
        
        This corresponds to the wave equation solution with boundary conditions:
        ∂u/∂x|_{x=0,a} = 0, ∂u/∂y|_{y=0,b} = 0
        
        High-frequency components decay more rapidly, which regularizes the solution
        and captures multi-scale feature interactions.
        
        Args:
            resolution: Spatial resolution tuple (H, W) (default (224, 224))
            device: Torch device for tensor allocation (default CPU)
            dtype: Tensor data type (default float32)
            
        Returns:
            Decay weight matrix of shape [H, W] with exponentially decayed frequency components
        """
        # (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt)
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        # Quadratic term for wave equation
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed: Optional[torch.Tensor] = None, test_index: Optional[int] = None) -> torch.Tensor:
        """Forward pass: feature preprocessing → frequency transform → wave modulation → spatial restoration → gating.
        
        Implements a physics-inspired operator that solves the 2D damped wave equation in a learnable way.
        
        Args:
            x: Input feature tensor of shape [B, C, H, W]
                - B: batch size
                - C: input channels
                - H, W: spatial dimensions (height, width)
                
            freq_embed: Optional frequency embedding of shape [B|1, H, W, C].
                Defaults to None, where it's initialized to zeros.
                If shape[0]==1, it's automatically broadcast to batch size B.
                Controls temporal dynamics through learned parameters c (wave speed) and alpha (damping).
                
            test_index: Integer index for attention visualization (debug mode only).
                Only used when save_attention=True.
            
        Returns:
            Output tensor of shape [B, C, H, W] (same layout as input).
            
        Raises:
            ValueError: If input tensor shapes are incompatible.
        """
        # Validate input shapes
        self._validate_input_shape(x, freq_embed)
        
        # Detect device transfer and clear cache if needed
        current_device = x.device
        if self._device_cache is not None and self._device_cache != current_device:
            self._clear_cache()
        self._device_cache = current_device
        
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)
        C = self.hidden_dim  # Update C to actual channel count after linear layer

        # === Forward DCT: Frequency Domain Transformation ===
        # Cache DCT basis functions (orthonormal cosine matrices) per resolution
        # This avoids recomputing them for every forward pass at fixed resolution
        cached_weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and cached_weight_cosn is not None and (
                cached_weight_cosn.device == x.device) and cached_weight_cosn.dtype == x.dtype:
            # Cache hit: reuse precomputed DCT bases (device and dtype match)
            weight_cosn = cached_weight_cosn
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            # Cache miss: compute and store DCT bases with input dtype for mixed precision support
            weight_cosn = self.get_cos_map(H, device=x.device, dtype=x.dtype).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device, dtype=x.dtype).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device, dtype=x.dtype).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]
        weight_cosn_kernel = weight_cosn.view(H, 1, H)
        weight_cosm_kernel = weight_cosm.view(W, 1, W)
        x_perm = x.permute(0, 3, 2, 1).contiguous()  # [B, C, W, H]
        x_flat_H = x_perm.view(-1, 1, H)  # [B*C*W, 1, H]
        x_u0 = F.conv1d(x_flat_H, weight_cosn_kernel).squeeze(-1)  # [B*C*W, H]
        x_u0 = x_u0.view(B, C, W, H).permute(0, 3, 2, 1).contiguous()  # [B, H, W, C]
        x_perm = x_u0.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x_flat_W = x_perm.view(-1, 1, W)  # [B*C*H, 1, W]
        x_u0 = F.conv1d(x_flat_W, weight_cosm_kernel).squeeze(-1)  # [B*C*H, W]
        x_u0 = x_u0.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()
        x_perm = x.permute(0, 3, 2, 1).contiguous()  # [B, C, W, H]
        x_flat_H = x_perm.view(-1, 1, H)  # [B*C*W, 1, H]
        x_v0 = F.conv1d(x_flat_H, weight_cosn_kernel).squeeze(-1)  # [B*C*W, H]
        x_v0 = x_v0.view(B, C, W, H).permute(0, 3, 2, 1).contiguous()  # [B, H, W, C]
        x_perm = x_v0.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x_flat_W = x_perm.view(-1, 1, W)  # [B*C*H, 1, W]
        x_v0 = F.conv1d(x_flat_W, weight_cosm_kernel).squeeze(-1)  # [B*C*H, W]
        x_v0 = x_v0.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()
        if freq_embed is None:
            freq_embed = torch.zeros(B, H, W, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            # Handle broadcasting if freq_embed has batch dimension 1
            if freq_embed.shape[0] == 1 and B > 1:
                freq_embed = freq_embed.expand(B, -1, -1, -1)
        
        t = self.to_k(freq_embed)
        c_t = self.c * t
        cos_term = torch.cos(c_t)
        
        # Numerical stability: clamp wave speed to avoid near-zero division
        c_safe = torch.clamp(self.c, min=1e-8)
        sin_term = torch.sin(c_t) / c_safe
        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        final_term = wave_term + velocity_term
        
        # === Inverse DCT: Spatial Domain Restoration ===
        # Cache IDCT basis functions for the same resolution 
        # (uses transposed DCT bases, so cached separately)
        cached_weight_cosn_idct = getattr(self, "__WEIGHT_COSN_IDCT__", None)
        cached_weight_cosm_idct = getattr(self, "__WEIGHT_COSM_IDCT__", None)

        if ((H, W) == getattr(self, "__RES_IDCT__", (0, 0))) and \
                cached_weight_cosn_idct is not None and \
                cached_weight_cosn_idct.device == x.device and \
                cached_weight_cosn_idct.dtype == x.dtype:
            # Cache hit: reuse precomputed IDCT bases (device and dtype match)
            weight_cosn = cached_weight_cosn_idct
            weight_cosm = cached_weight_cosm_idct
        else:
            # Cache miss: compute and store IDCT bases with input dtype for mixed precision support
            weight_cosn = self.get_cos_map(H, device=x.device, dtype=x.dtype).detach_()  # (H, H)
            weight_cosm = self.get_cos_map(W, device=x.device, dtype=x.dtype).detach_()  # (W, W)
            setattr(self, "__RES_IDCT__", (H, W))
            setattr(self, "__WEIGHT_COSN_IDCT__", weight_cosn)
            setattr(self, "__WEIGHT_COSM_IDCT__", weight_cosm)
        x_w = final_term.permute(0, 1, 3, 2).contiguous().view(B * H * C, 1, W)  # (B*H*C, 1, W)
        weight_cosm_kernel_t = weight_cosm.t().contiguous().view(W, 1, W)  # (W,1,W)
        x_w = F.conv1d(x_w, weight_cosm_kernel_t).squeeze(-1)  # (B*H*C, W)
        x_w = x_w.view(B, H, C, W).permute(0, 1, 3, 2).contiguous()  # (B,H,W,C)
        x_h = x_w.permute(0, 2, 3, 1).contiguous().view(B * W * C, 1, H)  # (B*W*C,1,H)
        weight_cosn_kernel_t = weight_cosn.t().contiguous().view(H, 1, H)  # (H,1,H)
        x_h = F.conv1d(x_h, weight_cosn_kernel_t).squeeze(-1)  # (B*W*C,H)
        x_final = x_h.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()  # (B, C, H, W)
        # Apply gating and output projection
        # Ensure proper tensor layout: (B, H, W, C) for norm and linear operations
        assert x_final.shape == (B, C, H, W), f"Expected x_final shape (B={B}, C={C}, H={H}, W={W}), got {x_final.shape}"
        assert z.shape == (B, H, W, C), f"Expected z shape (B={B}, H={H}, W={W}, C={C}), got {z.shape}"
        
        x = x_final.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        assert x.shape == (B, H, W, C), f"After permute, expected (B={B}, H={H}, W={W}, C={C}), got {x.shape}"
        
        x = self.out_norm(x)  # Apply norm on channel-last format
        gate = nn.functional.silu(z)  # z is already (B, H, W, C)
        x_gated = x * gate  # Element-wise multiplication (B, H, W, C)
        x = self.out_linear(x_gated)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # Back to (B, C, H, W)
        assert x.shape == (B, C, H, W), f"Expected output shape (B={B}, C={C}, H={H}, W={W}), got {x.shape}"
        if test_index is not None and hasattr(self, 'save_attention') and self.save_attention:
            center_h, center_w = H // 2, W // 2
            attention_map = (x_final * x_final[:, center_h:center_h + 1, center_w:center_w + 1, :]).sum(-1)

            import matplotlib.pyplot as plt

            # Ensure the save directory exists
            save_dir = "./save/attention_map"
            os.makedirs(save_dir, exist_ok=True)

            # Move attention_map to cpu and convert to numpy
            att_map_np = attention_map.detach().cpu().numpy()

            # Normalize for visualization with numerical stability
            att_map_min = att_map_np.min()
            att_map_max = att_map_np.max()
            att_map_range = att_map_max - att_map_min
            if att_map_range > 1e-8:
                att_map_norm = (att_map_np - att_map_min) / att_map_range
            else:
                att_map_norm = att_map_np - att_map_min  # Avoid division by near-zero

            # Save each sample in the batch
            for i in range(att_map_norm.shape[0]):
                filename = os.path.join(save_dir, f"attention_map_{test_index}_{i}.png")
                plt.imsave(filename, att_map_norm[i], cmap='viridis')

        return x

    def get_parameter_summary(self) -> str:
        """Get a summary of learnable wave parameters.
        
        Returns:
            String summary of wave speed (c) and damping coefficient (alpha)
        """
        return f"Wave2D(c={self.c.item():.4f}, alpha={self.alpha.item():.4f})"

    def get_cache_stats(self) -> dict:
        """Get cache statistics for performance monitoring and debugging.
        
        Returns:
            Dictionary containing:
            - 'cached_resolutions': List of cached (H, W) tuples
            - 'cache_size_mb': Approximate memory usage in MB
            - 'device': Device where cached tensors are stored
        """
        cache_info = {
            'cached_resolutions': [],
            'cache_size_mb': 0.0,
            'device': None,
        }
        
        if hasattr(self, '__RES__'):
            cache_info['cached_resolutions'].append(self.__RES__)
            if hasattr(self, '__WEIGHT_COSN__'):
                size_bytes = (self.__WEIGHT_COSN__.numel() * self.__WEIGHT_COSN__.element_size())
                cache_info['cache_size_mb'] += size_bytes / (1024 * 1024)
                cache_info['device'] = str(self.__WEIGHT_COSN__.device)
        
        if hasattr(self, '__RES_IDCT__'):
            cache_info['cached_resolutions'].append(self.__RES_IDCT__)
        
        return cache_info