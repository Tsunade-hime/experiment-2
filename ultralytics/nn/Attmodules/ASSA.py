# Adaptive Sparse Self-Attention (ASSA) Module
# 自适应稀疏自注意力模块（Adaptive Sparse Self-Attention, ASSA）
# 核心设计：整合"动态深度卷积+自适应稀疏策略+训练-测试双模式"，通过"特征投影→动态增强→稀疏注意力计算→特征还原"的流程，
# 自适应控制注意力稀疏度，在训练时探索全局依赖，测试时通过局部窗口（TLC）提升推理效率，平衡模型性能与部署效率

import math
import numbers
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_

# Try to import IDynamicDWConv, with fallback implementation
try:
    from .idynamic_dwconv import IDynamicDWConv
except (ImportError, ModuleNotFoundError):
    # Fallback: Standard depthwise convolution with dynamic gating
    class IDynamicDWConv(nn.Module):
        """Fallback Depthwise Convolution with simple feature enhancement.
        
        Args:
            in_channels: Number of input channels
            kernel_size: Kernel size for depthwise convolution
            bias: Whether to use bias
        """
        def __init__(self, in_channels: int, kernel_size: int = 3, bias: bool = False) -> None:
            super(IDynamicDWConv, self).__init__()
            padding = kernel_size // 2
            self.dwconv = nn.Conv2d(
                in_channels, in_channels, 
                kernel_size=kernel_size, 
                padding=padding, 
                groups=in_channels, 
                bias=bias
            )
            self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply depthwise convolution with gating mechanism."""
            out = self.dwconv(x)
            gate = torch.sigmoid(self.gate(x))
            return out * gate


# ============================================================
# Layer Normalization Utilities
# ============================================================

def to_3d(x: torch.Tensor) -> torch.Tensor:
    """Convert 4D tensor (B, C, H, W) to 3D (B, H*W, C)."""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Convert 3D tensor (B, H*W, C) to 4D (B, C, H, W)."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    """Layer Normalization with learnable bias and weight.
    
    Args:
        normalized_shape: Shape of the normalized dimension
    """
    def __init__(self, normalized_shape: int) -> None:
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1, "Only 1D normalization supported"
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization with learnable affine parameters."""
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Layer Normalization wrapper supporting 4D tensor input.
    
    Args:
        dim: Number of channels
        LayerNorm_type: Type of layer norm to use
    """
    def __init__(self, dim: int, LayerNorm_type: str = 'WithBias') -> None:
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'WithBias':
            self.body = WithBias_LayerNorm(dim)
        else:
            self.body = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer norm to 4D tensor (B, C, H, W)."""
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ============================================================
# Activation Functions for Sparsity Control
# ============================================================

class MaskedSoftmax(nn.Module):
    """Softmax that masks out negative values (zeros after softmax)."""
    
    def __init__(self) -> None:
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softmax and mask negative values to zero."""
        mask = x > 0
        x = self.softmax(x)
        x = torch.where(mask > 0, x, torch.zeros_like(x))
        return x


class TopK(nn.Module):
    """Top-K selection for sparse attention."""
    
    def __init__(self) -> None:
        super(TopK, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select top 1/4 of attention values."""
        b, h, C, _ = x.shape
        mask = torch.zeros(b, h, C, C, device=x.device, requires_grad=False)
        index = torch.topk(x, k=int(C / 4), dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        result = torch.where(mask > 0, x, torch.zeros_like(x))
        return result


# ============================================================
# Feed-Forward Network (Restormer-style GDFN)
# ============================================================

class FeedForward(nn.Module):
    """Feed-Forward Network with gated linear unit style (GDFN from Restormer).
    
    Args:
        dim: Input/output dimension
        ffn_expansion_factor: Expansion factor for hidden dimension
        bias: Whether to use bias in convolutions
    """
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.0, bias: bool = False) -> None:
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, 
            kernel_size=3, stride=1, padding=1, 
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network with gating."""
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ============================================================
# Sparse Self-Attention
# ============================================================

class SparseSelfAttention(nn.Module):
    """Adaptive Sparse Self-Attention with TLC (Test-time Local Cropping).
    
    Combines dynamic feature enhancement with sparse attention mechanisms to balance
    global dependency capture and computational efficiency.
    
    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
        bias: Whether to use bias in projections
        tlc_flag: Enable TLC for test-time efficiency
        tlc_kernel: Kernel size for local windows in TLC
        activation: Sparsity control activation ('relu', 'softmax', 'maskedsoftmax', 'topk', 'gelu', 'sigmoid')
    """
    
    # Activation mapping for cleaner code
    _ACTIVATIONS: Dict[str, nn.Module] = {
        'relu': nn.ReLU(),
        'softmax': nn.Softmax(dim=-1),
        'maskedsoftmax': MaskedSoftmax(),
        'topk': TopK(),
        'gelu': nn.GELU(),
        'sigmoid': nn.Sigmoid(),
        'identity': nn.Identity(),
    }
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        bias: bool = False, 
        tlc_flag: bool = True, 
        tlc_kernel: int = 48, 
        activation: str = 'relu'
    ) -> None:
        super(SparseSelfAttention, self).__init__()
        
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.tlc_flag = tlc_flag
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        # Learnable temperature for attention scaling
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Feature projection and dynamic enhancement
        self.project_in = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.dynamic_conv = IDynamicDWConv(dim * 2, kernel_size=3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # Sparsity activation
        activation = activation.lower()
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(self._ACTIVATIONS.keys())}")
        self.act = self._ACTIVATIONS[activation]
        
        # TLC window configuration
        self.kernel_size = [tlc_kernel, tlc_kernel]
        
        # Storage for grid indices and original shape (used in test mode)
        self.original_size: Optional[tuple] = None
        self.idxes: Optional[list] = None
        self.nr: Optional[int] = None
        self.nc: Optional[int] = None
    
    def _forward(self, qv: torch.Tensor) -> torch.Tensor:
        """Core attention computation.
        
        Args:
            qv: Query-Value tensor (B, C*2, H, W)
            
        Returns:
            Attention output (B, heads, head_dim, H*W)
        """
        q, v = qv.chunk(2, dim=1)
        
        # Rearrange to (B, heads, head_dim, H*W)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        # Normalize Q and K for stable attention
        q = F.normalize(q, dim=-1)
        k = F.normalize(v, dim=-1)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        
        # Apply sparsity activation
        attn = self.act(attn)
        
        # Weighted aggregation
        out = attn @ v
        
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional TLC for efficient test-time inference.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # Dynamic feature enhancement
        qv = self.dynamic_conv(self.project_in(x))
        
        # Two modes: training (full global attention) and test with TLC (local windows)
        if self.training or not self.tlc_flag:
            # Global sparse attention during training
            out = self._forward(qv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', 
                          head=self.num_heads, h=h, w=w)
            out = self.project_out(out)
            return out
        
        # Test-time local cropping (TLC) for efficiency
        qv = self.grids(qv)  # Convert to local windows
        out = self._forward(qv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', 
                      head=self.num_heads, h=qv.shape[-2], w=qv.shape[-1])
        out = self.grids_inverse(out)  # Reconstruct full image
        
        out = self.project_out(out)
        return out
    
    def grids(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image into overlapping local windows for TLC.
        
        Code adapted from: https://github.com/megvii-research/TLC
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Concatenated local windows (B*num_windows, C, kernel_h, kernel_w)
        """
        b, c, h, w = x.shape
        self.original_size = (b, c // 2, h, w)
        
        if b != 1:
            # For batch size > 1, process each sample separately
            parts = []
            all_idxes = []
            for bi in range(b):
                parts_i, idxes_i = self._process_single_grid(x[bi:bi+1])
                parts.append(parts_i)
                all_idxes.append(idxes_i)
            self.idxes = all_idxes
            parts = torch.cat(parts, dim=0)
            return parts
        
        # Single batch processing
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col
        
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)
        
        parts = []
        idxes = []
        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i
        
        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts
    
    def _process_single_grid(self, x: torch.Tensor) -> tuple:
        """Process single sample grid division."""
        _, c, h, w = x.shape
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)
        
        parts = []
        idxes = []
        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i
        
        parts = torch.cat(parts, dim=0)
        return parts, idxes
    
    def grids_inverse(self, outs: torch.Tensor) -> torch.Tensor:
        """Reconstruct full image from local window outputs (TLC inverse).
        
        Code adapted from: https://github.com/megvii-research/TLC
        
        Args:
            outs: Output from attention on local windows
            
        Returns:
            Reconstructed full image (1, C, H, W)
        """
        b, c, h, w = self.original_size
        preds = torch.zeros(self.original_size, dtype=outs.dtype, device=outs.device)
        count_mt = torch.zeros((b, 1, h, w), dtype=outs.dtype, device=outs.device)
        
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        
        # Handle batch processing
        if isinstance(self.idxes[0], list):
            # Multiple batch samples
            out_idx = 0
            for bi in range(b):
                for each_idx in self.idxes[bi]:
                    i = each_idx['i']
                    j = each_idx['j']
                    preds[bi:bi+1, :, i:i + k1, j:j + k2] += outs[out_idx:out_idx+1]
                    count_mt[bi:bi+1, :, i:i + k1, j:j + k2] += 1.
                    out_idx += 1
        else:
            # Single batch
            for cnt, each_idx in enumerate(self.idxes):
                i = each_idx['i']
                j = each_idx['j']
                preds[0, :, i:i + k1, j:j + k2] += outs[cnt]
                count_mt[0, 0, i:i + k1, j:j + k2] += 1.
        
        return preds / count_mt


# ============================================================
# Attention Block (Combines Attention + Norm + FFN)
# ============================================================

class AttBlock(nn.Module):
    """Attention Block combining layer normalization, sparse attention, and feed-forward.
    
    Structure: 
        - x → LayerNorm → SparseSelfAttention → + x
        - → LayerNorm → FeedForward → + x
    
    Args:
        dim: Channel dimension
        num_heads: Number of attention heads
        ffn_expansion_factor: Expansion factor for FFN
        bias: Whether to use bias
        tlc_flag: Enable TLC for test-time efficiency
        tlc_kernel: Kernel size for TLC windows
        activation: Sparsity activation type
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_expansion_factor: float = 2.0,
        bias: bool = False,
        tlc_flag: bool = True,
        tlc_kernel: int = 48,
        activation: str = 'relu'
    ) -> None:
        super(AttBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        
        self.attn = SparseSelfAttention(
            dim, 
            num_heads=num_heads, 
            bias=bias,
            tlc_flag=tlc_flag, 
            tlc_kernel=tlc_kernel, 
            activation=activation
        )
        self.ffn = FeedForward(dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention block with residual connections."""
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


# ============================================================
# Weight Initialization Utilities
# ============================================================

def _init_weights(m: nn.Module) -> None:
    """Initialize module weights using trunc_normal_ for transformer modules.
    
    Args:
        m: Module to initialize
    """
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        # Kaiming initialization for conv2d
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ============================================================
# Main ASSA Module (backward compatibility)
# ============================================================

# Alias for backward compatibility with existing code
class ASSA(SparseSelfAttention):
    """Backward compatibility alias for SparseSelfAttention."""
    pass


if __name__ == '__main__':
    """Test the ASSA implementation with sample inputs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    batch_size = 2
    channels = 64
    height, width = 224, 224
    num_heads = 8
    
    # Create test tensor
    x = torch.randn(batch_size, channels, height, width, device=device)
    
    print(f"Input shape: {x.shape}")
    
    # Test 1: SparseSelfAttention module
    print("\n[Test 1] SparseSelfAttention")
    attn = SparseSelfAttention(channels, num_heads=num_heads, tlc_flag=False).to(device)
    attn.train()
    out_attn = attn(x)
    print(f"Output shape: {out_attn.shape}")
    assert out_attn.shape == x.shape, "Shape mismatch!"
    
    # Test 2: AttBlock module
    print("\n[Test 2] AttBlock")
    att_block = AttBlock(channels, num_heads=num_heads, tlc_flag=False).to(device)
    att_block.train()
    out_block = att_block(x)
    print(f"Output shape: {out_block.shape}")
    assert out_block.shape == x.shape, "Shape mismatch!"
    
    # Test 3: Test mode with TLC
    print("\n[Test 3] TLC Test Mode (batch=1)")
    x_single = torch.randn(1, channels, 256, 256, device=device)
    attn_tlc = SparseSelfAttention(channels, num_heads=num_heads, tlc_flag=True, tlc_kernel=96).to(device)
    attn_tlc.eval()
    out_tlc = attn_tlc(x_single)
    print(f"Output shape: {out_tlc.shape}")
    assert out_tlc.shape == x_single.shape, "Shape mismatch!"
    
    # Test 4: Different activations
    print("\n[Test 4] Different Activations")
    for activation in ['relu', 'softmax', 'gelu', 'sigmoid']:
        attn_act = SparseSelfAttention(channels, num_heads=num_heads, activation=activation).to(device)
        attn_act.train()
        out_act = attn_act(x)
        print(f"  {activation}: {out_act.shape}")
        assert out_act.shape == x.shape, f"Shape mismatch for {activation}!"
    
    print("\n✓ All tests passed!")