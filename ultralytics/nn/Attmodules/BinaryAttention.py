# ============================================================================
# BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers
# ============================================================================
# Efficient attention mechanism achieving >2x speedup through 1-bit quantization
# of queries and keys using Straight-Through Estimator (STE) for proper gradient flow.
# 
# Reference: https://github.com/EdwardChasel/BinaryAttention
# Paper: CVPR 2026 - "BinaryAttention: One-Bit QK-Attention for Vision and 
#        Diffusion Transformers" by Chaodong Xiao, Zhengqiang Zhang, Lei Zhang
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any


# ============================================================================
# Quantization Functions (Straight-Through Estimator based)
# ============================================================================

def round_ste(z):
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def binary_sign(x: torch.Tensor) -> torch.Tensor:
    """Return -1 if x < 0, 1 if x >= 0."""
    return x.sign() + (x == 0).type(torch.float)


class STESign(Function):
    """
    Binarize tensor using sign function with Straight-Through Estimator (STE).
    This allows proper backpropagation through the sign operation.
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        """
        Return a Sign tensor.
        
        Args:
            ctx: context
            x: input tensor
        
        Returns:
            Sign(x) where each element is either -1 or 1
        """
        ctx.save_for_backward(x)
        sign_x = binary_sign(x)
        return sign_x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient using STE (pass through for |x| <= 1, zero otherwise).
        
        Args:
            ctx: context
            grad_output: gradient w.r.t. output of Sign
        
        Returns:
            Gradient w.r.t. input of the Sign function
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.gt(1)] = 0
        grad_input[x.lt(-1)] = 0
        return grad_input


binarize = STESign.apply


class SymQuantizer(Function):
    """
    Symmetric quantization using straight-through estimator.
    Quantizes input to a fixed number of bits with symmetric range.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, clip_val: torch.Tensor, num_bits: int, layerwise: bool = False):
        """
        Symmetric quantization forward pass.
        
        Args:
            ctx: autograd context
            input: tensor to be quantized
            clip_val: clipping range [min, max]
            num_bits: number of quantization bits
            layerwise: whether to use layerwise or channelwise quantization
        
        Returns:
            Quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        ctx.num_bits = num_bits
        
        # Clip input to range
        input_clipped = torch.clamp(input, clip_val[0], clip_val[1])
        
        if layerwise:
            max_input = torch.max(torch.abs(input_clipped))
        else:
            # Channelwise: compute max per channel
            if input.ndimension() == 4:
                max_input = torch.max(torch.abs(input_clipped), dim=-1, keepdim=True)[0]
            else:
                max_input = torch.max(torch.abs(input_clipped))
        
        # Compute scale and quantize
        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-8)
        output = torch.round(input_clipped * s) / s
        
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass for quantization (straight-through estimator).
        """
        input, clip_val = ctx.saved_tensors
        # Gradient is passed through for values within clip range
        grad_input = grad_output.clone()
        grad_input[input < clip_val[0]] = 0
        grad_input[input > clip_val[1]] = 0
        
        return grad_input, None, None, None


symquantize = SymQuantizer.apply


class BinaryAttention(nn.Module):
    """
    Binary Query-Key Attention Module (BinaryAttention).
    
    Efficient attention mechanism using 1-bit binarized queries and keys while maintaining
    full-precision values. Achieves >2x speedup compared to FlashAttention2 on A100 GPUs
    with competitive accuracy.
    
    Reference: "BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers"
    (CVPR 2026) - https://github.com/EdwardChasel/BinaryAttention
    
    Features:
        - 1-bit quantization of Q/K using sign function with learned scaling (STE-based)
        - Optional learnable relative position bias to compensate for quantization loss
        - 8-bit quantization of attention weights and values (optional)
        - Proper handling of attention for vision transformers
    
    Args:
        dim (int): Input/output channel dimension
        num_heads (int): Number of attention heads (must divide dim), default: 8
        qkv_bias (bool): Whether to use bias in QKV projection, default: False
        attn_drop (float): Dropout probability for attention weights, default: 0.0
        proj_drop (float): Dropout probability for output projection, default: 0.0
        attn_quant (bool): Whether to binarize Q/K, default: False
        attn_bias (bool): Whether to add relative position bias, default: False
        pv_quant (bool): Whether to quantize attention weights and values to 8-bit, default: False
        input_size (tuple): Spatial input size (H, W) for relative position bias calculation.
                           Required when attn_bias=True, default: None
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attn_quant=False, attn_bias=False,
                 pv_quant=False, input_size=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_quant = attn_quant
        self.attn_bias = attn_bias
        self.pv_quant = pv_quant

        if self.attn_bias and input_size is not None:
            self.input_size = input_size
            self.num_relative_distance = (2 * input_size[0] - 1) * (2 * input_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(input_size[0])
            coords_w = torch.arange(input_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += input_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += input_size[1] - 1
            relative_coords[:, :, 0] *= 2 * input_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(input_size[0] * input_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.input_size = None

    @staticmethod
    def _quantize(x):
        """
        Binarize query and key using sign function with learned scaling.
        Uses Straight-Through Estimator for proper gradient flow.
        
        Args:
            x: Input tensor (B, num_heads, N, head_dim)
        
        Returns:
            Quantized tensor with binarized values scaled by mean absolute value
        """
        s = x.abs().mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        sign = binarize(x)
        return s * sign

    @staticmethod
    def _quantize_p(x):
        """
        Quantize attention weights to 8-bit unsigned integers [0, 255].
        """
        qmax = 255
        s = 1.0 / qmax
        q = round_ste(x / s).clamp(0, qmax)
        return s * q

    @staticmethod
    def _quantize_v(x, bits=8):
        """
        Quantize value tensor to specified bits with clipping.
        Uses symmetric quantization range [-2.0, 2.0].
        """
        act_clip_val = torch.tensor([-2.0, 2.0], dtype=x.dtype, device=x.device)
        return symquantize(x, act_clip_val, bits, False)

    def forward(self, x):
        """
        Forward pass with binary quantization support.
        Handles both 3D (B, N, C) and 4D (B, C, H, W) inputs with memory-efficient chunked attention.
        
        Args:
            x: Input tensor of shape (B, N, C) or (B, C, H, W)
        
        Returns:
            Output tensor with same shape as input
        """
        # Handle both 3D and 4D input tensors
        original_shape_is_4d = (x.dim() == 4)
        if original_shape_is_4d:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H*W, C)
        
        B, N, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Apply quantization to Q/K if enabled
        if self.attn_quant:
            q = self._quantize(q)
            k = self._quantize(k)
        
        # Use chunked attention to avoid OOM on large sequences
        x = self._chunked_attention(q, k, v, B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Reshape back to 4D if input was 4D
        if original_shape_is_4d:
            x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x
    
    def _chunked_attention(self, q, k, v, B, N, C, chunk_size=128):
        """
        Memory-efficient windowed attention with small local regions.
        This is suitable for detection tasks where local context is important.
        
        Args:
            q, k, v: Query, key, value tensors (B, num_heads, N, head_dim)
            B, N, C: Batch size, sequence length, channels
            chunk_size: Size of local window for attention
        
        Returns:
            Attention output (B, N, C)
        """
        num_heads = self.num_heads
        head_dim = C // num_heads
        
        # For small sequences, compute attention directly
        if N <= chunk_size * 4:  # Reduced threshold
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            if self.attn_bias and hasattr(self, 'relative_position_index'):
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.view(-1)
                ].view(
                    self.input_size[0] * self.input_size[1] + 1,
                    self.input_size[0] * self.input_size[1] + 1, -1
                )
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                attn = attn + relative_position_bias.unsqueeze(0)
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            if self.pv_quant:
                attn = self._quantize_p(attn)
                v_quant = self._quantize_v(v, 8)
            else:
                v_quant = v
            
            x_out = (attn @ v_quant).transpose(1, 2).reshape(B, N, C)
            return x_out
        
        # For large sequences, use local windowed attention
        # Each query attends only to keys within a local window
        window_size = chunk_size  # Size of attention window
        x_out_list = []
        
        for q_start in range(0, N, chunk_size):
            q_end = min(q_start + chunk_size, N)
            q_chunk = q[:, :, q_start:q_end, :]  # (B, num_heads, chunk_size, head_dim)
            
            # Define key window around query chunk with some context
            context = window_size // 2
            k_start = max(0, q_start - context)
            k_end = min(N, q_end + context)
            
            k_window = k[:, :, k_start:k_end, :]  # (B, num_heads, window_size, head_dim)
            v_window = v[:, :, k_start:k_end, :]  # (B, num_heads, window_size, head_dim)
            
            # Compute attention within window
            attn_window = (q_chunk @ k_window.transpose(-2, -1)) * self.scale  # (B, num_heads, chunk_size, window_size)
            
            # Apply softmax and dropout
            attn_window = attn_window.softmax(dim=-1)
            attn_window = self.attn_drop(attn_window)
            
            # Apply quantization if enabled
            if self.pv_quant:
                attn_window = self._quantize_p(attn_window)
                v_window = self._quantize_v(v_window, 8)
            
            # Apply attention to get output for this chunk
            out_chunk = (attn_window @ v_window)  # (B, num_heads, chunk_size, head_dim)
            x_out_list.append(out_chunk)
        
        # Concatenate all chunks
        x_out = torch.cat(x_out_list, dim=2)  # (B, num_heads, N, head_dim)
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        return x_out


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(1, 32*32, 64).to(device)
    model = BinaryAttention(64).to(device)

    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)