# 循环注意力模块（Circulant Attention）
# 核心设计：基于频域DFT变换优化注意力计算，通过“时域→频域→元素乘法→时域”的流程，
# 将传统自注意力O(N²)复杂度降至O(NlogN)，在保证全局依赖捕捉能力的同时，大幅提升计算效率

import torch
import torch.nn as nn


class Linear(nn.Linear):
    """
    复数线性层：适配复数输入的线性变换，支持频域特征的线性投影
    核心逻辑：将复数拆分为实部-虚部，完成线性变换后重组为复数
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        device: 设备（默认None）
        dtype: 数据类型（默认None）
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__(in_features, out_features, False, device, dtype)

    def forward(self, x):
        # 复数→实虚部分离→转置适配线性层→线性变换→还原复数格式
        # 处理混合精度：权重转换到与输入相同的dtype（处理float16/float32混合）
        x = torch.view_as_real(x).transpose(-2, -1)
        weight = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        x = torch.nn.functional.linear(x, weight).transpose(-2, -1)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.contiguous())
        return x


class CirculantAttention(nn.Module):
    """
    循环注意力模块（Circulant Attention）
    
    功能：频域优化的全局注意力，高效捕捉长距离依赖
    
    核心设计：
        - 频域降复杂度：利用DFT性质，将注意力矩阵乘法转化为频域元素乘法，复杂度从O(N²)降至O(NlogN)
        - FFT优化顺序：在QKV投影前应用FFT，减少冗余FFT调用，与空域等效但吞吐量更高
        - 复数线性投影：使用Complex Linear层处理频域复数特征
        - 门控调制：通过SiLU门控动态调整特征权重，提升表达能力
        - 正交归一化：FFT使用ortho norm，隐含1/N归一化因子，保证计算稳定性
        - rfft2高效计算：利用Hermitian对称性，相比fft2节省~50%计算和内存
    
    数学基础（论文公式15-16）：
        公式15 - 注意力权重：
            a = (1/N√d)[F^{-1}(conj(F(Q)) ⊙ F(K))] · 1_{d×1}
            其中 ⊙ 为元素乘法（Hadamard product）
        
        公式16 - 值聚合输出：
            O = F^{-1}(conj(F(σ(a))) ⊙ F(V))
            其中 σ(·) 为softmax归一化
        
        说明：
            - d=1在实践中使用（参见论文Table 5）
            - 1/N因子通过norm='ortho'隐含实现
            - N为序列长度（空间像素总数）
    
    输入/输出：
        - 输入形状：[batch, channels, height, width]（CNN格式，与YOLO和其他卷积网络兼容）
        - 输出形状：与输入相同 [batch, channels, height, width]
        - 内部自动处理坐标变换以实现频域计算
    
    参考：
        - 论文：https://arxiv.org/abs/2512.21542
        - 官方实现：https://github.com/LeapLabTHU/Circulant-Attention
    
    参数：
        dim (int): 通道数（channels）
        proj_drop (float): 输出投影后的dropout概率，默认0.0
    """

    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.qkv = Linear(dim, dim * 3)  # 复数QKV生成（1个线性层生成3路特征）
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())  # 门控调制
        self.proj = nn.Linear(dim, dim)  # 输出投影
        self.proj_drop = nn.Dropout(proj_drop)  # 输出dropout

    def forward(self, x):
        """
        前向传播：时域→频域→注意力计算→时域→输出
        
        输入/输出格式：
            - 输入形状 [b, c, h, w]：批大小、通道数、高度、宽度（CNN格式，与YOLO兼容）
            - 输出形状 [b, c, h, w]：与输入相同
        
        计算流程：
            1. 转换输入格式 [b, c, h, w] → [b, n, c]（n=h×w）
            2. 计算门控向量 t（基于原始序列特征）
            3. 转换为空间形式 [b, h, w, c]
            4. 应用rfft2获得频域表示（float32以兼容cuFFT）
            5. QKV投影在频域进行（减少冗余FFT调用）
            6. 公式15：注意力权重计算 a = (1/N√d)[F^{-1}(conj(Q)⊙K)]·1_{d×1}
            7. 公式16：加权聚合 O = F^{-1}(conj(σ(a))⊙V)
            8. 门控调制、投影与输出
        
        论文参考：https://arxiv.org/abs/2512.21542
        """
        # 输入形状：[b, c, h, w]
        b, c, h, w = x.shape
        n = h * w  # 序列长度
        
        # 保存原始数据类型，处理混合精度训练（AMP）
        dtype_original = x.dtype

        # 步骤1：转换输入格式 [b, c, h, w] → [b, n, c]（序列形式，保持原始dtype）
        x_seq = x.permute(0, 2, 3, 1).reshape(b, n, c)  # [b, c, h, w] → [b, h, w, c] → [b, n, c]

        # 步骤2：计算门控向量 t（基于原始序列特征，保持原始dtype与网络一致）
        # 说明：在任何FFT转换前计算gate，保证特征一致性和dtype匹配
        t = self.gate(x_seq)  # [b, n, c]

        # 步骤3-5：准备Q、K、V
        # 注意：cuFFT 在半精度(float16)下仅支持2的幂次尺寸
        # 因此只在FFT操作前临时转换为float32，其他操作保持原始dtype
        x_spatial = x_seq.reshape(b, h, w, c)  # [b, n, c] → [b, h, w, c]
        
        # 仅为FFT转换到float32
        if dtype_original == torch.float16:
            x_spatial = x_spatial.float()
        
        x_fft = torch.fft.rfft2(x_spatial, dim=(1, 2), norm='ortho')  # rfft2节省50%计算量（只保存正频率）
        qkv = self.qkv(x_fft)  # [b, h, w, 3c]
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)  # 各 [b, h, w, c]

        # 步骤6：论文公式15 - 注意力权重计算
        # 公式：a = (1/N√d)[F^{-1}(conj(F(Q)) ⊙ F(K))] · 1_{d×1}
        # 说明：
        #   - conj(q) * k 在频域计算元素乘法（替代时域N²的矩阵乘法）
        #   - 1/N因子通过norm='ortho'隐含实现
        #   - d=1在实践中使用（参见论文Table 5）
        attn = torch.conj(q) * k  # 频域元素乘法
        attn = torch.fft.irfft2(attn, s=(h, w), dim=(1, 2), norm='ortho')  # 频域→时域

        # 步骤7：论文公式16 - 加权值聚合
        # 公式：O = F^{-1}(conj(F(σ(a))) ⊙ F(V))
        # 说明：
        #   - attn在时域应用softmax（维度dim=1为序列维度）
        #   - 再转频域与V进行元素乘法（替代时域矩阵乘法）
        attn = attn.reshape(b, n, c).softmax(dim=1).reshape(b, h, w, c)  # 时域softmax归一化
        attn = torch.fft.rfft2(attn, dim=(1, 2))  # 注意力权重→频域
        out = torch.conj(attn) * v  # 频域加权V
        out = torch.fft.irfft2(out, s=(h, w), dim=(1, 2), norm='ortho')  # 频域→时域

        # 恢复原始数据类型（如果之前转换为float32）
        if dtype_original == torch.float16:
            out = out.half()

        # 步骤8：门控调制、投影与输出
        out = out.reshape(b, n, c) * t  # 门控加权，[b, h, w, c] → [b, n, c]
        out = self.proj(out)  # 线性投影
        out = self.proj_drop(out)  # Dropout正则化
        
        # 转换输出格式 [b, n, c] → [b, c, h, w]（还原CNN格式）
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [b, n, c] → [b, h, w, c] → [b, c, h, w]

        return out