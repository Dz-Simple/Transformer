"""
Transformer Decoder 实现

Decoder Block 结构：
    输入 -> 带掩码的多头自注意力 -> 残差连接+LayerNorm 
        -> 交叉注意力（Encoder-Decoder Attention） -> 残差连接+LayerNorm
        -> 前馈网络 -> 残差连接+LayerNorm -> 输出
"""

import torch
import torch.nn as nn
from .components import MultiHeadAttention, PositionWiseFeedForward, SublayerConnection


class DecoderBlock(nn.Module):
    """单个 Decoder 块"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout 比率
        """
        super(DecoderBlock, self).__init__()
        
        # 带掩码的自注意力（用于目标序列）
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 交叉注意力（Encoder-Decoder Attention）
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # 三个子层连接
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder 输入 [batch_size, tgt_len, d_model]
            encoder_output: Encoder 输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（因果掩码）
        Returns:
            输出张量 [batch_size, tgt_len, d_model]
        """
        # 1. 带掩码的自注意力（确保不会看到未来的信息）
        x = self.sublayer1(x, lambda x: self.self_attention(x, tgt_mask)[0])
        
        # 2. 交叉注意力（Query来自Decoder，Key和Value来自Encoder）
        # 注意：这里需要修改 MultiHeadAttention 来支持不同的Q, K, V
        # 为简单起见，我们先使用相同的输入
        x = self.sublayer2(x, lambda x: self._cross_attention(x, encoder_output, src_mask))
        
        # 3. 前馈网络
        x = self.sublayer3(x, self.feed_forward)
        
        return x
    
    def _cross_attention(self, query, encoder_output, mask):
        """
        交叉注意力的辅助函数
        Query来自Decoder，Key和Value来自Encoder
        """
        # 使用query作为Q，encoder_output作为K和V
        attn_output, _ = self.cross_attention(
            query, 
            mask=mask, 
            key=encoder_output, 
            value=encoder_output
        )
        return attn_output


class Decoder(nn.Module):
    """完整的 Transformer Decoder"""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            num_layers: Decoder 块的数量
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout 比率
        """
        super(Decoder, self).__init__()
        
        # 堆叠多个 Decoder 块
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Pre-LN架构需要最终的LayerNorm
        from .components import LayerNorm
        self.final_norm = LayerNorm(d_model)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder 输入 [batch_size, tgt_len, d_model]
            encoder_output: Encoder 输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（因果掩码）
        Returns:
            输出张量 [batch_size, tgt_len, d_model]
        """
        # 依次通过每个 Decoder 块
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Pre-LN架构的最终归一化
        x = self.final_norm(x)
        
        return x


def generate_causal_mask(seq_len):
    """
    生成因果掩码（下三角矩阵）
    防止 Decoder 在训练时看到未来的信息
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0)  # [1, seq_len, seq_len]


def test_decoder():
    """测试 Decoder"""
    print("\n" + "="*50)
    print("测试 Transformer Decoder")
    print("="*50)
    
    # 参数配置
    batch_size = 2
    src_len = 10
    tgt_len = 8
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    
    print(f"\n配置:")
    print(f"  - Decoder 层数: {num_layers}")
    print(f"  - 模型维度: {d_model}")
    print(f"  - 注意力头数: {num_heads}")
    print(f"  - 前馈网络维度: {d_ff}")
    
    # 创建 Decoder
    decoder = Decoder(num_layers, d_model, num_heads, d_ff)
    decoder.eval()
    
    # 创建输入
    tgt_input = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)
    
    print(f"\nDecoder 输入形状: {tgt_input.shape}")
    print(f"Encoder 输出形状: {encoder_output.shape}")
    
    # 生成因果掩码
    tgt_mask = generate_causal_mask(tgt_len)
    print(f"因果掩码形状: {tgt_mask.shape}")
    print(f"\n因果掩码（防止看到未来）:")
    print(tgt_mask[0].int())
    
    # 前向传播
    with torch.no_grad():
        output = decoder(tgt_input, encoder_output, tgt_mask=tgt_mask)
    
    print(f"\n输出形状: {output.shape}")
    
    # 验证
    assert output.shape == tgt_input.shape, "输出形状不匹配！"
    print("✅ 形状验证通过")
    
    # 参数统计
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder 总参数量: {total_params:,}")
    
    print("\nDecoder 测试完成！\n")


if __name__ == "__main__":
    test_decoder()

