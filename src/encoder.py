"""
Transformer Encoder 实现

Encoder Block 结构：
    输入 -> 多头自注意力 -> 残差连接+LayerNorm -> 前馈网络 -> 残差连接+LayerNorm -> 输出
"""

import torch
import torch.nn as nn
from .components import MultiHeadAttention, PositionWiseFeedForward, SublayerConnection


class EncoderBlock(nn.Module):
    """单个 Encoder 块"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout 比率
        """
        super(EncoderBlock, self).__init__()
        
        # 多头自注意力层
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # 两个子层连接（残差+LayerNorm）
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 自注意力子层
        x = self.sublayer1(x, lambda x: self.self_attention(x, mask)[0])
        
        # 前馈网络子层
        x = self.sublayer2(x, self.feed_forward)
        
        return x


class Encoder(nn.Module):
    """完整的 Transformer Encoder"""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            num_layers: Encoder 块的数量
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout 比率
        """
        super(Encoder, self).__init__()
        
        # 堆叠多个 Encoder 块
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Pre-LN架构需要最终的LayerNorm
        from .components import LayerNorm
        self.final_norm = LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 依次通过每个 Encoder 块
        for layer in self.layers:
            x = layer(x, mask)
        
        # Pre-LN架构的最终归一化
        x = self.final_norm(x)
        
        return x


def test_encoder():
    """测试 Encoder"""
    print("\n" + "="*50)
    print("测试 Transformer Encoder")
    print("="*50)
    
    # 参数配置
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    
    print(f"\n配置:")
    print(f"  - Encoder 层数: {num_layers}")
    print(f"  - 模型维度: {d_model}")
    print(f"  - 注意力头数: {num_heads}")
    print(f"  - 前馈网络维度: {d_ff}")
    
    # 创建 Encoder
    encoder = Encoder(num_layers, d_model, num_heads, d_ff)
    encoder.eval()
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = encoder(x)
    
    print(f"输出形状: {output.shape}")
    
    # 验证
    assert output.shape == x.shape, "输出形状不匹配！"
    print("✅ 形状验证通过")
    
    # 测试带掩码
    print("\n测试带掩码的 Encoder...")
    # 创建填充掩码（mask掉部分位置）
    mask = torch.ones(batch_size, 1, 1, seq_len)  # [batch, 1, 1, seq_len]
    mask[0, :, :, 5:] = 0  # 第一个样本mask掉后半部分
    with torch.no_grad():
        output_masked = encoder(x, mask)
    print(f"带掩码输出形状: {output_masked.shape}")
    print("✅ 掩码测试通过")
    
    # 参数统计
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder 总参数量: {total_params:,}")
    
    print("\nEncoder 测试完成！\n")


if __name__ == "__main__":
    test_encoder()

