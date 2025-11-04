"""
位置前馈神经网络 (Position-wise Feed-Forward Network)

两层全连接网络，中间使用 ReLU 激活函数
公式：
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

在 Transformer 中，每个位置的向量独立地通过相同的前馈网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """位置前馈神经网络层"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度（输入和输出维度）
            d_ff: 前馈网络的隐藏层维度（通常是 d_model 的 4 倍）
            dropout: dropout 比率
        """
        super(PositionWiseFeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 第一层：d_model -> d_ff，应用 ReLU
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层：d_ff -> d_model
        # 注意：原始Transformer只在ReLU后使用一次dropout
        x = self.linear2(x)
        
        return x


class PositionWiseFeedForwardGELU(nn.Module):
    """使用 GELU 激活函数的位置前馈网络（类似 GPT）"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络的隐藏层维度
            dropout: dropout 比率
        """
        super(PositionWiseFeedForwardGELU, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x


def test_feed_forward():
    """测试前馈神经网络"""
    print("\n" + "="*50)
    print("测试位置前馈神经网络 (Position-wise FFN)")
    print("="*50)
    
    # 参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048  # 通常是 d_model 的 4 倍
    
    print(f"\n配置:")
    print(f"  - 模型维度 (d_model): {d_model}")
    print(f"  - 前馈网络隐藏层维度 (d_ff): {d_ff}")
    print(f"  - d_ff / d_model = {d_ff / d_model}")
    
    # 测试 ReLU 版本
    print("\n[1] 测试 ReLU 版本")
    ffn_relu = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
    ffn_relu.eval()  # 关闭 dropout
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output_relu = ffn_relu(x)
    print(f"输出形状: {output_relu.shape}")
    
    # 验证形状
    assert output_relu.shape == x.shape, "输出形状不匹配！"
    print("✅ 形状验证通过")
    
    # 测试 GELU 版本
    print("\n[2] 测试 GELU 版本")
    ffn_gelu = PositionWiseFeedForwardGELU(d_model=d_model, d_ff=d_ff)
    ffn_gelu.eval()  # 关闭 dropout
    with torch.no_grad():
        output_gelu = ffn_gelu(x)
    print(f"输出形状: {output_gelu.shape}")
    assert output_gelu.shape == x.shape, "输出形状不匹配！"
    print("✅ 形状验证通过")
    
    # 比较参数量
    print("\n[3] 参数统计")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params_relu = count_parameters(ffn_relu)
    params_gelu = count_parameters(ffn_gelu)
    
    print(f"ReLU FFN 参数量: {params_relu:,}")
    print(f"GELU FFN 参数量: {params_gelu:,}")
    print(f"参数量计算: d_model * d_ff + d_ff + d_ff * d_model + d_model")
    print(f"            = {d_model} * {d_ff} + {d_ff} + {d_ff} * {d_model} + {d_model}")
    print(f"            = {params_relu:,}")
    
    # 测试不同维度
    print("\n[4] 测试不同的隐藏层维度")
    for ff_dim in [1024, 2048, 4096]:
        ffn_test = PositionWiseFeedForward(d_model=d_model, d_ff=ff_dim)
        out = ffn_test(x)
        params = count_parameters(ffn_test)
        print(f"  - d_ff={ff_dim}: 参数量 {params:,}")
    
    # 验证位置独立性
    print("\n[5] 验证位置独立性")
    # 前馈网络应该对每个位置独立应用相同的变换
    x_single = x[:, 0:1, :]  # 取第一个位置
    with torch.no_grad():
        output_single = ffn_relu(x_single)
    assert torch.allclose(output_single, output_relu[:, 0:1, :], atol=1e-6), "位置不独立！"
    print("✅ 位置独立性验证通过（每个位置使用相同的权重）")
    
    print("✅ 前馈神经网络测试成功！")
    print("\n前馈神经网络测试完成！\n")


if __name__ == "__main__":
    test_feed_forward()

