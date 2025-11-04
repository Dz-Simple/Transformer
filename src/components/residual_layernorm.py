"""
残差连接和层归一化 (Residual Connection & Layer Normalization)

残差连接：
    output = x + Sublayer(x)
    
层归一化：
    LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
    其中 μ 和 σ 是在特征维度上计算的均值和标准差
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """层归一化"""
    
    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: 模型维度
            eps: 防止除零的小常数
        """
        super(LayerNorm, self).__init__()
        
        # 可学习的缩放和偏移参数
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            归一化后的张量
        """
        # 在最后一个维度（特征维度）上计算均值和标准差
        mean = x.mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        std = x.std(dim=-1, keepdim=True, unbiased=False)    # [batch_size, seq_len, 1]
        
        # 归一化
        x_norm = (x - mean) / (std + self.eps)
        
        # 缩放和平移
        return self.gamma * x_norm + self.beta


class ResidualConnection(nn.Module):
    """残差连接 + Dropout + Layer Normalization
    
    实现 Post-LN 或 Pre-LN 两种方式：
    - Post-LN: x + LayerNorm(Sublayer(x))
    - Pre-LN: x + Sublayer(LayerNorm(x))
    """
    
    def __init__(self, d_model, dropout=0.1, pre_norm=False):
        """
        Args:
            d_model: 模型维度
            dropout: dropout 比率
            pre_norm: 是否使用 Pre-LN（True）还是 Post-LN（False）
        """
        super(ResidualConnection, self).__init__()
        
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        
    def forward(self, x, sublayer):
        """
        Args:
            x: 输入张量
            sublayer: 子层（函数或 nn.Module）
        Returns:
            残差连接后的输出
        """
        if self.pre_norm:
            # Pre-LN: x + Sublayer(LayerNorm(x))
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            # Post-LN: LayerNorm(x + Sublayer(x))
            return self.norm(x + self.dropout(sublayer(x)))


class SublayerConnection(nn.Module):
    """
    完整的子层连接：残差连接 + LayerNorm
    这是 Transformer 中常用的包装器
    
    修改为 Pre-LN 以提高训练稳定性
    """
    
    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            dropout: dropout 比率
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        应用残差连接到任何具有相同大小的子层
        Pre-LN 方式: x + Dropout(Sublayer(LayerNorm(x)))
        这比原始的Post-LN更稳定，训练更容易
        """
        return x + self.dropout(sublayer(self.norm(x)))


def test_layer_norm():
    """测试层归一化"""
    print("\n" + "="*50)
    print("测试层归一化 (Layer Normalization)")
    print("="*50)
    
    # 参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    # 创建层归一化
    ln = LayerNorm(d_model=d_model)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入形状: {x.shape}")
    print(f"输入统计: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # 应用层归一化
    output = ln(x)
    print(f"\n输出形状: {output.shape}")
    print(f"输出统计: mean={output.mean():.4f}, std={output.std():.4f}")
    
    # 验证形状
    assert output.shape == x.shape, "形状不匹配！"
    print("✅ 形状验证通过")
    
    # 验证归一化效果（每个位置的特征维度应该是归一化的）
    output_mean = output.mean(dim=-1)
    output_std = output.std(dim=-1)
    print(f"\n每个位置的统计:")
    print(f"  - 均值范围: [{output_mean.min():.4f}, {output_mean.max():.4f}]")
    print(f"  - 标准差范围: [{output_std.min():.4f}, {output_std.max():.4f}]")
    
    # 比较 PyTorch 内置的 LayerNorm
    ln_torch = nn.LayerNorm(d_model)
    ln_torch.weight.data = ln.gamma.data
    ln_torch.bias.data = ln.beta.data
    output_torch = ln_torch(x)
    
    assert torch.allclose(output, output_torch, atol=1e-5), "与 PyTorch 实现不一致！"
    print("✅ 与 PyTorch LayerNorm 结果一致")


def test_residual_connection():
    """测试残差连接"""
    print("\n" + "="*50)
    print("测试残差连接 (Residual Connection)")
    print("="*50)
    
    # 参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    # 创建一个简单的子层（线性变换）
    sublayer = nn.Linear(d_model, d_model)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入形状: {x.shape}")
    
    # 测试 Post-LN
    print("\n[1] 测试 Post-LN (原始 Transformer)")
    residual_post = ResidualConnection(d_model=d_model, pre_norm=False)
    output_post = residual_post(x, sublayer)
    print(f"输出形状: {output_post.shape}")
    assert output_post.shape == x.shape, "形状不匹配！"
    print("✅ Post-LN 测试通过")
    
    # 测试 Pre-LN
    print("\n[2] 测试 Pre-LN (更稳定的训练)")
    residual_pre = ResidualConnection(d_model=d_model, pre_norm=True)
    output_pre = residual_pre(x, sublayer)
    print(f"输出形状: {output_pre.shape}")
    assert output_pre.shape == x.shape, "形状不匹配！"
    print("✅ Pre-LN 测试通过")
    
    # 测试 SublayerConnection
    print("\n[3] 测试 SublayerConnection")
    sublayer_conn = SublayerConnection(d_model=d_model)
    output_conn = sublayer_conn(x, sublayer)
    print(f"输出形状: {output_conn.shape}")
    assert output_conn.shape == x.shape, "形状不匹配！"
    print("✅ SublayerConnection 测试通过")
    
    # 验证残差连接的效果
    print("\n[4] 验证残差连接效果")
    # 如果子层输出为零，残差连接应该保持输入不变（忽略 dropout 和 norm）
    zero_sublayer = lambda x: torch.zeros_like(x)
    residual_test = ResidualConnection(d_model=d_model, dropout=0.0, pre_norm=True)
    
    # Pre-LN 且子层输出为0: output = x + 0 = x
    with torch.no_grad():
        output_zero = residual_test(x, zero_sublayer)
        # 由于有 LayerNorm，不会完全相等，但应该保持信息
        print(f"  - 输入和输出的相关性: {torch.corrcoef(torch.stack([x.flatten(), output_zero.flatten()]))[0,1]:.4f}")
    
    print("✅ 残差连接保持了输入信息")


def test_combined():
    """测试完整的 Transformer 块结构"""
    print("\n" + "="*50)
    print("测试完整 Transformer 块")
    print("="*50)
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    # 模拟一个 Transformer 块：
    # x -> [残差(注意力)] -> [残差(前馈)]
    
    attention_layer = nn.Linear(d_model, d_model)
    ffn_layer = nn.Sequential(
        nn.Linear(d_model, 2048),
        nn.ReLU(),
        nn.Linear(2048, d_model)
    )
    
    residual1 = SublayerConnection(d_model)
    residual2 = SublayerConnection(d_model)
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入形状: {x.shape}")
    
    # 注意力子层
    x = residual1(x, attention_layer)
    print(f"经过注意力子层后: {x.shape}")
    
    # 前馈子层
    x = residual2(x, ffn_layer)
    print(f"经过前馈子层后: {x.shape}")
    
    print("✅ 完整 Transformer 块测试通过")


def main():
    """主测试函数"""
    test_layer_norm()
    test_residual_connection()
    test_combined()
    print("\n" + "="*50)
    print("✅ 残差连接和层归一化测试成功！")
    print("所有测试完成！")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

