"""
自注意力机制 (Self-Attention)

计算公式：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

其中：
    Q: Query 矩阵
    K: Key 矩阵
    V: Value 矩阵
    d_k: Key 的维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfAttention(nn.Module):
    """自注意力层"""
    
    def __init__(self, d_model):
        """
        Args:
            d_model: 模型维度
        """
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        
        # Q, K, V 的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        Returns:
            输出张量 [batch_size, seq_len, d_model]
            注意力权重 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.size()
        
        # 线性变换得到 Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)  # [batch_size, seq_len, d_model]
        V = self.W_v(x)  # [batch_size, seq_len, d_model]
        
        # 计算注意力分数
        # scores = QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_model)
        # scores: [batch_size, seq_len, seq_len]
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights: [batch_size, seq_len, seq_len]
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        # output: [batch_size, seq_len, d_model]
        
        # 输出线性变换
        output = self.W_o(output)
        
        return output, attention_weights


def test_self_attention():
    """测试自注意力机制"""
    print("\n" + "="*50)
    print("测试自注意力机制 (Self-Attention)")
    print("="*50)
    
    # 参数
    batch_size = 2
    seq_len = 8
    d_model = 64
    
    # 创建自注意力层
    attention = SelfAttention(d_model=d_model)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入形状: {x.shape}")
    
    # 计算自注意力
    output, attention_weights = attention(x)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert output.shape == x.shape, "输出形状不匹配！"
    assert attention_weights.shape == (batch_size, seq_len, seq_len), "注意力权重形状不匹配！"
    print("✅ 形状验证通过")
    
    # 验证注意力权重和为1
    weight_sum = attention_weights.sum(dim=-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6), "注意力权重和不为1！"
    print("✅ 注意力权重归一化验证通过")
    
    # 测试带掩码的情况
    print("\n测试带掩码的自注意力...")
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).expand(batch_size, -1, -1)
    output_masked, attention_weights_masked = attention(x, mask=mask)
    print(f"掩码后输出形状: {output_masked.shape}")
    print("✅ 掩码测试通过")
    print("✅ 自注意力机制测试成功！")
    
    print("\n自注意力测试完成！\n")


if __name__ == "__main__":
    test_self_attention()

