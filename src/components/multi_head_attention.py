"""
多头注意力机制 (Multi-Head Attention)

将输入投影到多个子空间，并在每个子空间中独立执行注意力机制
公式：
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    """多头注意力层"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout 比率
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q, K, V 的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        """
        将最后一个维度分割成 (num_heads, d_k)
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """
        合并多个头
        Args:
            x: [batch_size, num_heads, seq_len, d_k]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
    
    def attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        Args:
            Q, K, V: [batch_size, num_heads, seq_len, d_k]
            mask: [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
        Returns:
            output: [batch_size, num_heads, seq_len, d_k]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        # scores: [batch_size, num_heads, seq_len, seq_len]
        
        # 应用掩码
        if mask is not None:
            # mask 可能是 [batch, 1, 1, seq_len] 或 [batch, 1, seq_len, seq_len]
            # 需要广播到 [batch, num_heads, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x, mask=None, key=None, value=None):
        """
        Args:
            x: 查询输入 [batch_size, seq_len_q, d_model]
            mask: 注意力掩码
            key: 键输入（可选，用于交叉注意力）[batch_size, seq_len_k, d_model]
            value: 值输入（可选，用于交叉注意力）[batch_size, seq_len_v, d_model]
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = x.size(0)
        
        # 如果没有提供key和value，则使用x（自注意力）
        if key is None:
            key = x
        if value is None:
            value = x
        
        # 线性变换
        Q = self.W_q(x)  # [batch_size, seq_len_q, d_model]
        K = self.W_k(key)  # [batch_size, seq_len_k, d_model]
        V = self.W_v(value)  # [batch_size, seq_len_v, d_model]
        
        # 分割多头
        Q = self.split_heads(Q)  # [batch_size, num_heads, seq_len, d_k]
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 应用注意力
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = self.combine_heads(attn_output)  # [batch_size, seq_len, d_model]
        
        # 输出线性变换
        output = self.W_o(attn_output)
        
        return output, attention_weights


def test_multi_head_attention():
    """测试多头注意力机制"""
    print("\n" + "="*50)
    print("测试多头注意力机制 (Multi-Head Attention)")
    print("="*50)
    
    # 参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    print(f"\n配置:")
    print(f"  - 模型维度 (d_model): {d_model}")
    print(f"  - 注意力头数 (num_heads): {num_heads}")
    print(f"  - 每个头的维度 (d_k): {d_model // num_heads}")
    
    # 创建多头注意力层
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()  # 设置为评估模式，关闭 dropout
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入形状: {x.shape}")
    
    # 计算多头注意力
    with torch.no_grad():
        output, attention_weights = mha(x)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证形状
    assert output.shape == x.shape, "输出形状不匹配！"
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), "注意力权重形状不匹配！"
    print("✅ 形状验证通过")
    
    # 验证注意力权重和为1
    weight_sum = attention_weights.sum(dim=-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6), "注意力权重和不为1！"
    print("✅ 注意力权重归一化验证通过")
    
    # 测试不同的头数
    print("\n测试不同的注意力头数...")
    for heads in [4, 8, 16]:
        if d_model % heads == 0:
            mha_test = MultiHeadAttention(d_model=d_model, num_heads=heads)
            out, _ = mha_test(x)
            print(f"  - {heads} 个头: ✅")
    
    print("✅ 多头注意力机制测试成功！")
    print("\n多头注意力测试完成！\n")


if __name__ == "__main__":
    test_multi_head_attention()

