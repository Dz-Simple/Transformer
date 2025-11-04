"""
位置编码 (Positional Encoding)

使用正弦和余弦函数为序列中的每个位置生成唯一的编码
公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码层"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型的维度（embedding 维度）
            max_len: 最大序列长度
            dropout: dropout 比率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算 div_term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加 batch 维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # 注册为 buffer（不是模型参数，但需要保存）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            添加位置编码后的张量
        """
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def test_positional_encoding():
    """测试位置编码"""
    print("\n" + "="*50)
    print("测试位置编码 (Positional Encoding)")
    print("="*50)
    
    # 参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    # 创建位置编码层
    pe = PositionalEncoding(d_model=d_model, max_len=100)
    
    # 创建输入（词嵌入）
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入形状: {x.shape}")
    
    # 应用位置编码
    output = pe(x)
    print(f"输出形状: {output.shape}")
    
    # 验证形状
    assert output.shape == x.shape, "形状不匹配！"
    print("✅ 形状验证通过")
    print("✅ 位置编码测试成功！")
    
    print("\n位置编码测试完成！\n")


if __name__ == "__main__":
    test_positional_encoding()

