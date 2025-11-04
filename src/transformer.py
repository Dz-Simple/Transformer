"""
完整的 Transformer 模型实现

结构：
    输入 -> Embedding + 位置编码 -> Encoder -> Decoder -> 输出层
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .encoder import Encoder
from .decoder import Decoder, generate_causal_mask
from .components import PositionalEncoding

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Transformer(nn.Module):
    """完整的 Transformer 模型"""
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=5000,
        dropout=0.1,
        pad_idx=0
    ):
        """
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_encoder_layers: Encoder 层数
            num_decoder_layers: Decoder 层数
            d_ff: 前馈网络隐藏层维度
            max_seq_len: 最大序列长度
            dropout: dropout 比率
            pad_idx: padding索引（默认0）
        """
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx  # 保存pad_idx供generate使用
        
        # 源序列和目标序列的 Embedding 层（设置padding_idx）
        # padding_idx处的embedding向量不会被更新，且会被自动清零
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder 和 Decoder
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """
        改进的参数初始化策略
        参考: "Attention Is All You Need" 和后续最佳实践
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # 对于线性层和embedding，使用xavier初始化
                if 'embedding' in name:
                    nn.init.normal_(p, mean=0, std=self.d_model ** -0.5)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(p)
            # LayerNorm参数保持默认初始化(gamma=1, beta=0)
    
    def encode(self, src, src_mask=None):
        """
        Encoder 前向传播
        Args:
            src: 源序列 [batch_size, src_len]
            src_mask: 源序列掩码
        Returns:
            Encoder 输出 [batch_size, src_len, d_model]
        """
        # Embedding + 位置编码
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        src_emb = self.positional_encoding(src_emb)
        
        # 通过 Encoder
        encoder_output = self.encoder(src_emb, src_mask)
        
        return encoder_output
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decoder 前向传播
        Args:
            tgt: 目标序列 [batch_size, tgt_len]
            encoder_output: Encoder 输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（因果掩码）
        Returns:
            Decoder 输出 [batch_size, tgt_len, d_model]
        """
        # Embedding + 位置编码
        tgt_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        # 通过 Decoder
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
        
        return decoder_output
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        完整的前向传播
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        Returns:
            输出 logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # 输出投影
        output = self.output_projection(decoder_output)
        
        return output
    
    def generate(self, src, max_len=50, start_token=1, end_token=2):
        """
        自回归生成（贪心解码）
        Args:
            src: 源序列 [batch_size, src_len]
            max_len: 最大生成长度
            start_token: 开始标记
            end_token: 结束标记
        Returns:
            生成的序列 [batch_size, gen_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # 🔧 修复BUG: 创建src_mask以忽略padding（与训练时保持一致）
        # src_mask: [batch_size, 1, 1, src_len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Encode 源序列（传递src_mask）
        encoder_output = self.encode(src, src_mask)
        
        # 初始化目标序列（只有开始标记）
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        # 自回归生成
        with torch.no_grad():
            for _ in range(max_len - 1):
                # 生成因果掩码
                tgt_mask = generate_causal_mask(tgt.size(1)).to(device)
                
                # Decode（传递src_mask和tgt_mask）
                decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
                
                # 预测下一个token
                next_token_logits = self.output_projection(decoder_output[:, -1, :])
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                # 添加到序列
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 如果所有序列都生成了结束标记，停止
                if (next_token == end_token).all():
                    break
        
        return tgt


def visualize_model_structure():
    """可视化 Transformer 模型结构"""
    print("\n" + "="*60)
    print("Transformer 模型结构")
    print("="*60)
    
    structure = """
    
    输入序列 (Source)              目标序列 (Target)
         |                              |
         v                              v
    +----------+                  +----------+
    | Embedding|                  | Embedding|
    +----------+                  +----------+
         |                              |
         v                              v
    +----------+                  +----------+
    | Pos Enc  |                  | Pos Enc  |
    +----------+                  +----------+
         |                              |
         v                              |
    ┌──────────┐                       |
    │ Encoder  │                       |
    │  Block 1 │                       |
    └──────────┘                       |
         |                              |
         v                              |
    ┌──────────┐                       |
    │ Encoder  │                       |
    │  Block N │                       |
    └──────────┘                       |
         |                              |
         v                              v
         └─────────────>┌──────────────┐
                        │   Decoder    │
                        │   Block 1    │
                        └──────────────┘
                               |
                               v
                        ┌──────────────┐
                        │   Decoder    │
                        │   Block N    │
                        └──────────────┘
                               |
                               v
                        +--------------+
                        | Linear Layer |
                        +--------------+
                               |
                               v
                          输出 (Output)
    
    每个 Encoder Block:
      - Multi-Head Self-Attention
      - Add & Norm
      - Feed Forward Network
      - Add & Norm
    
    每个 Decoder Block:
      - Masked Multi-Head Self-Attention
      - Add & Norm
      - Multi-Head Cross-Attention
      - Add & Norm
      - Feed Forward Network
      - Add & Norm
    """
    
    print(structure)


def test_transformer():
    """测试完整的 Transformer 模型"""
    print("\n" + "="*50)
    print("测试完整 Transformer 模型")
    print("="*50)
    
    # 参数配置
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    print(f"\n模型配置:")
    print(f"  - 源词汇表大小: {src_vocab_size}")
    print(f"  - 目标词汇表大小: {tgt_vocab_size}")
    print(f"  - 模型维度: {d_model}")
    print(f"  - 注意力头数: {num_heads}")
    print(f"  - Encoder 层数: {num_encoder_layers}")
    print(f"  - Decoder 层数: {num_decoder_layers}")
    print(f"  - 前馈网络维度: {d_ff}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff
    )
    model.eval()
    
    # 创建输入数据
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    print(f"\n输入:")
    print(f"  - 源序列形状: {src.shape}")
    print(f"  - 目标序列形状: {tgt.shape}")
    
    # 生成掩码
    tgt_mask = generate_causal_mask(tgt_len)
    
    # 前向传播
    with torch.no_grad():
        output = model(src, tgt, tgt_mask=tgt_mask)
    
    print(f"\n输出:")
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 预期形状: ({batch_size}, {tgt_len}, {tgt_vocab_size})")
    
    # 验证
    assert output.shape == (batch_size, tgt_len, tgt_vocab_size), "输出形状不匹配！"
    print("✅ 形状验证通过")
    
    # 测试生成功能
    print("\n测试自回归生成...")
    with torch.no_grad():
        generated = model.generate(src, max_len=15)
    print(f"生成序列形状: {generated.shape}")
    print(f"生成的token: {generated[0, :10].tolist()}")
    print("✅ 生成测试通过")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型统计:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 可视化模型结构
    visualize_model_structure()
    
    print("\n✅ Transformer 模型测试完成！\n")


if __name__ == "__main__":
    test_transformer()

