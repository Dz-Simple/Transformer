"""
Transformer 模型实现

包含完整的 Transformer 模型及其所有组件：
- components/: 基础构件（位置编码、注意力机制、前馈网络等）
- encoder.py: Transformer Encoder
- decoder.py: Transformer Decoder
- transformer.py: 完整的 Transformer 模型
"""

__version__ = "2.0.0"
__author__ = "Your Name"

# 导入基础组件
from .components import (
    PositionalEncoding,
    SelfAttention,
    MultiHeadAttention,
    PositionWiseFeedForward,
    PositionWiseFeedForwardGELU,
    LayerNorm,
    ResidualConnection,
    SublayerConnection
)

# 导入 Encoder 和 Decoder
from .encoder import Encoder, EncoderBlock
from .decoder import Decoder, DecoderBlock, generate_causal_mask

# 导入完整的 Transformer 模型
from .transformer import Transformer

__all__ = [
    # 基础组件
    'PositionalEncoding',
    'SelfAttention',
    'MultiHeadAttention',
    'PositionWiseFeedForward',
    'PositionWiseFeedForwardGELU',
    'LayerNorm',
    'ResidualConnection',
    'SublayerConnection',
    
    # Encoder 和 Decoder
    'Encoder',
    'EncoderBlock',
    'Decoder',
    'DecoderBlock',
    'generate_causal_mask',
    
    # 完整模型
    'Transformer',
]
