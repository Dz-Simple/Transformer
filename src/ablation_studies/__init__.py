"""
Transformer 消融实验模块

可用的实验:
    1. ablation_1_num_heads - 注意力头数消融
    2. ablation_2_num_layers - 模型层数消融
    3. ablation_3_positional_encoding - 位置编码消融
    4. ablation_4_layer_norm - 层归一化策略消融

使用示例:
    python -m src.ablation_studies.ablation_1_num_heads
"""

__version__ = "1.0.0"

__all__ = [
    'run_ablation_num_heads',
    'run_ablation_num_layers',
    'run_ablation_positional_encoding',
    'run_ablation_layer_norm',
]
