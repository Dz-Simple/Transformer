"""
消融实验3: 位置编码 (Positional Encoding)

测试不同位置编码策略对模型性能的影响：
1. 标准正弦位置编码
2. 可学习位置编码
3. 无位置编码
"""
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math  # 使用math.exp而非np.exp，与train.py保持一致
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ablation_studies.ablation_config import (
    BASE_CONFIG, set_seed, save_results, prepare_data, 
    print_experiment_header, print_experiment_summary, PLOT_CONFIG
)
from src.transformer import Transformer
# from src.data.dataset import create_dataloaders
from train import train_epoch, evaluate

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NoPositionalEncoding(nn.Module):
    """无位置编码（仅保留dropout）"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(NoPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        return self.dropout(x)


def create_transformer_with_pe_type(src_vocab_size, tgt_vocab_size, pad_idx, pe_type='sinusoidal'):
    """创建指定位置编码类型的Transformer"""
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=BASE_CONFIG['d_model'],
        num_heads=BASE_CONFIG['num_heads'],
        num_encoder_layers=BASE_CONFIG['num_encoder_layers'],
        num_decoder_layers=BASE_CONFIG['num_decoder_layers'],
        d_ff=BASE_CONFIG['d_ff'],
        dropout=BASE_CONFIG['dropout'],
        max_seq_len=BASE_CONFIG['max_seq_len'],
        pad_idx=pad_idx  # 添加pad_idx参数
    )
    
    # 替换位置编码模块
    if pe_type == 'learnable':
        model.encoder.pos_encoding = LearnablePositionalEncoding(
            BASE_CONFIG['d_model'], 
            BASE_CONFIG['max_seq_len'], 
            BASE_CONFIG['dropout']
        )
        model.decoder.pos_encoding = LearnablePositionalEncoding(
            BASE_CONFIG['d_model'], 
            BASE_CONFIG['max_seq_len'], 
            BASE_CONFIG['dropout']
        )
    elif pe_type == 'none':
        model.encoder.pos_encoding = NoPositionalEncoding(
            BASE_CONFIG['d_model'], 
            BASE_CONFIG['max_seq_len'], 
            BASE_CONFIG['dropout']
        )
        model.decoder.pos_encoding = NoPositionalEncoding(
            BASE_CONFIG['d_model'], 
            BASE_CONFIG['max_seq_len'], 
            BASE_CONFIG['dropout']
        )
    # 'sinusoidal' 使用默认的位置编码
    
    return model


def run_ablation_positional_encoding(gpu_id=None):
    """运行位置编码消融实验"""
    
    experiment_name = "ablation_3_positional_encoding"
    description = "评估不同位置编码策略(正弦/可学习/无)对模型性能的影响"
    
    # 如果指定了GPU，覆盖配置
    if gpu_id is not None:
        BASE_CONFIG['device'] = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        print(f"🎯 使用GPU: {BASE_CONFIG['device']}")
    
    print_experiment_header(experiment_name, description)
    
    # 测试的位置编码类型
    pe_types = {
        'sinusoidal': '正弦位置编码',
        'learnable': '可学习位置编码',
        'none': '无位置编码'
    }
    
    # 准备数据（使用真实IWSLT14数据集）
    train_loader, val_loader, src_vocab_size, tgt_vocab_size, pad_idx = prepare_data(
        max_vocab_size=BASE_CONFIG['max_vocab_size'],
        batch_size=BASE_CONFIG['batch_size'],
        max_samples=BASE_CONFIG['max_samples'],
        data_dir=BASE_CONFIG['data_dir']
    )
    
    # 损失函数（使用与train.py一致的LabelSmoothing）
    from train import LabelSmoothing
    criterion = LabelSmoothing(
        vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        smoothing=0.05
    )
    
    all_results = {}
    all_history = {}
    
    for pe_type, pe_name in pe_types.items():
        print(f"\n{'='*60}")
        print(f"  测试配置: {pe_name}")
        print(f"{'='*60}\n")
        
        set_seed(BASE_CONFIG['seed'])
        
        # 创建模型
        model = create_transformer_with_pe_type(
            src_vocab_size, tgt_vocab_size, pad_idx, pe_type
        ).to(BASE_CONFIG['device'])
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {num_params:,}")
        
        # 优化器（使用与train.py一致的论文配置）
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=BASE_CONFIG['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 训练
        train_losses = []
        val_losses = []
        train_ppls = []
        val_ppls = []
        
        print(f"\n开始训练 ({BASE_CONFIG['num_epochs']} epochs)...")
        for epoch in range(1, BASE_CONFIG['num_epochs'] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, BASE_CONFIG['device'], pad_idx)
            val_loss = evaluate(model, val_loader, criterion, BASE_CONFIG['device'], pad_idx)
            
            # 计算困惑度（与train.py保持一致，添加溢出保护）
            train_ppl = math.exp(train_loss) if train_loss < 10 else float('inf')
            val_ppl = math.exp(val_loss) if val_loss < 10 else float('inf')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ppls.append(train_ppl)
            val_ppls.append(val_ppl)
            
            print(f"Epoch {epoch}/{BASE_CONFIG['num_epochs']}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val PPL={val_ppl:.2f}")
        
        all_results[pe_name] = {
            'pe_type': pe_type,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_train_ppl': train_ppls[-1],
            'final_val_ppl': val_ppls[-1],
            'num_params': num_params,
        }
        all_history[pe_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_ppls': train_ppls,
            'val_ppls': val_ppls,
        }
        
        print(f"✅ {pe_name} 训练完成\n")
    
    print_experiment_summary(all_results)
    
    # 保存结果
    results_data = {
        'config': BASE_CONFIG,
        'pe_types_tested': pe_types,
        'results': all_results,
        'history': all_history,
    }
    save_dir = save_results(experiment_name, results_data)
    
    # 绘图
    print("📊 生成对比图表...")
    plot_comparison(list(pe_types.values()), all_results, all_history, save_dir)
    
    print(f"\n{'='*60}")
    print("  实验完成！")
    print(f"{'='*60}\n")


def plot_comparison(pe_names, results, history, save_dir):
    """绘制对比图表"""
    
    colors = ['#1f77b4', '#ff7f0e', '#d62728']  # 蓝、橙、红
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    plt.rcParams.update({'font.size': 10})
    
    # 子图1: 训练损失曲线
    ax = axes[0, 0]
    for idx, pe_name in enumerate(pe_names):
        losses = history[pe_name]['train_losses']
        ax.plot(range(1, len(losses)+1), losses, 
                label=pe_name, 
                color=colors[idx], linewidth=2.5, marker='o', markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('训练损失', fontsize=12)
    ax.set_title('训练损失曲线对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 子图2: 验证损失曲线
    ax = axes[0, 1]
    for idx, pe_name in enumerate(pe_names):
        losses = history[pe_name]['val_losses']
        ax.plot(range(1, len(losses)+1), losses, 
                label=pe_name, 
                color=colors[idx], linewidth=2.5, marker='s', markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('验证损失', fontsize=12)
    ax.set_title('验证损失曲线对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 子图3: 最终性能对比
    ax = axes[1, 0]
    val_losses = [results[name]['final_val_loss'] for name in pe_names]
    bars = ax.bar(pe_names, val_losses, color=colors)
    ax.set_ylabel('最终验证损失', fontsize=12)
    ax.set_title('最终验证损失对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 子图4: 困惑度对比
    ax = axes[1, 1]
    val_ppls = [results[name]['final_val_ppl'] for name in pe_names]
    bars = ax.bar(pe_names, val_ppls, color=colors)
    ax.set_ylabel('最终验证困惑度', fontsize=12)
    ax.set_title('最终验证困惑度对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',  # 使用6位小数以显示细微差异
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('位置编码消融实验结果', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / 'position_encoding_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 对比图表已保存到: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='消融实验3: 位置编码')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (0, 1, 2, 3)')
    args = parser.parse_args()
    
    run_ablation_positional_encoding(gpu_id=args.gpu)

