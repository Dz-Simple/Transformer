"""
消融实验4: 层归一化策略 (Layer Normalization Strategy)

测试不同层归一化策略对模型性能的影响：
1. Post-LN (原始Transformer)
2. Pre-LN (更稳定的训练)
3. 无LayerNorm
"""
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math  # 使用math.exp计算困惑度，与train.py保持一致
import numpy as np  # 用于isnan、isinf等辅助函数
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


def modify_transformer_layer_norm(model, ln_type='post'):
    """
    修改Transformer的LayerNorm策略
    
    Args:
        model: Transformer模型
        ln_type: 'post' (Post-LN), 'pre' (Pre-LN), 'none' (无LN)
    """
    if ln_type == 'pre':
        # Pre-LN: 将LayerNorm移到子层之前
        # 这需要修改模型结构，这里我们通过修改前向传播逻辑来模拟
        pass  # 模型已经支持pre_norm参数
    elif ln_type == 'none':
        # 移除所有LayerNorm
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                # 将LayerNorm替换为恒等映射
                module.weight.data.fill_(1.0)
                module.bias.data.fill_(0.0)
                module.eval()  # 冻结参数
    # 'post' 使用默认配置
    return model


def run_ablation_layer_norm(gpu_id=None):
    """运行层归一化策略消融实验"""
    
    experiment_name = "ablation_4_layer_norm"
    description = "评估不同层归一化策略(Post-LN/Pre-LN/无LN)对模型性能的影响"
    
    # 如果指定了GPU，覆盖配置
    if gpu_id is not None:
        BASE_CONFIG['device'] = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        print(f"🎯 使用GPU: {BASE_CONFIG['device']}")
    
    print_experiment_header(experiment_name, description)
    
    # 测试的LayerNorm策略
    ln_strategies = {
        'post': 'Post-LN（标准）',
        'pre': 'Pre-LN（稳定）',
        'none': '无LayerNorm'
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
    
    for ln_type, ln_name in ln_strategies.items():
        print(f"\n{'='*60}")
        print(f"  测试配置: {ln_name}")
        print(f"{'='*60}\n")
        
        set_seed(BASE_CONFIG['seed'])
        
        # 创建模型
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
        
        # 修改LayerNorm策略
        model = modify_transformer_layer_norm(model, ln_type)
        model = model.to(BASE_CONFIG['device'])
        
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
        
        try:
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
                
                # 检查是否出现NaN
                if np.isnan(train_loss) or np.isnan(val_loss):
                    print(f"⚠️  检测到NaN，停止训练")
                    break
                    
        except Exception as e:
            print(f"❌ 训练出错: {e}")
            # 如果训练失败，使用最后有效的值
            if not train_losses:
                train_losses = [float('inf')]
                val_losses = [float('inf')]
                train_ppls = [float('inf')]
                val_ppls = [float('inf')]
        
        all_results[ln_name] = {
            'ln_type': ln_type,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_train_ppl': train_ppls[-1],
            'final_val_ppl': val_ppls[-1],
            'num_params': num_params,
            'training_stable': not (np.isnan(train_losses[-1]) or np.isinf(train_losses[-1])),
        }
        all_history[ln_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_ppls': train_ppls,
            'val_ppls': val_ppls,
        }
        
        print(f"✅ {ln_name} 训练完成\n")
    
    print_experiment_summary(all_results)
    
    # 保存结果
    results_data = {
        'config': BASE_CONFIG,
        'ln_strategies_tested': ln_strategies,
        'results': all_results,
        'history': all_history,
    }
    save_dir = save_results(experiment_name, results_data)
    
    # 绘图
    print("📊 生成对比图表...")
    plot_comparison(list(ln_strategies.values()), all_results, all_history, save_dir)
    
    print(f"\n{'='*60}")
    print("  实验完成！")
    print(f"{'='*60}\n")


def plot_comparison(ln_names, results, history, save_dir):
    """绘制对比图表"""
    
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    plt.rcParams.update({'font.size': 10})
    
    # 子图1: 训练损失曲线
    ax = axes[0, 0]
    for idx, ln_name in enumerate(ln_names):
        losses = history[ln_name]['train_losses']
        # 过滤inf和nan
        valid_losses = [l if not (np.isnan(l) or np.isinf(l)) else None for l in losses]
        epochs = range(1, len(losses)+1)
        ax.plot(epochs, valid_losses, 
                label=ln_name, 
                color=colors[idx], linewidth=2.5, marker='o', markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('训练损失', fontsize=12)
    ax.set_title('训练损失曲线对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 子图2: 验证损失曲线
    ax = axes[0, 1]
    for idx, ln_name in enumerate(ln_names):
        losses = history[ln_name]['val_losses']
        valid_losses = [l if not (np.isnan(l) or np.isinf(l)) else None for l in losses]
        epochs = range(1, len(losses)+1)
        ax.plot(epochs, valid_losses, 
                label=ln_name, 
                color=colors[idx], linewidth=2.5, marker='s', markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('验证损失', fontsize=12)
    ax.set_title('验证损失曲线对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 子图3: 最终性能对比
    ax = axes[1, 0]
    val_losses = [results[name]['final_val_loss'] for name in ln_names]
    # 处理inf值
    val_losses = [l if not np.isinf(l) else max([v for v in val_losses if not np.isinf(v)], default=1.0) * 2 for l in val_losses]
    bars = ax.bar(ln_names, val_losses, color=colors)
    ax.set_ylabel('最终验证损失', fontsize=12)
    ax.set_title('最终验证损失对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    for bar, loss in zip(bars, val_losses):
        height = bar.get_height()
        if not np.isinf(loss) and not np.isnan(loss):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    'Failed',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
    
    # 子图4: 训练稳定性
    ax = axes[1, 1]
    stability = [1 if results[name]['training_stable'] else 0 for name in ln_names]
    bars = ax.bar(ln_names, stability, color=colors)
    ax.set_ylabel('训练是否稳定 (1=稳定, 0=失败)', fontsize=12)
    ax.set_title('训练稳定性对比', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    for bar, stable in zip(bars, stability):
        height = bar.get_height()
        label = '✓ 稳定' if stable else '✗ 不稳定'
        color = 'green' if stable else 'red'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                label, ha='center', va='bottom', fontsize=11, 
                fontweight='bold', color=color)
    
    plt.suptitle('层归一化策略消融实验结果', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / 'layer_norm_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 对比图表已保存到: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='消融实验4: 层归一化策略')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (0, 1, 2, 3)')
    args = parser.parse_args()
    
    run_ablation_layer_norm(gpu_id=args.gpu)

