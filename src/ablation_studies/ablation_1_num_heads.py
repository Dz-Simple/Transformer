"""
消融实验1: 注意力头数 (Number of Attention Heads)

测试不同的注意力头数对模型性能的影响
"""
import sys
import torch
import matplotlib.pyplot as plt
import math  # 使用math.exp而非np.exp，与train.py保持一致
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ablation_studies.ablation_config import (
    BASE_CONFIG, set_seed, save_results, prepare_data,
    print_experiment_header, print_experiment_summary, PLOT_CONFIG
)
from src.transformer import Transformer
from train import train_epoch, evaluate

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_ablation_num_heads(gpu_id=None):
    """运行注意力头数消融实验"""
    
    experiment_name = "ablation_1_num_heads"
    description = "评估不同注意力头数(1, 2, 4, 8)对模型性能的影响"
    
    # 如果指定了GPU，覆盖配置
    if gpu_id is not None:
        BASE_CONFIG['device'] = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        print(f"🎯 使用GPU: {BASE_CONFIG['device']}")
    
    print_experiment_header(experiment_name, description)
    
    # 测试的头数配置（确保能被d_model整除）
    num_heads_list = [1, 2, 4, 8]  # 完整测试配置
    
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
    
    # 存储所有结果
    all_results = {}
    all_history = {}
    
    # 对每个头数配置进行实验
    for num_heads in num_heads_list:
        print(f"\n{'='*60}")
        print(f"  测试配置: {num_heads} 个注意力头")
        print(f"{'='*60}\n")
        
        set_seed(BASE_CONFIG['seed'])
        
        # 创建模型
        config = {
            **BASE_CONFIG,
            'num_heads': num_heads,
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
        }
        
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config['d_model'],
            num_heads=num_heads,
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            pad_idx=pad_idx  # 添加pad_idx参数
        ).to(config['device'])
        
        # 计算参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {num_params:,}")
        
        # 优化器（使用与train.py一致的论文配置）
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 训练历史
        train_losses = []
        val_losses = []
        train_ppls = []
        val_ppls = []
        
        # 训练循环
        print(f"\n开始训练 ({config['num_epochs']} epochs)...")
        for epoch in range(1, config['num_epochs'] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, config['device'], pad_idx)
            val_loss = evaluate(model, val_loader, criterion, config['device'], pad_idx)
            
            # 计算困惑度（与train.py保持一致，添加溢出保护）
            train_ppl = math.exp(train_loss) if train_loss < 10 else float('inf')
            val_ppl = math.exp(val_loss) if val_loss < 10 else float('inf')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ppls.append(train_ppl)
            val_ppls.append(val_ppl)
            
            print(f"Epoch {epoch}/{config['num_epochs']}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val PPL={val_ppl:.2f}")
        
        # 保存结果
        config_name = f"{num_heads}_heads"
        all_results[config_name] = {
            'num_heads': num_heads,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_train_ppl': train_ppls[-1],
            'final_val_ppl': val_ppls[-1],
            'num_params': num_params,
        }
        all_history[config_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_ppls': train_ppls,
            'val_ppls': val_ppls,
        }
        
        print(f"✅ {num_heads} 个头训练完成\n")
    
    # 打印结果摘要
    print_experiment_summary(all_results)
    
    # 保存结果
    results_data = {
        'config': BASE_CONFIG,
        'num_heads_tested': num_heads_list,
        'results': all_results,
        'history': all_history,
    }
    save_dir = save_results(experiment_name, results_data)
    
    # 绘制对比图
    print("📊 生成对比图表...")
    plot_comparison(num_heads_list, all_results, all_history, save_dir)
    
    print(f"\n{'='*60}")
    print("  实验完成！")
    print(f"{'='*60}\n")


def plot_comparison(num_heads_list, results, history, save_dir):
    """绘制对比图表"""
    
    colors = PLOT_CONFIG['colors']
    
    # 图1: 训练和验证损失曲线
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    plt.rcParams.update({'font.size': 10})
    
    # 子图1: 训练损失
    ax = axes[0, 0]
    for idx, num_heads in enumerate(num_heads_list):
        config_name = f"{num_heads}_heads"
        losses = history[config_name]['train_losses']
        ax.plot(range(1, len(losses)+1), losses, 
                label=f'{num_heads} heads', 
                color=colors[idx], linewidth=2, marker='o')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('训练损失', fontsize=12)
    ax.set_title('训练损失曲线对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图2: 验证损失
    ax = axes[0, 1]
    for idx, num_heads in enumerate(num_heads_list):
        config_name = f"{num_heads}_heads"
        losses = history[config_name]['val_losses']
        ax.plot(range(1, len(losses)+1), losses, 
                label=f'{num_heads} heads', 
                color=colors[idx], linewidth=2, marker='s')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('验证损失', fontsize=12)
    ax.set_title('验证损失曲线对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图3: 最终性能对比（柱状图）
    ax = axes[1, 0]
    val_losses = [results[f"{h}_heads"]['final_val_loss'] for h in num_heads_list]
    bars = ax.bar([str(h) for h in num_heads_list], val_losses, color=colors[:len(num_heads_list)])
    ax.set_xlabel('注意力头数', fontsize=12)
    ax.set_ylabel('最终验证损失', fontsize=12)
    ax.set_title('最终验证损失对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # 子图4: 参数量对比
    ax = axes[1, 1]
    params = [results[f"{h}_heads"]['num_params']/1e6 for h in num_heads_list]
    ax.plot([str(h) for h in num_heads_list], params, 
            marker='D', markersize=10, linewidth=2, color=colors[3])
    ax.set_xlabel('注意力头数', fontsize=12)
    ax.set_ylabel('参数量 (M)', fontsize=12)
    ax.set_title('模型参数量对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标注数值
    for i, (heads, param) in enumerate(zip(num_heads_list, params)):
        ax.text(i, param, f'{param:.2f}M', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = save_dir / 'comparison_plots.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 对比图表已保存到: {save_path}")
    plt.close()
    
    # 图2: 困惑度对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    plt.rcParams.update({'font.size': 10})
    
    # 验证困惑度曲线
    for idx, num_heads in enumerate(num_heads_list):
        config_name = f"{num_heads}_heads"
        ppls = history[config_name]['val_ppls']
        ax1.plot(range(1, len(ppls)+1), ppls, 
                label=f'{num_heads} heads', 
                color=colors[idx], linewidth=2, marker='o')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('验证困惑度', fontsize=12)
    ax1.set_title('验证困惑度曲线对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 最终困惑度柱状图
    final_ppls = [results[f"{h}_heads"]['final_val_ppl'] for h in num_heads_list]
    bars = ax2.bar([str(h) for h in num_heads_list], final_ppls, 
                   color=colors[:len(num_heads_list)])
    ax2.set_xlabel('注意力头数', fontsize=12)
    ax2.set_ylabel('最终验证困惑度', fontsize=12)
    ax2.set_title('最终验证困惑度对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',  # 使用6位小数以显示细微差异
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = save_dir / 'perplexity_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 困惑度对比图已保存到: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='消融实验1: 注意力头数')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (0, 1, 2, 3)')
    args = parser.parse_args()
    
    run_ablation_num_heads(gpu_id=args.gpu)

