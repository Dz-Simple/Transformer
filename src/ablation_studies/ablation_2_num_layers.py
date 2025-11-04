"""
消融实验2: 模型层数 (Number of Layers)

测试不同的编码器/解码器层数对模型性能的影响
"""
import sys
import torch
import matplotlib.pyplot as plt
import math  # 使用math.exp计算困惑度，与train.py保持一致
import numpy as np  # 用于isnan、isinf、mean等辅助函数
import argparse
import time
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


def run_ablation_num_layers(gpu_id=None):
    """运行层数消融实验"""
    
    experiment_name = "ablation_2_num_layers"
    description = "评估不同模型层数(1, 2, 3, 4, 6)对模型性能的影响"
    
    # 如果指定了GPU，覆盖配置
    if gpu_id is not None:
        BASE_CONFIG['device'] = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        print(f"🎯 使用GPU: {BASE_CONFIG['device']}")
    
    print_experiment_header(experiment_name, description)
    
    # 测试的层数配置  
    num_layers_list = [1, 2, 3, 4, 6]  # 完整测试配置
    
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
    
    for num_layers in num_layers_list:
        print(f"\n{'='*60}")
        print(f"  测试配置: {num_layers} 层编码器/解码器")
        print(f"{'='*60}\n")
        
        set_seed(BASE_CONFIG['seed'])
        
        # 创建模型
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=BASE_CONFIG['d_model'],
            num_heads=BASE_CONFIG['num_heads'],
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            d_ff=BASE_CONFIG['d_ff'],
            dropout=BASE_CONFIG['dropout'],
            max_seq_len=BASE_CONFIG['max_seq_len'],
            pad_idx=pad_idx  # 添加pad_idx参数
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
        epoch_times = []
        
        print(f"\n开始训练 ({BASE_CONFIG['num_epochs']} epochs)...")
        for epoch in range(1, BASE_CONFIG['num_epochs'] + 1):
            start_time = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, BASE_CONFIG['device'], pad_idx)
            val_loss = evaluate(model, val_loader, criterion, BASE_CONFIG['device'], pad_idx)
            epoch_time = time.time() - start_time
            
            # 计算困惑度（与train.py保持一致，添加溢出保护）
            train_ppl = math.exp(train_loss) if train_loss < 10 else float('inf')
            val_ppl = math.exp(val_loss) if val_loss < 10 else float('inf')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ppls.append(train_ppl)
            val_ppls.append(val_ppl)
            epoch_times.append(epoch_time)
            
            print(f"Epoch {epoch}/{BASE_CONFIG['num_epochs']}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Time={epoch_time:.1f}s")
        
        config_name = f"{num_layers}_layers"
        all_results[config_name] = {
            'num_layers': num_layers,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_train_ppl': train_ppls[-1],
            'final_val_ppl': val_ppls[-1],
            'num_params': num_params,
            'avg_epoch_time': np.mean(epoch_times),
        }
        all_history[config_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_ppls': train_ppls,
            'val_ppls': val_ppls,
            'epoch_times': epoch_times,
        }
        
        print(f"✅ {num_layers} 层训练完成\n")
    
    print_experiment_summary(all_results)
    
    # 保存结果
    results_data = {
        'config': BASE_CONFIG,
        'num_layers_tested': num_layers_list,
        'results': all_results,
        'history': all_history,
    }
    save_dir = save_results(experiment_name, results_data)
    
    # 绘图
    print("📊 生成对比图表...")
    plot_comparison(num_layers_list, all_results, all_history, save_dir)
    
    print(f"\n{'='*60}")
    print("  实验完成！")
    print(f"{'='*60}\n")


def plot_comparison(num_layers_list, results, history, save_dir):
    """绘制对比图表"""
    
    colors = PLOT_CONFIG['colors']
    
    # 图1: 性能、参数量和时间的综合对比
    fig = plt.figure(figsize=(7, 10))
    plt.rcParams.update({'font.size': 10})
    
    # 子图1: 验证损失曲线
    ax1 = plt.subplot(3, 2, 1)
    for idx, num_layers in enumerate(num_layers_list):
        config_name = f"{num_layers}_layers"
        losses = history[config_name]['val_losses']
        ax1.plot(range(1, len(losses)+1), losses, 
                label=f'{num_layers} 层', 
                color=colors[idx], linewidth=2, marker='o')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('验证损失', fontsize=11)
    ax1.set_title('验证损失曲线', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 困惑度曲线
    ax2 = plt.subplot(3, 2, 2)
    for idx, num_layers in enumerate(num_layers_list):
        config_name = f"{num_layers}_layers"
        ppls = history[config_name]['val_ppls']
        ax2.plot(range(1, len(ppls)+1), ppls, 
                label=f'{num_layers} 层', 
                color=colors[idx], linewidth=2, marker='s')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('验证困惑度', fontsize=11)
    ax2.set_title('验证困惑度曲线', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 最终性能对比
    ax3 = plt.subplot(3, 2, 3)
    val_losses = [results[f"{l}_layers"]['final_val_loss'] for l in num_layers_list]
    bars = ax3.bar([str(l) for l in num_layers_list], val_losses, 
                   color=colors[:len(num_layers_list)])
    ax3.set_xlabel('层数', fontsize=11)
    ax3.set_ylabel('最终验证损失', fontsize=11)
    ax3.set_title('最终性能对比', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 子图4: 参数量对比
    ax4 = plt.subplot(3, 2, 4)
    params = [results[f"{l}_layers"]['num_params']/1e6 for l in num_layers_list]
    ax4.plot([str(l) for l in num_layers_list], params, 
            marker='D', markersize=10, linewidth=2, color=colors[2])
    ax4.set_xlabel('层数', fontsize=11)
    ax4.set_ylabel('参数量 (M)', fontsize=11)
    ax4.set_title('模型参数量', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    for i, (layers, param) in enumerate(zip(num_layers_list, params)):
        ax4.text(i, param, f'{param:.1f}M', 
                ha='center', va='bottom', fontsize=9)
    
    # 子图5: 训练时间对比
    ax5 = plt.subplot(3, 2, 5)
    times = [results[f"{l}_layers"]['avg_epoch_time'] for l in num_layers_list]
    bars = ax5.bar([str(l) for l in num_layers_list], times, 
                   color=colors[:len(num_layers_list)])
    ax5.set_xlabel('层数', fontsize=11)
    ax5.set_ylabel('平均Epoch时间 (秒)', fontsize=11)
    ax5.set_title('训练时间对比', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 子图6: 性能-效率权衡
    ax6 = plt.subplot(3, 2, 6)
    # 绘制性能vs参数量
    ax6_twin = ax6.twinx()
    line1 = ax6.plot([str(l) for l in num_layers_list], val_losses, 
                     marker='o', markersize=10, linewidth=2, 
                     color=colors[0], label='验证损失')
    line2 = ax6_twin.plot([str(l) for l in num_layers_list], params, 
                          marker='s', markersize=10, linewidth=2, 
                          color=colors[1], label='参数量(M)')
    ax6.set_xlabel('层数', fontsize=11)
    ax6.set_ylabel('验证损失', fontsize=11, color=colors[0])
    ax6_twin.set_ylabel('参数量 (M)', fontsize=11, color=colors[1])
    ax6.set_title('性能-参数量权衡', fontsize=13, fontweight='bold')
    ax6.tick_params(axis='y', labelcolor=colors[0])
    ax6_twin.tick_params(axis='y', labelcolor=colors[1])
    ax6.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper left', fontsize=9)
    
    plt.suptitle('模型层数消融实验结果', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = save_dir / 'comprehensive_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 综合对比图已保存到: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='消融实验2: 模型层数')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (0, 1, 2, 3)')
    args = parser.parse_args()
    
    run_ablation_num_layers(gpu_id=args.gpu)

