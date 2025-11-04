"""
消融实验通用配置
"""
import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# 导入数据处理模块
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data import (
    SimpleTokenizer, TranslationDataset, collate_fn,
    load_iwslt14_de_en, load_vocabulary_from_file
)

# 基础配置
BASE_CONFIG = {
    # 数据参数
    'max_vocab_size': 10000,
    'batch_size': 64,
    'max_seq_len': 5000,  # 与train.py保持一致（Transformer默认值）
    'max_samples': 50000,  # 使用50000条数据
    'data_dir': 'datasets/iwslt14',  # 数据集路径
    
    # 模型参数（GPU优化配置）
    'd_model': 256,  # 标准模型大小
    'num_heads': 4,
    'num_encoder_layers': 3,  # 标准层数
    'num_decoder_layers': 3,  # 标准层数
    'd_ff': 1024,  # 标准FFN维度
    'dropout': 0.1,
    
    # 训练参数
    'num_epochs': 10,  # 统一使用10个epoch
    'learning_rate': 0.0001,
    
    # 其他
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'num_runs': 1,  # 每个配置运行次数（可设为3取平均）
}

# 结果保存路径
ABLATION_RESULTS_DIR = Path('results')
ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def prepare_data(max_vocab_size=10000, batch_size=64, max_samples=50000, data_dir='datasets/iwslt14'):
    """
    准备训练和验证数据（使用真实IWSLT14数据集）
    
    Args:
        max_vocab_size: 最大词汇表大小
        batch_size: 批量大小
        max_samples: 最大训练样本数（None表示使用全部）
        data_dir: 数据集目录
    
    Returns:
        train_loader, val_loader, src_vocab_size, tgt_vocab_size, pad_idx
    """
    print(f"📦 加载IWSLT14数据集...")
    print(f"  数据目录: {data_dir}")
    print(f"  最大样本数: {max_samples if max_samples else '全部'}")
    
    # 创建分词器
    tokenizer = SimpleTokenizer(lowercase=False)
    
    # 加载真实数据集
    train_src, train_tgt, val_src, val_tgt = load_iwslt14_de_en(
        data_dir=data_dir,
        max_samples=max_samples
    )
    
    print(f"  训练集大小: {len(train_src)}")
    print(f"  验证集大小: {len(val_src)}")
    
    # 加载词汇表
    src_vocab = load_vocabulary_from_file(f"{data_dir}/vocab.de")
    tgt_vocab = load_vocabulary_from_file(f"{data_dir}/vocab.en")
    
    print(f"  源词汇表大小: {len(src_vocab)}")
    print(f"  目标词汇表大小: {len(tgt_vocab)}")
    
    # 创建数据集
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, tokenizer)
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=src_vocab.pad_idx),
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=src_vocab.pad_idx),
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(src_vocab), len(tgt_vocab), src_vocab.pad_idx
    

def save_results(experiment_name, results):
    """保存实验结果到JSON文件"""
    save_dir = ABLATION_RESULTS_DIR / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = save_dir / 'results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存到: {results_file}")
    return save_dir


def load_results(experiment_name):
    """加载已保存的实验结果"""
    results_file = ABLATION_RESULTS_DIR / experiment_name / 'results.json'
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def print_experiment_header(experiment_name, description):
    """打印实验标题"""
    print("\n" + "="*70)
    print(f"  消融实验: {experiment_name}")
    print("="*70)
    print(f"描述: {description}")
    print(f"设备: {BASE_CONFIG['device']}")
    print(f"基础配置: d_model={BASE_CONFIG['d_model']}, "
          f"epochs={BASE_CONFIG['num_epochs']}, "
          f"max_samples={BASE_CONFIG['max_samples']}")
    print("="*70 + "\n")


def print_experiment_summary(results):
    """打印实验结果摘要"""
    print("\n" + "="*70)
    print("  实验结果摘要")
    print("="*70)
    
    for config_name, metrics in results.items():
        print(f"\n【{config_name}】")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[-1], (int, float)):
                        print(f"  {key} (最终): {value[-1]:.4f}")
        else:
            print(f"  结果: {metrics}")
    
    print("="*70 + "\n")


# 可视化样式配置
PLOT_CONFIG = {
    'figure_size': (7, 3.5),  # 7英寸宽度
    'dpi': 150,
    'line_width': 2,
    'marker_size': 6,
    'font_size': 10,  # 10pt字体
    'title_size': 12,
    'legend_size': 9,
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
}

