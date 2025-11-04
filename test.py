"""
Transformer 机器翻译测试脚本

在测试集上评估训练好的模型，计算BLEU等评价指标并绘制图像
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src import Transformer, generate_causal_mask
from src.data import (
    SimpleTokenizer,
    TranslationDataset,
    collate_fn,
    load_vocabulary_from_file
)

# 配置中文字体 - 使用系统中支持中文的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    NLTK_AVAILABLE = True
    
    # 优化：只在第一次失败时才检查和下载
    # 如果import成功，说明NLTK数据大概率可用
    # 避免每次都调用 nltk.data.find() 导致卡顿
    
except ImportError:
    NLTK_AVAILABLE = False
    print("警告: NLTK未安装，将使用简化的BLEU计算")
except Exception as e:
    NLTK_AVAILABLE = False
    print(f"警告: NLTK初始化失败: {e}，将使用简化的BLEU计算")


def create_masks(src, tgt, pad_idx):
    """
    创建源序列和目标序列的掩码
    Args:
        src: 源序列 [batch, src_len]
        tgt: 目标序列 [batch, tgt_len]
        pad_idx: padding索引
    Returns:
        src_mask, tgt_mask
    """
    batch_size = src.size(0)
    src_len = src.size(1)
    tgt_len = tgt.size(1)
    
    # 源序列掩码：mask掉padding位置
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # 目标序列掩码：因果掩码 + padding掩码
    tgt_causal_mask = generate_causal_mask(tgt_len).bool().to(src.device)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1)
    tgt_mask = tgt_causal_mask.unsqueeze(0) & tgt_pad_mask.unsqueeze(2)
    
    return src_mask, tgt_mask


def load_test_data(data_dir="datasets/iwslt14"):
    """
    加载测试集数据
    """
    test_src_file = os.path.join(data_dir, "test.de")
    test_tgt_file = os.path.join(data_dir, "test.en")
    
    if not os.path.exists(test_src_file) or not os.path.exists(test_tgt_file):
        raise FileNotFoundError(f"测试数据文件不存在: {test_src_file} 或 {test_tgt_file}")
    
    print("正在加载测试集...")
    test_src = []
    test_tgt = []
    with open(test_src_file, 'r', encoding='utf-8') as f:
        test_src = [line.strip() for line in f if line.strip()]
    with open(test_tgt_file, 'r', encoding='utf-8') as f:
        test_tgt = [line.strip() for line in f if line.strip()]
    
    if len(test_src) != len(test_tgt):
        raise ValueError(f"测试集源语言和目标语言数量不匹配: {len(test_src)} vs {len(test_tgt)}")
    
    print(f"✅ 测试集加载成功: {len(test_src)} 条")
    return test_src, test_tgt


def calculate_bleu(reference, candidate, smoothing=None):
    """
    计算BLEU分数
    """
    if not NLTK_AVAILABLE:
        # 简化的BLEU计算（基于n-gram重叠）
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return 0.0
        
        # 简单的1-gram精确度
        ref_set = set(ref_tokens)
        cand_set = set(cand_tokens)
        if len(cand_set) == 0:
            return 0.0
        precision = len(ref_set & cand_set) / len(cand_set)
        
        # 简单的召回率
        recall = len(ref_set & cand_set) / len(ref_set) if len(ref_set) > 0 else 0.0
        
        # 简单的F1
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    try:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        
        if smoothing is None:
            smoothing = SmoothingFunction().method1
        
        # sentence_bleu需要reference是列表的列表
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    except Exception as e:
        # 如果NLTK数据缺失，降级到简化计算
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return 0.0
        ref_set = set(ref_tokens)
        cand_set = set(cand_tokens)
        if len(cand_set) == 0:
            return 0.0
        precision = len(ref_set & cand_set) / len(cand_set)
        recall = len(ref_set & cand_set) / len(ref_set) if len(ref_set) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def calculate_meteor(reference, candidate):
    """
    计算METEOR分数
    """
    if not NLTK_AVAILABLE:
        return 0.0
    
    try:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        return meteor_score([ref_tokens], cand_tokens)
    except Exception as e:
        # METEOR可能需要额外的NLTK数据（wordnet），失败时返回0
        return 0.0


def translate_batch(model, src_batch, src_vocab, tgt_vocab, tokenizer, device, max_len=80):
    """
    批量翻译句子
    """
    model.eval()
    translations = []
    
    with torch.no_grad():
        for src_text in src_batch:
            # 分词和编码
            src_tokens = tokenizer.tokenize(src_text)
            src_indices = src_vocab.encode(src_tokens, add_special_tokens=True)
            src = torch.tensor([src_indices]).to(device)
            
            # 生成
            generated = model.generate(
                src,
                max_len=max_len,
                start_token=tgt_vocab.sos_idx,
                end_token=tgt_vocab.eos_idx
            )
            
            # 解码
            tgt_indices = generated[0].tolist()
            tgt_tokens = tgt_vocab.decode(tgt_indices, skip_special_tokens=True)
            translation = tokenizer.detokenize(tgt_tokens)
            translations.append(translation)
    
    return translations


def evaluate_on_test_set(model, test_src, test_tgt, src_vocab, tgt_vocab, tokenizer, device, batch_size=16, max_samples=None):
    """
    在测试集上评估模型
    """
    print("\n开始测试集评估...")
    
    # 限制测试样本数量（用于快速测试）
    if max_samples is not None:
        test_src = test_src[:max_samples]
        test_tgt = test_tgt[:max_samples]
        print(f"限制测试样本数: {len(test_src)}")
    
    model.eval()
    all_translations = []
    all_references = []
    
    # 批量翻译
    print("正在生成翻译...")
    for i in tqdm(range(0, len(test_src), batch_size), desc="翻译进度"):
        batch_src = test_src[i:i+batch_size]
        batch_translations = translate_batch(model, batch_src, src_vocab, tgt_vocab, tokenizer, device)
        all_translations.extend(batch_translations)
        all_references.extend(test_tgt[i:i+batch_size])
    
    # 对参考翻译也进行 detokenize（移除 BPE 标记）
    print("\n处理参考翻译（移除BPE标记）...")
    all_references_detok = []
    for ref in all_references:
        # 对参考翻译进行简单的分词然后detokenize，以移除@@标记
        ref_tokens = ref.split()  # 简单空格分词
        ref_detok = tokenizer.detokenize(ref_tokens)
        all_references_detok.append(ref_detok)
    
    # 计算评价指标
    print("\n正在计算评价指标...")
    bleu_scores = []
    meteor_scores = []
    
    smoothing = SmoothingFunction().method1 if NLTK_AVAILABLE else None
    
    for ref, cand in tqdm(zip(all_references_detok, all_translations), desc="计算指标", total=len(all_references_detok)):
        bleu = calculate_bleu(ref, cand, smoothing)
        bleu_scores.append(bleu)
        
        if NLTK_AVAILABLE:
            meteor = calculate_meteor(ref, cand)
            meteor_scores.append(meteor)
    
    # 计算平均值
    avg_bleu = np.mean(bleu_scores)
    avg_meteor = np.mean(meteor_scores) if meteor_scores else 0.0
    
    return {
        'translations': all_translations,
        'references': all_references_detok,  # 返回处理后的参考翻译
        'bleu_scores': bleu_scores,
        'meteor_scores': meteor_scores,
        'avg_bleu': avg_bleu,
        'avg_meteor': avg_meteor
    }


def plot_evaluation_results(results, save_dir):
    """
    绘制评估结果图像
    参考train.py的绘图风格，图片大小保持一致
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    plt.rcParams.update({'font.size': 10})
    
    # BLEU分数分布
    ax1 = axes[0]
    bleu_scores = results['bleu_scores']
    ax1.hist(bleu_scores, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(results['avg_bleu'], color='red', linestyle='--', linewidth=2, label=f'平均BLEU: {results["avg_bleu"]:.4f}')
    ax1.set_xlabel('BLEU分数')
    ax1.set_ylabel('样本数量')
    ax1.set_title('BLEU分数分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # METEOR分数分布（如果可用）
    ax2 = axes[1]
    if results['meteor_scores'] and len(results['meteor_scores']) > 0:
        meteor_scores = results['meteor_scores']
        ax2.hist(meteor_scores, bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
        ax2.axvline(results['avg_meteor'], color='blue', linestyle='--', linewidth=2, label=f'平均METEOR: {results["avg_meteor"]:.4f}')
        ax2.set_xlabel('METEOR分数')
        ax2.set_ylabel('样本数量')
        ax2.set_title('METEOR分数分布')
        ax2.legend()
    else:
        # 如果没有METEOR分数，显示BLEU分数的箱线图
        ax2.boxplot(bleu_scores, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('BLEU分数')
        ax2.set_title('BLEU分数箱线图')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_evaluation_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  评估结果图表已保存到: {save_path}")


def save_evaluation_log(results, save_dir):
    """
    保存评估日志
    """
    # 创建详细的数据框
    data = {
        '参考翻译': results['references'],
        '模型翻译': results['translations'],
        'BLEU分数': [f'{score:.4f}' for score in results['bleu_scores']]
    }
    
    if results['meteor_scores']:
        data['METEOR分数'] = [f'{score:.4f}' for score in results['meteor_scores']]
    
    df = pd.DataFrame(data)
    
    # 保存为CSV
    csv_path = os.path.join(save_dir, 'test_evaluation_log.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  详细评估日志已保存到: {csv_path}")
    
    # 保存汇总报告
    summary = {
        '评价指标': ['平均BLEU分数', '平均METEOR分数', '测试样本数'],
        '数值': [
            f'{results["avg_bleu"]:.4f}',
            f'{results["avg_meteor"]:.4f}' if results['meteor_scores'] else 'N/A',
            len(results['references'])
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    
    # 保存汇总为CSV
    summary_csv_path = os.path.join(save_dir, 'test_evaluation_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    
    # 保存为Markdown
    md_path = os.path.join(save_dir, 'test_evaluation_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 测试集评估结果汇总\n\n")
        f.write(f"## 评价指标\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## 详细结果\n\n")
        f.write("详细结果请查看 test_evaluation_log.csv 文件。\n")
    
    print(f"  评估汇总已保存到: {md_path}")


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='测试Transformer机器翻译模型')
    parser.add_argument('--model_path', type=str, default='results/models_20251103_204641/best_model.pt',
                       help='训练好的模型路径 (例如: results/models_20251102_153059/best_model.pt)')
    parser.add_argument('--data_dir', type=str, default='datasets/iwslt14',
                       help='数据集目录路径')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='翻译时的批量大小（注意：当前实现是逐句翻译，batch_size主要影响内存循环次数）')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='限制测试样本数量（用于快速测试，None表示使用全部）')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda:0, cpu等)，None表示自动选择')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Transformer 机器翻译模型测试")
    print("="*60)
    
    # 设置设备
    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n使用设备: {device}")
    
    # 加载模型
    print(f"\n正在加载模型: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    print(f"模型配置:")
    print(f"  d_model: {config['d_model']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  num_encoder_layers: {config['num_encoder_layers']}")
    print(f"  num_decoder_layers: {config['num_decoder_layers']}")
    print(f"  batch_size: {config['batch_size']}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        pad_idx=src_vocab.pad_idx  # 🔧 修复: 传递padding索引
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ 模型加载成功")
    
    # 创建分词器
    tokenizer = SimpleTokenizer(lowercase=False)
    
    # 加载测试数据
    test_src, test_tgt = load_test_data(args.data_dir)
    
    # 评估模型
    results = evaluate_on_test_set(
        model, test_src, test_tgt, src_vocab, tgt_vocab, 
        tokenizer, device, batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("测试集评估结果")
    print("="*60)
    print(f"\n平均BLEU分数: {results['avg_bleu']:.4f}")
    if results['meteor_scores']:
        print(f"平均METEOR分数: {results['avg_meteor']:.4f}")
    print(f"测试样本数: {len(results['references'])}")
    
    # 保存结果
    model_dir = os.path.dirname(args.model_path)
    test_results_dir = os.path.join(model_dir, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    print("\n保存评估结果...")
    plot_evaluation_results(results, test_results_dir)
    save_evaluation_log(results, test_results_dir)
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print(f"\n结果保存位置: {test_results_dir}")


if __name__ == "__main__":
    main()

