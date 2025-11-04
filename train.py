"""
Transformer 机器翻译训练脚本

使用IWSLT 2017数据集训练Transformer模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from src import Transformer, generate_causal_mask
from src.data import (
    SimpleTokenizer,
    TranslationDataset,
    collate_fn,
    load_iwslt14_de_en,
    load_vocabulary_from_file
)

# 配置中文字体 - 使用系统中支持中文的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class NoamScheduler:
    """
    Noam学习率调度器 (来自 "Attention Is All You Need" 论文)
    
    公式: lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        """
        Args:
            optimizer: 优化器
            d_model: 模型维度
            warmup_steps: warmup步数（默认4000，根据论文）
            factor: 缩放因子（默认1.0）
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        # 初始化时计算初始学习率（step=1时的学习率）
        self._rate = self._calculate_rate(1)
    
    def _calculate_rate(self, step):
        """计算指定步数的学习率"""
        return self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
    
    def step(self):
        """更新学习率"""
        self._step += 1
        rate = self._rate = self._calculate_rate(self._step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        return rate
    
    def get_lr(self):
        """获取当前学习率"""
        if self._step == 0:
            # 如果还没有执行过step，返回step=1时的学习率
            return self._calculate_rate(1)
        return self._rate
    
    def state_dict(self):
        """保存状态"""
        return {
            'step': self._step,
            'warmup_steps': self.warmup_steps,
            'factor': self.factor
        }
    
    def load_state_dict(self, state_dict):
        """加载状态"""
        self._step = state_dict['step']
        self.warmup_steps = state_dict.get('warmup_steps', 4000)
        self.factor = state_dict.get('factor', 1.0)


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
    # [batch, 1, 1, src_len]
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # 目标序列掩码：因果掩码 + padding掩码
    # 因果掩码: [1, tgt_len, tgt_len]
    tgt_causal_mask = generate_causal_mask(tgt_len).bool().to(src.device)
    
    # Padding掩码: [batch, 1, tgt_len]
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1)
    
    # 组合：[batch, 1, tgt_len, tgt_len]
    tgt_mask = tgt_causal_mask.unsqueeze(0) & tgt_pad_mask.unsqueeze(2)
    
    return src_mask, tgt_mask


class LabelSmoothing(nn.Module):
    """标签平滑"""
    
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
    
    def forward(self, x, target):
        """
        x: [batch * seq_len, vocab_size]
        target: [batch * seq_len]
        """
        assert x.size(1) == self.vocab_size
        
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        
        mask = torch.nonzero(target == self.pad_idx, as_tuple=False)
        if mask.dim() > 0 and mask.size(0) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        return self.criterion(x, true_dist)


def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx, scheduler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)
        
        # 准备decoder输入和目标
        tgt_input = tgt[:, :-1]  # 移除最后一个token
        tgt_output = tgt[:, 1:]  # 移除第一个token (SOS)
        
        # 创建掩码
        src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        # 计算损失
        # output: [batch, tgt_len-1, vocab_size]
        # tgt_output: [batch, tgt_len-1]
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output.log_softmax(dim=-1), tgt_output)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新学习率（在optimizer.step()之前调用，确保使用正确的学习率）
        if scheduler is not None:
            current_lr = scheduler.step()
        
        # 优化器更新参数（此时使用调度器更新的学习率）
        optimizer.step()
        
        # 统计
        n_tokens = (tgt_output != pad_idx).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens
        
        # 显示当前学习率
        if scheduler is not None:
            pbar.set_postfix({
                'loss': f'{loss.item() / n_tokens:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        else:
            pbar.set_postfix({'loss': f'{loss.item() / n_tokens:.4f}'})
    
    return total_loss / total_tokens


def evaluate(model, dataloader, criterion, device, pad_idx):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)
            
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output.log_softmax(dim=-1), tgt_output)
            
            n_tokens = (tgt_output != pad_idx).sum().item()
            total_loss += loss.item()
            total_tokens += n_tokens
    
    return total_loss / total_tokens


def translate(model, src_sentence, src_vocab, tgt_vocab, tokenizer, device, max_len=50):
    """
    翻译一个句子
    """
    model.eval()
    
    # 分词和编码
    src_tokens = tokenizer.tokenize(src_sentence)
    src_indices = src_vocab.encode(src_tokens, add_special_tokens=True)
    src = torch.tensor([src_indices]).to(device)
    
    # 生成
    with torch.no_grad():
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
    
    return translation


def plot_training_curves(train_losses, val_losses, train_ppls, val_ppls, save_dir):
    """
    绘制训练曲线
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_ppls: 训练困惑度列表
        val_ppls: 验证困惑度列表
        save_dir: 保存目录
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    plt.rcParams.update({'font.size': 10})
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='验证损失', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和验证损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 困惑度曲线
    ax2.plot(epochs, train_ppls, 'b-', label='训练困惑度', marker='o')
    ax2.plot(epochs, val_ppls, 'r-', label='验证困惑度', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('困惑度')
    ax2.set_title('训练和验证困惑度曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  训练曲线已保存到: {os.path.join(save_dir, 'training_curves.png')}")


def save_training_log(log_data, save_dir):
    """
    保存训练日志为CSV和Markdown表格
    Args:
        log_data: 训练日志数据列表
        save_dir: 保存目录
    """
    df = pd.DataFrame(log_data)
    
    # 保存为CSV
    csv_path = os.path.join(save_dir, 'training_log.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  训练日志已保存到: {csv_path}")
    
    # 保存为Markdown表格
    md_path = os.path.join(save_dir, 'training_log.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 训练日志\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"  Markdown表格已保存到: {md_path}")


def main():
    """主训练函数"""
    print("="*60)
    print("Transformer 机器翻译训练")
    print("="*60)
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    models_dir = os.path.join(results_dir, f"models_{timestamp}")
    plots_dir = os.path.join(results_dir, f"plots_{timestamp}")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n结果保存路径:")
    print(f"  模型目录: {models_dir}")
    print(f"  图表目录: {plots_dir}")
    
    # 超参数
    config = {
        # 模型架构（已改为Pre-LN，训练更稳定，可以适当增大）
        'd_model': 512,  # 恢复到标准Transformer的d_model
        'num_heads': 8,
        'num_encoder_layers': 6,  # 恢复到标准6层（Pre-LN更稳定）
        'num_decoder_layers': 6,  # 恢复到标准6层
        'd_ff': 2048,  # 恢复到标准FFN维度（d_model * 4）
        'dropout': 0.1,  # 降低dropout（Pre-LN架构训练更稳定，且已移除FFN额外dropout）
        
        # 训练超参数
        'batch_size': 64,  # 保持当前batch size以适应GPU显存
        'num_epochs': 30,  # 增加训练轮数，早停会自动终止
        'learning_rate': 0.0001,
        'max_vocab_size': 10000,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        
        # 学习率调度器参数
        'warmup_steps': 4000,  # 标准warmup步数（论文值）
        
        # 早停机制参数
        'early_stopping_patience': 10,  # 耐心值，允许验证集loss有10轮不改善
        'early_stopping_min_delta': 0.001,  # 最小改善量
        'early_stopping_enabled': True  # 启用早停
    }
    
    # 确保只使用一个GPU（第一个GPU）
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"\n检测到 {num_gpus} 块GPU，仅使用第一块GPU (cuda:0)")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    device = torch.device(config['device'])
    print(f"\n使用设备: {device}")
    
    # 创建分词器
    print("\n准备数据...")
    tokenizer = SimpleTokenizer(lowercase=False)
    
    # 加载真实数据集（使用全部数据）
    print("加载IWSLT14数据集...")
    train_src, train_tgt, val_src, val_tgt = load_iwslt14_de_en(
        data_dir="datasets/iwslt14",
        max_samples=None  # 使用全部160k数据
    )
    
    print(f"训练集大小: {len(train_src)}")
    print(f"验证集大小: {len(val_src)}")
    
    # 加载词汇表
    print("\n加载词汇表...")
    src_vocab = load_vocabulary_from_file("datasets/iwslt14/vocab.de")
    tgt_vocab = load_vocabulary_from_file("datasets/iwslt14/vocab.en")
    
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    # 创建数据集和数据加载器
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, tokenizer)
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=src_vocab.pad_idx),
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=src_vocab.pad_idx),
        num_workers=2,
        pin_memory=True
    )
    
    # 创建模型
    print("\n创建Transformer模型...")
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
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 损失函数和优化器
    # 降低标签平滑以避免过度正则化（0.1 -> 0.05）
    criterion = LabelSmoothing(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab.pad_idx,
        smoothing=0.05
    )
    
    # 优化器：使用较小的初始学习率（Noam调度器会自动调整）
    # 根据论文，初始学习率设置为1.0（但实际会由调度器管理）
    optimizer = optim.Adam(
        model.parameters(),
        lr=1.0,  # Noam调度器会动态调整，这里设置为1.0作为基准
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # 创建Noam学习率调度器
    # 根据数据集大小调整 warmup_steps
    # 对于大数据集，建议 warmup 约占总训练的 5-10%
    warmup_steps = config.get('warmup_steps', 8000)  # 增加到8000步（约3.2个epoch）
    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=config['d_model'],
        warmup_steps=warmup_steps,
        factor=1.0
    )
    # 立即设置初始学习率到optimizer（在第一次训练迭代之前）
    # 第一次调用scheduler.step()时_step会从0变成1，所以我们预先设置step=1的学习率
    initial_lr = scheduler._calculate_rate(1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr
    print(f"\n使用Noam学习率调度器 (warmup_steps={warmup_steps})")
    print(f"初始学习率 (step=1): {initial_lr:.2e} (已预先设置到optimizer)")
    
    # 训练
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    log_data = []
    
    # 早停机制相关变量
    early_stopping_patience = config.get('early_stopping_patience', 10)
    early_stopping_min_delta = config.get('early_stopping_min_delta', 0.001)
    early_stopping_enabled = config.get('early_stopping_enabled', True)
    patience_counter = 0  # 连续没有改善的epoch数
    best_epoch = 0  # 最佳验证损失的epoch
    
    if early_stopping_enabled:
        print(f"\n早停机制已启用:")
        print(f"  耐心值 (patience): {early_stopping_patience}")
        print(f"  最小改善量 (min_delta): {early_stopping_min_delta}")
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, src_vocab.pad_idx, scheduler)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device, src_vocab.pad_idx)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # 计算困惑度
        train_ppl = math.exp(train_loss) if train_loss < 10 else float('inf')
        val_ppl = math.exp(val_loss) if val_loss < 10 else float('inf')
        
        # 记录数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        
        # 获取当前学习率
        current_lr = scheduler.get_lr()
        
        log_data.append({
            'Epoch': epoch + 1,
            '训练损失': f'{train_loss:.4f}',
            '验证损失': f'{val_loss:.4f}',
            '训练困惑度': f'{train_ppl:.2f}',
            '验证困惑度': f'{val_ppl:.2f}',
            '学习率': f'{current_lr:.2e}',
            '用时(秒)': f'{epoch_time:.2f}'
        })
        
        print(f"\nEpoch {epoch + 1} 完成:")
        print(f"  训练损失: {train_loss:.4f} | 训练困惑度: {train_ppl:.2f}")
        print(f"  验证损失: {val_loss:.4f} | 验证困惑度: {val_ppl:.2f}")
        print(f"  学习率: {current_lr:.2e}")
        print(f"  用时: {epoch_time:.2f}秒")
        
        # 保存最佳模型和早停判断
        if val_loss < best_val_loss - early_stopping_min_delta:
            # 验证损失有明显改善
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # 重置计数器
            
            model_path = os.path.join(models_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'config': config,
            }, model_path)
            print(f"  ✓ 保存最佳模型到: {model_path} (验证损失改善: {best_val_loss:.4f})")
        else:
            # 验证损失没有改善
            if early_stopping_enabled:
                patience_counter += 1
                print(f"  验证损失未改善 (耐心计数: {patience_counter}/{early_stopping_patience})")
                
                # 检查是否需要早停
                if patience_counter >= early_stopping_patience:
                    print(f"\n{'='*60}")
                    print("触发早停机制")
                    print(f"{'='*60}")
                    print(f"验证损失已连续 {early_stopping_patience} 个epoch未改善")
                    print(f"最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
                    print(f"当前验证损失: {val_loss:.4f}")
                    print(f"训练已停止在第 {epoch + 1} 个epoch")
                    break
        
        # 保存检查点
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config['num_epochs']:
            checkpoint_path = os.path.join(models_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"  ✓ 保存检查点到: {checkpoint_path}")
    
    # 绘制训练曲线
    print("\n生成训练曲线...")
    plot_training_curves(train_losses, val_losses, train_ppls, val_ppls, plots_dir)
    
    # 保存训练日志
    print("\n保存训练日志...")
    save_training_log(log_data, plots_dir)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"总训练轮数: {len(train_losses)}/{config['num_epochs']}")
    if early_stopping_enabled and patience_counter >= early_stopping_patience:
        print(f"训练因早停机制而结束")
    print(f"最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
    if len(val_losses) > 0:
        print(f"最终验证损失: {val_losses[-1]:.4f}")
    print(f"\n结果保存位置:")
    print(f"  模型: {models_dir}")
    print(f"  图表和日志: {plots_dir}")


if __name__ == "__main__":
    main()

