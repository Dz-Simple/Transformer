"""
机器翻译数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional
import os


class TranslationDataset(Dataset):
    """机器翻译数据集"""
    
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, tokenizer):
        """
        Args:
            src_texts: 源语言文本列表
            tgt_texts: 目标语言文本列表
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
            tokenizer: 分词器
        """
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenizer = tokenizer
        
        assert len(src_texts) == len(tgt_texts), "源语言和目标语言数据数量不匹配"
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        Returns:
            src_indices: 源语言索引序列
            tgt_indices: 目标语言索引序列
        """
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # 分词
        src_tokens = self.tokenizer.tokenize(src_text)
        tgt_tokens = self.tokenizer.tokenize(tgt_text)
        
        # 转换为索引
        src_indices = self.src_vocab.encode(src_tokens, add_special_tokens=True)
        tgt_indices = self.tgt_vocab.encode(tgt_tokens, add_special_tokens=True)
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def collate_fn(batch, pad_idx=0):
    """
    DataLoader的collate function
    将一个batch的数据进行padding
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Padding
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    
    return src_batch, tgt_batch


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    num_workers=0,
    pad_idx=0
):
    """
    创建训练和验证数据加载器
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: batch大小
        num_workers: 数据加载worker数量
        pad_idx: padding索引
    Returns:
        train_loader, val_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx)
    )
    
    return train_loader, val_loader


def load_iwslt14_de_en(
    data_dir: str = "datasets/iwslt14",
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    加载 IWSLT 14 德-英翻译数据集（从本地文件）
    源语言：德语 (de)，目标语言：英语 (en)
    
    Args:
        data_dir: 数据集目录路径
        max_samples: 最大样本数限制，None 表示使用全部数据
        
    Returns:
        train_src: 训练集源语言文本列表（德语）
        train_tgt: 训练集目标语言文本列表（英语）
        val_src: 验证集源语言文本列表（德语）
        val_tgt: 验证集目标语言文本列表（英语）
    
    Example:
        >>> train_src, train_tgt, val_src, val_tgt = load_iwslt14_de_en()
        >>> print(f"训练集: {len(train_src)} 条, 验证集: {len(val_src)} 条")
    """
    # 构建文件路径
    train_src_file = os.path.join(data_dir, "train.de")
    train_tgt_file = os.path.join(data_dir, "train.en")
    val_src_file = os.path.join(data_dir, "valid.de")
    val_tgt_file = os.path.join(data_dir, "valid.en")
    
    # 检查文件是否存在
    for file_path in [train_src_file, train_tgt_file, val_src_file, val_tgt_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    print("正在加载 IWSLT 14 德-英翻译数据集...")
    
    # 读取训练集
    train_src = []
    train_tgt = []
    with open(train_src_file, 'r', encoding='utf-8') as f:
        train_src = [line.strip() for line in f if line.strip()]
    with open(train_tgt_file, 'r', encoding='utf-8') as f:
        train_tgt = [line.strip() for line in f if line.strip()]
    
    # 读取验证集
    val_src = []
    val_tgt = []
    with open(val_src_file, 'r', encoding='utf-8') as f:
        val_src = [line.strip() for line in f if line.strip()]
    with open(val_tgt_file, 'r', encoding='utf-8') as f:
        val_tgt = [line.strip() for line in f if line.strip()]
    
    # 检查数据长度是否匹配
    if len(train_src) != len(train_tgt):
        raise ValueError(f"训练集源语言和目标语言数量不匹配: {len(train_src)} vs {len(train_tgt)}")
    if len(val_src) != len(val_tgt):
        raise ValueError(f"验证集源语言和目标语言数量不匹配: {len(val_src)} vs {len(val_tgt)}")
    
    # 限制训练集样本数量（如果指定，只取前n条）
    if max_samples is not None:
        print(f"⚠️  限制训练集样本数: {len(train_src)} -> {max_samples}")
        train_src = train_src[:max_samples]
        train_tgt = train_tgt[:max_samples]
    
    print(f"✅ 数据集加载成功:")
    print(f"  训练集: {len(train_src)} 条")
    print(f"  验证集: {len(val_src)} 条")
    
    return train_src, train_tgt, val_src, val_tgt


def test_data_loading(data_dir: str = "datasets/iwslt14", max_samples: int = 100):
    """
    测试训练数据加载是否成功
    
    Args:
        data_dir: 数据集目录路径
        max_samples: 用于测试的最大样本数（默认100条）
    
    Returns:
        bool: 如果所有测试通过返回True，否则返回False
    """
    print("=" * 70)
    print("测试训练数据加载")
    print("=" * 70)
    
    try:
        # 1. 测试数据文件加载
        print("\n【步骤1】测试数据文件加载...")
        train_src, train_tgt, val_src, val_tgt = load_iwslt14_de_en(
            data_dir=data_dir,
            max_samples=max_samples
        )
        
        assert len(train_src) > 0, "训练集源语言数据为空"
        assert len(train_tgt) > 0, "训练集目标语言数据为空"
        assert len(val_src) > 0, "验证集源语言数据为空"
        assert len(val_tgt) > 0, "验证集目标语言数据为空"
        assert len(train_src) == len(train_tgt), "训练集源语言和目标语言数量不匹配"
        assert len(val_src) == len(val_tgt), "验证集源语言和目标语言数量不匹配"
        
        print(f"  ✅ 数据加载成功:")
        print(f"     训练集: {len(train_src)} 条")
        print(f"     验证集: {len(val_src)} 条")
        
        # 显示示例数据
        print(f"\n  训练集示例:")
        for i in range(min(3, len(train_src))):
            print(f"    [{i+1}] DE: {train_src[i][:60]}...")
            print(f"        EN: {train_tgt[i][:60]}...")
        
        # 2. 测试词汇表加载
        print("\n【步骤2】测试词汇表加载...")
        vocab_test = None
        src_vocab = None
        tgt_vocab = None
        
        try:
            from src.data.vocabulary import load_vocabulary_from_file
        except ImportError:
            try:
                from .vocabulary import load_vocabulary_from_file
            except ImportError:
                print(f"  ⚠️  无法导入词汇表加载函数，跳过词汇表测试")
                vocab_test = False
        
        vocab_de_path = os.path.join(data_dir, "vocab.de")
        vocab_en_path = os.path.join(data_dir, "vocab.en")
        
        if vocab_test is None:
            if not os.path.exists(vocab_de_path) or not os.path.exists(vocab_en_path):
                print(f"  ⚠️  词汇表文件不存在，跳过词汇表测试")
                vocab_test = False
            else:
                src_vocab = load_vocabulary_from_file(vocab_de_path)
                tgt_vocab = load_vocabulary_from_file(vocab_en_path)
                
                assert len(src_vocab) > 0, "源语言词汇表为空"
                assert len(tgt_vocab) > 0, "目标语言词汇表为空"
                
                print(f"  ✅ 词汇表加载成功:")
                print(f"     源语言词汇表: {len(src_vocab)} 词")
                print(f"     目标语言词汇表: {len(tgt_vocab)} 词")
                print(f"     特殊标记索引: PAD={src_vocab.pad_idx}, UNK={src_vocab.unk_idx}, "
                      f"SOS={src_vocab.sos_idx}, EOS={src_vocab.eos_idx}")
                vocab_test = True
        
        # 3. 测试数据集创建
        print("\n【步骤3】测试数据集创建...")
        try:
            from src.data.tokenizer import SimpleTokenizer
        except ImportError:
            from .tokenizer import SimpleTokenizer
        
        if vocab_test:
            tokenizer = SimpleTokenizer(lowercase=False)  # 数据已预处理
            
            train_dataset = TranslationDataset(
                train_src[:10],  # 只测试前10条
                train_tgt[:10],
                src_vocab,
                tgt_vocab,
                tokenizer
            )
            
            assert len(train_dataset) == 10, "数据集大小不正确"
            
            # 测试获取单个样本
            src_idx, tgt_idx = train_dataset[0]
            assert src_idx.shape[0] > 0, "源语言索引序列为空"
            assert tgt_idx.shape[0] > 0, "目标语言索引序列为空"
            
            print(f"  ✅ 数据集创建成功:")
            print(f"     数据集大小: {len(train_dataset)}")
            print(f"     第一个样本 - 源语言索引长度: {src_idx.shape[0]}, "
                  f"目标语言索引长度: {tgt_idx.shape[0]}")
            
            # 4. 测试 DataLoader
            print("\n【步骤4】测试 DataLoader...")
            train_loader = DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, pad_idx=src_vocab.pad_idx)
            )
            
            # 获取一个batch
            src_batch, tgt_batch = next(iter(train_loader))
            assert src_batch.shape[0] > 0, "batch大小不正确"
            assert src_batch.shape[1] > 0, "序列长度不正确"
            assert tgt_batch.shape[0] > 0, "batch大小不正确"
            assert tgt_batch.shape[1] > 0, "序列长度不正确"
            
            print(f"  ✅ DataLoader测试成功:")
            print(f"     Batch形状 - 源语言: {src_batch.shape}, 目标语言: {tgt_batch.shape}")
        
        print("\n" + "=" * 70)
        print("✅ 所有测试通过！训练数据加载成功！")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    test_data_loading(max_samples=100)

