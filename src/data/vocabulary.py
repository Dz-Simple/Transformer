"""
词汇表构建模块
用于机器翻译任务的词汇表管理
"""

from collections import Counter
from typing import List, Dict
import os


class Vocabulary:
    """词汇表类"""
    
    # 特殊标记
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<sos>'  # Start of Sequence
    EOS_TOKEN = '<eos>'  # End of Sequence
    
    def __init__(self, max_size=None, min_freq=1):
        """
        Args:
            max_size: 词汇表最大大小
            min_freq: 最小词频
        """
        self.max_size = max_size
        self.min_freq = min_freq
        
        # 词到索引的映射
        self.word2idx: Dict[str, int] = {}
        # 索引到词的映射
        self.idx2word: Dict[int, str] = {}
        
        # 添加特殊标记
        self._add_special_tokens()
        
        # 词频统计
        self.word_freq = Counter()
    
    def _add_special_tokens(self):
        """添加特殊标记"""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
        ]
        for token in special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token
    
    def load_from_file(self, vocab_file: str):
        """
        从词汇表文件加载词汇表
        Args:
            vocab_file: 词汇表文件路径，每行一个词
        """
        # 特殊标记映射（数据集使用<s>, </s>, 代码使用<sos>, <eos>）
        token_mapping = {
            '<s>': self.SOS_TOKEN,
            '</s>': self.EOS_TOKEN,
            '<pad>': self.PAD_TOKEN,
            '<unk>': self.UNK_TOKEN,
        }
        
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"词汇表文件不存在: {vocab_file}")
        
        # 清空现有词汇表，然后重新添加特殊标记
        self.word2idx = {}
        self.idx2word = {}
        self._add_special_tokens()
        
        # 从文件读取词汇
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if not word:  # 跳过空行
                    continue
                
                # 映射特殊标记
                if word in token_mapping:
                    word = token_mapping[word]
                
                # 添加词（如果不存在）
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
    
    def build_from_texts(self, texts: List[List[str]]):
        """
        从文本列表构建词汇表
        Args:
            texts: 分词后的文本列表
        """
        # 统计词频
        for tokens in texts:
            self.word_freq.update(tokens)
        
        # 按词频排序
        sorted_words = sorted(
            self.word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 添加词到词汇表
        for word, freq in sorted_words:
            if freq < self.min_freq:
                break
            
            if self.max_size and len(self.word2idx) >= self.max_size:
                break
            
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, tokens: List[str], add_special_tokens=True) -> List[int]:
        """
        将token序列转换为索引序列
        Args:
            tokens: token列表
            add_special_tokens: 是否添加特殊标记
        Returns:
            索引列表
        """
        indices = []
        
        if add_special_tokens:
            indices.append(self.word2idx[self.SOS_TOKEN])
        
        for token in tokens:
            idx = self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
            indices.append(idx)
        
        if add_special_tokens:
            indices.append(self.word2idx[self.EOS_TOKEN])
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens=True) -> List[str]:
        """
        将索引序列转换为token序列
        Args:
            indices: 索引列表
            skip_special_tokens: 是否跳过特殊标记
        Returns:
            token列表
        """
        tokens = []
        
        special_indices = {
            self.word2idx[self.PAD_TOKEN],
            self.word2idx[self.SOS_TOKEN],
            self.word2idx[self.EOS_TOKEN],
        }
        
        for idx in indices:
            if skip_special_tokens and idx in special_indices:
                continue
            
            token = self.idx2word.get(idx, self.UNK_TOKEN)
            tokens.append(token)
        
        return tokens
    
    def __len__(self):
        """返回词汇表大小"""
        return len(self.word2idx)
    
    @property
    def pad_idx(self):
        """返回padding索引"""
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def unk_idx(self):
        """返回unknown索引"""
        return self.word2idx[self.UNK_TOKEN]
    
    @property
    def sos_idx(self):
        """返回start-of-sequence索引"""
        return self.word2idx[self.SOS_TOKEN]
    
    @property
    def eos_idx(self):
        """返回end-of-sequence索引"""
        return self.word2idx[self.EOS_TOKEN]


def build_vocabulary(texts: List[List[str]], max_size=None, min_freq=1):
    """
    构建词汇表的辅助函数
    Args:
        texts: 分词后的文本列表
        max_size: 最大词汇表大小
        min_freq: 最小词频
    Returns:
        Vocabulary对象
    """
    vocab = Vocabulary(max_size=max_size, min_freq=min_freq)
    vocab.build_from_texts(texts)
    return vocab


def load_vocabulary_from_file(vocab_file: str):
    """
    从词汇表文件加载词汇表
    Args:
        vocab_file: 词汇表文件路径，每行一个词
    Returns:
        Vocabulary对象
    """
    vocab = Vocabulary()
    vocab.load_from_file(vocab_file)
    return vocab


def test_vocabulary():
    """测试词汇表"""
    print("\n" + "="*50)
    print("测试词汇表")
    print("="*50)
    
    # 示例文本
    texts = [
        ['hello', 'world'],
        ['hello', 'machine', 'translation'],
        ['world', 'is', 'beautiful'],
        ['machine', 'learning', 'is', 'amazing'],
    ]
    
    # 构建词汇表
    vocab = build_vocabulary(texts, max_size=20, min_freq=1)
    
    print(f"\n词汇表大小: {len(vocab)}")
    print(f"前10个词: {list(vocab.word2idx.keys())[:10]}")
    
    # 测试编码和解码
    test_tokens = ['hello', 'world', 'unknown_word']
    indices = vocab.encode(test_tokens, add_special_tokens=True)
    decoded = vocab.decode(indices, skip_special_tokens=True)
    
    print(f"\n原始tokens: {test_tokens}")
    print(f"编码后索引: {indices}")
    print(f"解码后tokens: {decoded}")
    
    # 打印特殊标记索引
    print(f"\n特殊标记:")
    print(f"  PAD: {vocab.pad_idx}")
    print(f"  UNK: {vocab.unk_idx}")
    print(f"  SOS: {vocab.sos_idx}")
    print(f"  EOS: {vocab.eos_idx}")
    
    print("\n词汇表测试完成！\n")


if __name__ == "__main__":
    test_vocabulary()

