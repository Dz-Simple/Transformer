"""
简单的分词器实现
用于机器翻译任务的文本分词
"""

import re


class SimpleTokenizer:
    """简单的空格分词器"""
    
    def __init__(self, lowercase=True):
        """
        Args:
            lowercase: 是否转换为小写
        """
        self.lowercase = lowercase
    
    def tokenize(self, text):
        """
        分词
        Args:
            text: 输入文本
        Returns:
            tokens列表
        """
        if self.lowercase:
            text = text.lower()
        
        # 简单的空格分词，保留标点符号
        # 在标点符号前后添加空格
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)  # 移除多余空格
        
        tokens = text.strip().split()
        return tokens
    
    def detokenize(self, tokens):
        """
        反分词（支持BPE格式）
        Args:
            tokens: token列表
        Returns:
            文本字符串
        """
        text = ' '.join(tokens)
        
        # 处理 BPE 标记：移除 @@ 并合并子词
        # BPE格式: "ja@@ gu@@ ar" -> "jaguar"
        text = text.replace('@@ ', '')  # 移除 @@ 和后面的空格，合并子词
        
        # 移除标点符号前的空格
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text


def test_tokenizer():
    """测试分词器"""
    print("\n" + "="*50)
    print("测试简单分词器")
    print("="*50)
    
    tokenizer = SimpleTokenizer(lowercase=True)
    
    texts = [
        "Hello, world!",
        "This is a test sentence.",
        "Machine translation is amazing!",
    ]
    
    for text in texts:
        tokens = tokenizer.tokenize(text)
        detokenized = tokenizer.detokenize(tokens)
        
        print(f"\n原始文本: {text}")
        print(f"分词结果: {tokens}")
        print(f"反分词: {detokenized}")
    
    print("\n分词器测试完成！\n")


if __name__ == "__main__":
    test_tokenizer()

