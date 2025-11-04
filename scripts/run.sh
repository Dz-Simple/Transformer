#!/bin/bash

# Transformer 模型测试脚本

echo "========================================="
echo "  Transformer 模型完整测试"
echo "========================================="
echo ""

# 激活 conda 环境（如果需要）
# conda activate torch

# 切换到项目根目录
cd "$(dirname "$0")/.."

echo "📦 测试基础组件..."
echo ""

# 测试各个基础组件
echo "  [1/5] 位置编码..."
python -m src.components.positional_encoding

echo ""
echo "  [2/5] 自注意力机制..."
python -m src.components.self_attention

echo ""
echo "  [3/5] 多头注意力..."
python -m src.components.multi_head_attention

echo ""
echo "  [4/5] 前馈神经网络..."
python -m src.components.feed_forward

echo ""
echo "  [5/5] 残差连接和层归一化..."
python -m src.components.residual_layernorm

echo ""
echo "========================================="
echo "  测试完整模型"
echo "========================================="
echo ""

echo "🏗️  测试 Encoder..."
python -m src.encoder

echo ""
echo "🏗️  测试 Decoder..."
python -m src.decoder

echo ""
echo "🎯 测试完整 Transformer 模型..."
python -m src.transformer

echo ""
echo "========================================="
echo "  训练模型"
echo "========================================="
echo ""

echo "🚀 开始训练 Transformer 模型..."
echo "  (这可能需要几分钟时间)"
python train.py

echo ""
echo "========================================="
echo "✅ 所有测试和训练完成！"
echo "========================================="

