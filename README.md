# Transformer 机器翻译项目

本项目从零开始实现了完整的 Transformer 架构，用于德英机器翻译任务，使用 IWSLT14 数据集进行训练和评估。

## 📁 项目结构

```
Mid-term assignment/
├── src/                              
│   ├── components/                     # Transformer 核心组件
│   │   ├── positional_encoding.py      # 位置编码
│   │   ├── self_attention.py           # 自注意力机制
│   │   ├── multi_head_attention.py     # 多头注意力
│   │   ├── feed_forward.py             # 前馈神经网络
│   │   └── residual_layernorm.py       # 残差连接和层归一化（Pre-LN）
│   ├── data/                           # 数据处理模块
│   │   ├── tokenizer.py                # 分词器（BPE支持）
│   │   ├── vocabulary.py               # 词汇表
│   │   └── dataset.py               # 数据集（IWSLT14）
│   ├── ablation_studies/            # 消融实验
│   │   ├── ablation_config.py       # 统一配置
│   │   ├── ablation_1_num_heads.py  # 注意力头数消融
│   │   ├── ablation_2_num_layers.py # 模型层数消融
│   │   ├── ablation_3_positional_encoding.py  # 位置编码消融
│   │   ├── ablation_4_layer_norm.py # LayerNorm策略消融
│   │   └── test_imports.py          # 导入测试
│   ├── encoder.py                   # Transformer Encoder
│   ├── decoder.py                   # Transformer Decoder
│   ├── transformer.py               # 完整 Transformer 模型
│   └── __init__.py
├── scripts/                         # 脚本
│   ├── run.sh                       # 完整测试和训练脚本
│   └── test_data.sh                 # 数据模块测试（可选）
├── datasets/                       # 数据集目录
│   └── iwslt14/                    # IWSLT14 德英翻译数据集
│       ├── train.de, train.en      # 训练集
│       ├── valid.de, valid.en      # 验证集
│       ├── test.de, test.en        # 测试集
│       └── vocab.de, vocab.en       # 词汇表文件
├── output/                          # 可视化输出（自动生成）
├── results/                         # 训练结果（自动生成）
│   ├── models_YYYYMMDD_HHMMSS/     # 模型检查点
│   │   ├── best_model.pt            # 最佳模型
│   │   └── test_results/           # 测试集评估结果
│   ├── plots_YYYYMMDD_HHMMSS/      # 训练曲线和日志
│   ├── ablation_1_num_heads/       # 消融实验1结果
│   ├── ablation_2_num_layers/      # 消融实验2结果
│   ├── ablation_3_positional_encoding/  # 消融实验3结果
│   └── ablation_4_layer_norm/      # 消融实验4结果
├── train.py                         # 训练脚本
├── test.py                          # 测试脚本（评估模型）
├── requirements.txt                 # Python 依赖
└── report.tex                       # 实验报告（LaTeX）
```

## 🚀 快速开始

### 1. 环境配置

#### 使用 Conda（推荐）

```bash
# 创建conda环境
conda create -n ctorch python=3.10 -y

# 激活环境
conda activate ctorch

# 安装PyTorch（GPU版本，根据CUDA版本选择）
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

#### 使用 pip

```bash
pip install -r requirements.txt
```

### 2. 数据集准备

本项目使用 IWSLT14 德英翻译数据集。数据集应放置在 `datasets/iwslt14/` 目录下，包含以下文件：

- `train.de`, `train.en` - 训练集（约160,239对）
- `valid.de`, `valid.en` - 验证集（约7,283对）
- `test.de`, `test.en` - 测试集（约6,750对）
- `vocab.de`, `vocab.en` - 词汇表文件（BPE分词）

### 3. 运行测试

#### 测试所有组件和完整流程

```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

该脚本会依次：
1. 测试所有 Transformer 核心组件
2. 测试 Encoder 和 Decoder
3. 测试完整 Transformer 模型
4. 训练模型
5. 在测试集上评估模型

#### 单独测试组件

测试数据处理模块:
```bash
chmod +x scripts/test_data.sh
./scripts/test_data.sh
```

测试消融实验导入:
```bash
python -m src.ablation_studies.test_imports
```

### 4. 训练模型

```bash
python train.py
```

**训练配置**：
- 模型：6层Encoder/Decoder，8头注意力，$d_{model}=512$
- 数据集：IWSLT14（完整训练集）
- 优化器：Adam ($\beta_1=0.9, \beta_2=0.98$)
- 学习率调度：Noam Scheduler (warmup=4000)
- 早停机制：patience=10，min_delta=0.001
- 标签平滑：0.05
- 批大小：64
- 最大epochs：30

**训练结果**：
- 模型检查点：`results/models_YYYYMMDD_HHMMSS/best_model.pt`
- 训练曲线：`results/plots_YYYYMMDD_HHMMSS/training_curves.png`
- 训练日志：`results/plots_YYYYMMDD_HHMMSS/training_log.md`

### 5. 测试模型

在训练好的模型上进行测试集评估：

```bash
python test.py --model_path results/models_YYYYMMDD_HHMMSS/best_model.pt
```

如果不指定模型路径，默认使用最新的模型：
```bash
python test.py
```

**评估指标**：
- BLEU 分数（n-gram重叠度）
- METEOR 分数（词序和同义词匹配）
- 困惑度（Perplexity）

**测试结果**：
- 评估结果：`results/models_YYYYMMDD_HHMMSS/test_results/test_evaluation_summary.md`
- 可视化图表：`results/models_YYYYMMDD_HHMMSS/test_results/test_evaluation_results.png`

### 6. 运行消融实验

消融实验使用较小的模型（$d_{model}=256$，3层）和50,000条训练数据，每个实验训练10个epoch：

```bash
# 实验1：注意力头数影响（1, 2, 4, 8头）
python -m src.ablation_studies.ablation_1_num_heads --gpu 0

# 实验2：模型层数影响（1, 2, 3, 4, 6层）
python -m src.ablation_studies.ablation_2_num_layers --gpu 1

# 实验3：位置编码策略（正弦/可学习/无）
python -m src.ablation_studies.ablation_3_positional_encoding --gpu 2

# 实验4：LayerNorm策略（Post-LN/Pre-LN/无）
python -m src.ablation_studies.ablation_4_layer_norm --gpu 2
```

**注意**：可以使用 `--gpu` 参数指定GPU设备，支持多GPU并行运行多个实验。

**消融实验结果**：
- JSON数据：`results/ablation_*/results.json`
- 对比图表：`results/ablation_*/*.png`

## 📊 实验结果

### 主实验结果

- **最佳验证困惑度**：4.72（第10轮）
- **测试集BLEU**：26.31%
- **测试集METEOR**：56.12%
- **训练时间**：约10分钟/epoch（Tesla V100）

### 消融实验结果

1. **注意力头数**：单头表现最佳（验证PPL 20.40），可能与模型大小相关
2. **模型层数**：6层最佳（验证PPL 21.04），深度带来性能提升
3. **位置编码**：三种策略性能相近（验证PPL 21.74-21.76）
4. **LayerNorm**：三种策略在浅层网络上性能相同（验证PPL 21.74）

详细结果请参考 `report.tex` 实验报告。

## 🔧 项目特性

- ✅ **完整实现**：从零实现 Transformer 所有核心组件
- ✅ **Pre-LN架构**：使用Pre-LayerNorm，训练更稳定
- ✅ **Noam学习率调度**：实现论文中的学习率预热策略
- ✅ **早停机制**：自动防止过拟合
- ✅ **标签平滑**：提升模型泛化能力
- ✅ **BPE分词支持**：处理子词级别的翻译
- ✅ **多GPU支持**：消融实验支持指定GPU设备
- ✅ **完整评估**：BLEU、METEOR、困惑度等指标

## 📝 依赖说明

主要依赖包：
- `torch>=2.0.0` - PyTorch深度学习框架
- `numpy>=1.24.0` - 数值计算
- `nltk>=3.8.1` - 自然语言处理工具（BLEU/METEOR）
- `matplotlib>=3.7.0` - 可视化
- `tqdm>=4.65.0` - 进度条
- `pandas>=2.0.0` - 数据处理

详细依赖请查看 `requirements.txt`。

## 📚 参考文档

- 实验报告：`report.tex`（LaTeX格式）
- 训练日志：`results/plots_*/training_log.md`
- 测试结果：`results/models_*/test_results/test_evaluation_summary.md`

## ⚠️ 注意事项

1. **GPU要求**：训练需要GPU支持（推荐CUDA 11.4+），CPU训练会非常慢
2. **内存要求**：主训练需要约17GB GPU显存，消融实验需要约8-10GB
3. **数据集路径**：确保 `datasets/iwslt14/` 目录包含所有必需文件
4. **NLTK数据**：首次运行测试脚本时，NLTK会自动下载必要数据（punkt、wordnet等）

## 🐛 问题排查

- **CUDA内存不足**：减小 `batch_size` 或使用梯度累积
- **NLTK数据下载失败**：手动下载NLTK数据到本地
- **中文显示问题**：确保系统安装了中文字体（Noto Sans CJK等）

---

**作者**：Mid-term Assignment  
**日期**：2025年11月
