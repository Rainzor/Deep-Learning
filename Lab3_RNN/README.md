# Lab3: Recurrent Neural Networks

> SA2422016 王润泽
>
> 循环神经网络：文本分类

## 1. Overview

开发一个RNN语言模型，并使用训练好的词向量实现RNN模型用于文本分类。

实验测试和对比的网络架构：

- `RNN`, `Bi-RNN`
- `GRU`, `Bi-RNN`
- `LSTM`,`Bi-RNN`: Last, Mean, Max, Attention
- `Transformer`

## 2. Experiment

### 2.0 Environment

本实验在 Linux/Windows 操作系统下进行，主要包含的库有：

- pytorch
- numpy
- tqdm
- tensorboard
- transformer

### 2.1 Dataset

使用 [Yelp2013]([Yelp-Data-Challenge-2013/yelp_challenge/yelp_phoenix_academic_dataset at master · rekiksab/Yelp-Data-Challenge-2013 · GitHub](https://github.com/rekiksab/Yelp-Data-Challenge-2013/tree/master/yelp_challenge/yelp_phoenix_academic_dataset)) 数据集，将 `test.json` 用作测试集，并从 ` `中手动划分训练集和验证集。仅需使用stars评分和text评论内容。本实验采用 `val_ratio=0.05` 比例划分数据。

**Note**：训练集的前1000个为 `test.json` 的内容，需要手动剔除。

#### 2.1.1 Tokenizer

为了方便建立字典表，采用了 `transformers.AutoTokenizer` 来将文本进行向量化处理，使用的是 `bert-base-uncased` 的向量化方法。

本实验从头训练 **Word Embedding**，没有使用训练好的 Word Embedding 结果。经过实验测试，