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

#### 2.1.2 Word Embedding

本实验对比了重头训练 **Word Embedding** ，没有使用预先训练好的 **Word Embedding** 结果。

```python
class YelpData:
    text: str
    star: int

class YelpDataset(Dataset):
    def __init__(self, data_dir, tokenizer, train=True, max_length=512):
        """
        Dataset constructor
        :param data_dir: Directory of the data files
        :param train: Whether to load training data
        :param tokenizer_name: Name of the tokenizer to use
        :param max_length: Maximum length for padding and truncation
        """
        self.is_train = train
        self.data_path = os.path.join(data_dir, 'train.json') if train else os.path.join(data_dir, 'test.json')
        self.raw_data = self._read_json(self.data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
 
    def _read_json(self, file_path):
        """
        Load training/test data from the specified directory
        :param data_dir: Directory containing the data files
        :param train: Whether to load the training data
        :return: List of data instances
        """
		...
		
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx)
        text = self.raw_data[idx].text
        label = self.raw_data[idx].star-1
        encoding = self.tokenizer(text,
                            add_special_tokens=True,
                            truncation=True,
                            padding='max_length',
                            max_length=self.max_length,
                            return_attention_mask=True,
                            return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)}
```

### 2.2 Models

#### 2.2.1 RNN

传统的RNN结构为了处理序列的文本信息，引入了隐状态 $H_t$ 的概念，用于存储前 $t$ 的时间步所学到的信息，更新方式如下：
$$
H_t=\sigma(X_tW_{xh}+H_{t-1}W_{hh}+b_h)
$$
这些变量捕获并保留了序列直到其当前时间步的历史信息， 就如当前时间步下神经网络的状态或记忆， 因此这样的隐藏变量被称为*隐状态*（hidden state）

<img src="assets/image-20241225161726818.png" alt="image-20241225161726818" style="zoom: 33%;" />

#### 2.2.2 GRU

门控循环单元(GRU)与普通的循环神经网络之间的关键区别在于：GRU支持隐状态的门控。这意味着模型有专门的门控机制：*重置门*（reset gate）和*更新门*（update gate），来确定应该何时更新隐状态， 以及应该何时重置隐状态。 这能够帮助网络更好的捕捉较为重要的字词。更新方式如下：
$$
\hat{H}_t =\sigma(X_tW_{xh}+(R_t\odot H_{t-1})W_{hh}+b_h)\\
H_t=Z_t \odot H_{t-1}+(1-Z_t)\odot \hat{H}_t
$$
<img src="assets/image-20241225162540053.png" alt="image-20241225162540053" style="zoom:33%;" />