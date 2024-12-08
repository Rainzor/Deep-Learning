# Lab2: Convolution Neural Network

> 卷积神经网络：图像分类
>
> SA24229016 王润泽、

## 1. Overview

使用 `pytorch` 实现卷积神经网络，在 ImageNet 数据集上进行图片分 类。研究 dropout, normalization, learning rate decay, residual connection, network depth等超参数对分类性能的影响。

实验测试和对比的网络架构：

- `VGG`: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- `ResNet`: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- `ResNeXt:` [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
- `T2T-ViT`: [Tokens-to-Token ViT: Training Vision Transformers From Scratch on ImageNet](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.html)

## 2. Experiment

### 2.0 Environment

本实验在 Linux/Windows 操作系统下进行，主要包含的库有：

- pytorch
- opencv
- numpy
- tqdm
- tensorboard

更多库见 `requirement.txt`

代码文件包含：

- `train.py` 和  `train_ddp.py`：分别用于单GPU训练和多GPU训练
- `dataset.py`：管理和加载数据
- `model/*` ：包含实验相关的网络架构 `VGG`、`ResNet` 、`ResNeXt`和 `T2T_ViT`
- `utils.py`：其他的函数

## 2.1 Dataset

实验使用 [`Tiny-Imagenet-200` ](http://cs231n.stanford.edu/tiny-imagenet-200.zip) 数据集，包含 200 个类，每个类有 500 张训练图像，50 张验证图像和 50 张测试图像。由于测试图像没有标签，因此使用数据集中的 `val` 当作测试集，并从 `train` 中手动划分新的训练集和验证集。本实验采用 `val_ratio=0.2` 比例划分数据。

### 2.1.1 Load and preprocess

在 `dataset.txt` 中按照如下方式创建数据集，值得说明的是为了避免多次处理和加载数据，采用了 **文件Cache** 的方式保存图像和标签数据，保存在 `data_path/process` 目录下。

```python
class TinyImageNetDataset(Dataset):
    def __init__(self, type_, raw_data, transform=None, force_reload=False):
        """
        type_: 'train' or 'val'
        raw_data: RawData instance
        transform: torchvision transforms to apply
        force_reload: If True, ignore cached data and reprocess
        """
        self.type = type_
        self.raw_data = raw_data
        self.force_reload = force_reload

        # Create a directory to save processed data
        self.processed_path = os.path.join(self.raw_data.data_path, "process")
        os.makedirs(self.processed_path, exist_ok=True)

        # Load or preprocess data
        if self.type == "train":
            self.images, self.labels = self._load_or_preprocess_train_data()
        else:
            self.images, self.labels = self._load_or_preprocess_val_data()

        self.transform = transforms.ToTensor() if transform is None else transform
        ......
    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.labels)
```

### 2.1.2 Transform

为了高效利用图像数据，对训练数据进行相关变换操作，用于数据增强。