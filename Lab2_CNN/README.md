# Lab2: Convolution Neural Network

> 卷积神经网络：图像分类
>
> SA24229016 王润泽

使用 `pytorch` 实现卷积神经网络，在 ImageNet 数据集上进行图片分 类。研究 dropout, normalization, learning rate decay, residual connection, network depth等超参数对分类性能的影响。

实验测试网络架构：

- `VGG`: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

- `ResNet`: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- `DenseNet`
- `ResNeXt`
- `T2T-ViT`: [Tokens-to-Token ViT: Training Vision Transformers From Scratch on ImageNet](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.html)