# Lab4: Graph Neural Networks

> 图神经网络：节点分类和链路预测

## 1. Overview

### 1.1 Assignment

编写图卷积神经网络模型 (GCN)，并在相应的图结构数据集上完成节点分类和链路预测任务，分析`self-loop`、`layers number`、`drop edge` 、`PairNorm` 、`activation`等因素对模型的分类和预测性能的影响 。

----

### 1.2 Frameworks

实验测试对比的网络架构：

**Network:** `GCN`

**Task**: 

- Node Classification
- Link Prediction

**Datasets:**

- Cora
- Citeseer
- PPI

**Hyper parameters**

- `Self-loop`：在GNN中添加自循环的信息传递
- `Layers Number`：GNN的层数
- `Drop edge`：训练时随机丢弃一些边
- `PairNorm` ：对每层GNN的节点特征进行归一化处理
- `activation`：`relu`、`gelu`

## 2. Experiment

### 2.0 Environment

**OS:** Linux, Windows.

**Packages:** 

- pytorch
- torch_geometric
- transformers
- tensorboard
- dataclasses 
- lightning

### 2.1 Dataset

##### Cora 

[Core](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) 数据集是一个广泛使用的引文网络数据集，包含2708篇科学论文，这些论文被分为7个类别  `multiclass` 。每篇论文由一个稀疏的词袋表示，表示为一个特征向量。图中的节点代表论文，边代表引用关系。

数据特点：

- Graph：1
- Average Nodes: 2708
- Average node degree: 3.90
- Node Feature: 1433
- Class：7

##### Citeseer

[Citeseer](https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz) 数据集是一个引文网络数据集，包含3312篇科学论文，分为6个类别   `multiclass`  。每篇论文也由一个词袋特征向量表示，节点代表论文，边代表引用关系。

数据特点：

- Graph：1
- Average Nodes: 3327
- Average node degree: 2.74
- Node Feature: 3703
- Class：7

##### PPI

[PPI](http://snap.stanford.edu/graphsage/ppi.zip)（Protein-Protein Interaction）数据集来自于生物信息学领域，包含多个人类蛋白质相互作用网络。每个子图对应一个不同的细胞类型，节点代表蛋白质，边代表相互作用。节点的特征为蛋白质的生物学特征，任务为多标签节点分类   `multilabel`  。

数据特点：

- Graph: 24
- Average Nodes: 2245.30
- Average node degree: 27.31
- Node Feature: 50
- Label: 121

---

在 `torch_geometric.datasets` 已经对相应的数据集进行封装，本实验将它们封装在 `GraphDataset` 类别中，方便处理不同任务的调用。

```python
from torch_geometric.datasets import Planetoid, PPI
class GraphDataset:
    def __init__(self, dataset_name: str, root: str = 'data', task='node-cls'):
        ...

	def _load_planetoid(self):
		# dataset_name is Cora or CiteSeer
		self.dataset = Planetoid(root=self.root, name=self.dataset_name, transform=self.transform)
		...

	def _load_ppi(self):
        # dataset_name is PPI
		self.dataset = PPI(root=self.root, transform=self.transform)
        ...
```

### 2.1 图卷积神经网络 (GCN)

#### 2.1.1 GNN

图神经网络（Graph Neural Network, GNN）是一类专门用于处理图结构数据的神经网络模型。与传统的卷积神经网络（CNN）在网格状的数据（如图像、视频）上进行卷积运算不同，GNN需要面对图结构数据中不规则的拓扑关系——即节点之间的连接关系可能稀疏且无固定顺序。

为了解决这一问题，GNN主要基于“消息传递（**Message Passing**）”这一核心思想：节点从其邻居节点接收信息并进行聚合，从而不断更新自身表示（Embedding/ Feature）。

其迭代框架方式为 (**Massge Passing GNN**) ：
$$
\mathbf x_i^{(k)} = \gamma^{(k)}\left(\mathbf x_i^{k-1},\bigoplus_{j\in\mathcal N(i)}\phi^{(k)}(\mathbf x_i^{(k-1)},\mathbf x_j^{(k-1)},\mathbf e_{ij})\right)
$$
其中：

- $\mathbf x_i^{k}\in \mathbb R^F$ : 表示第 $k$ 层 layer 中节点 $\text{node}_i$ 输出的特征，
- $\mathcal N(i)$ : 表示和 $\text{node}_i$ 直接相连的节点，
- $\mathbf e_{ij}\in \mathbb R^D$ : 表示边的特征（optional）；
- $\bigoplus$ : 表示一种"求和"关系，可以是 `sum`, `mean`, `max`, `attention` 等；
- $\gamma,\phi$ : 表示一种可微函数，一般为线性变换 `MLP`。

#### 2.1.2 GCN

**GCN（Graph Convolutional Network）**是GNN的经典代表，其设计旨在有效地捕捉节点及其邻居的特征信息。

GCN 通过图卷积操作将节点的特征与其邻居的特征进行聚合，从而更新节点的表示。GCN的基本思想源于卷积神经网络 (CNN) 在图像上的应用，通过局部感受野和权重共享机制来实现高效的特征提取。

在 Message Passing GNN 的框架下，我们定义 GCN 的迭代方式如下：
$$
\mathbf x_i^{(k)}=\sum_{j\in\mathcal N(i)\and \{i\}} \frac{1}{\sqrt{\text{deg}(i)}\sqrt{\text{deg}(j)}}\cdot\left(\mathbf W^T\cdot\mathbf x_j^{(k-1)}+\mathbf b\right)
$$
其中：

- $\mathbf W^T\cdot\mathbf x_j^{(k-1)}+\mathbf b$ : 表示节点经过一个 `MLP` 的线性变换结果;
- $\frac{1}{\sqrt{\text{deg}(i)}\sqrt{\text{deg}(j)}}$ : 表示了节点之间的权重关系，为度 (degree) 较少的节点给予较多的权重 ;
- $\mathcal N(i)\and \{i\}$ : 通过自连接(**Self-loop**)，节点在更新时不仅依赖邻居节点的特征，还能保留自身的原始特征信息，避免在特征传递过程中信息的丢失。

##### Advantages:

- **简单高效：** GCN通过简单的线性变换和特征聚合，能够高效地处理大规模图数据。 

- **半监督学习能力：** GCN能够在部分节点有标签的情况下，通过邻居信息传播有效地进行节点分类。

##### Drawbacks:

- **Over-smoothing：** 随着图卷积层数的增加，节点的表示趋于一致，导致区分不同节点的能力下降。 
- **局部信息限制：** 虽然多层GCN可以扩大感受野，但仍然难以捕捉全局图结构信息。

#### 2.1.3 
