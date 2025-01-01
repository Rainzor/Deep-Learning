import torch
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures
from typing import Optional, Any, Dict, Tuple
import os

class GraphDataset:
    def __init__(self, dataset_name: str, root: str = 'data/'):
        """
        Initialize the GraphDataset with the specified dataset.

        Args:
            dataset_name (str): Name of the dataset to load. Supported names:
                                'Cora', 'CiteSeer', 'PubMed', 'PPI', etc.
            root (str): Root directory where the dataset should be saved/downloaded.
        """
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_features = None
        self.num_classes = None
        self.load_dataset()

    def load_dataset(self):
        """
        Load the dataset based on the dataset name provided during initialization.
        """
        if self.dataset_name in ['cora', 'citeseer']:
            self._load_planetoid()
        elif self.dataset_name == 'ppi':
            self._load_ppi()
        else:
            assert False, f"Dataset {self.dataset_name} not supported."

    def _load_planetoid(self):
        """
        Load Planetoid datasets: Cora, CiteSeer, PubMed.
        These datasets come with predefined train/val/test masks.
        """
        dataset_name_cap = self.dataset_name.capitalize()

        self.dataset = Planetoid(root=self.root, name=dataset_name_cap, transform=NormalizeFeatures())

        # Set attributes
        sample_data = self.dataset[0]
        self.num_features = self.dataset.num_node_features
        self.num_classes = self.dataset.num_classes

        print(f"Loaded {dataset_name_cap} dataset:")
        print(f" - Number of training nodes: {sample_data.train_mask.sum()}")
        print(f" - Number of validation nodes: {sample_data.val_mask.sum()}")
        print(f" - Number of test nodes: {sample_data.test_mask.sum()}")
        
        print(f" - Number of nodes: {sample_data.num_nodes}")
        print(f" - Number of edges: {sample_data.num_edges}")
        print(f" - Number of features: {self.num_features}")
        print(f" - Number of classes: {self.num_classes}")

    def _load_ppi(self):
        """
        Load the PPI dataset, which consists of multiple graphs for training, validation, and testing.
        """
        dir_path = os.path.join(self.root, 'PPI')
        self.train_dataset = PPI(root=dir_path, split='train', transform=NormalizeFeatures())
        self.val_dataset = PPI(root=dir_path, split='val', transform=NormalizeFeatures())
        self.test_dataset = PPI(root=dir_path, split='test', transform=NormalizeFeatures())

        self.num_features = self.train_dataset.num_features
        self.num_classes = self.train_dataset.num_classes

        data = self.train_dataset[0]

        print(f'Number of train graphs: {len(self.train_dataset)}')
        print(f'Number of val graphs: {len(self.val_dataset)}')
        print(f'Number of test graphs: {len(self.test_dataset)}')

        print(f'Number of features: {self.train_dataset.num_features}')
        print(f'Number of classes: {self.train_dataset.num_classes}')
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        # print(f'Number of training nodes: {data.train_mask.sum()}')
        # print(f'Number of validation nodes: {data.val_mask.sum()}')
        # print(f'Number of test nodes: {data.test_mask.sum()}')
        # print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
        print(f'Contains self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')

    def get_datasets(self) -> Dict[str, Optional[torch.utils.data.Dataset]]:
        """
        Retrieve the datasets based on the dataset type.

        Returns:
            dict: A dictionary containing train, val, test, and pred datasets as applicable.
        """
        datasets = {}
        if self.dataset_name in ['cora', 'citeseer', 'pubmed']:
            datasets['train'] = self.dataset
            datasets['val'] = self.dataset
            datasets['test'] = self.dataset
        else:  # self.dataset_name == 'ppi':
            datasets['train'] = self.train_dataset
            datasets['val'] = self.val_dataset
            datasets['test'] = self.test_dataset
        return datasets