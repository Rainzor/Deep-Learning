import torch
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
from typing import Optional, Any, Dict, Tuple
import os

class GraphDataset:
    def __init__(self, dataset_name: str, root: str = 'data/', task='node-cls'):
        """
        Initialize the GraphDataset with the specified dataset.

        Args:
            dataset_name (str): Name of the dataset to load. Supported names:
                                'Cora', 'CiteSeer', 'PubMed', 'PPI', etc.
            root (str): Root directory where the dataset should be saved/downloaded.
        """
        assert task in ['node-cls', 'link-pred'], f"Task {task} not supported."
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_features = None
        self.num_classes = None
        self.task = task
        if task == 'node-cls':
            self.transform = T.NormalizeFeatures()
        elif task == 'link-pred':
            self.transform = T.Compose([
                                T.NormalizeFeatures(),
                                T.RandomLinkSplit(
                                    num_val=0.05, 
                                    num_test=0.1, is_undirected=True, add_negative_train_samples=False)]
                                )
        print(f"Loading dataset {dataset_name}...")
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

        print(f"Loaded {dataset_name_cap} dataset:")
        self.dataset = Planetoid(root=self.root, name=dataset_name_cap, transform=self.transform)
        self.num_features = self.dataset.num_node_features

        # Set attributes
        sample_data = self.dataset[0]
        if self.task == 'link-pred':
            self.train_dataset, self.val_dataset, self.test_dataset = sample_data
            sample_data = self.train_dataset
            # only one graph in each dataset
            self.train_dataset = [self.train_dataset]
            self.val_dataset = [self.val_dataset]
            self.test_dataset = [self.test_dataset]
            self.num_classes = 2 # binary classification
        else:
            self.train_dataset = self.dataset
            self.val_dataset = self.dataset
            self.test_dataset = self.dataset

            self.num_classes = self.dataset.num_classes

        print(f" - Number of features: {self.num_features}")
        print(f" - Number of classes: {self.num_classes}")

        print(f" - Number of training nodes: {sample_data.train_mask.sum()}")
        print(f" - Number of validation nodes: {sample_data.val_mask.sum()}")
        print(f" - Number of test nodes: {sample_data.test_mask.sum()}")
        
        print(f" - Number of nodes: {sample_data.num_nodes}")
        print(f" - Number of edges: {sample_data.num_edges}")

    def _load_ppi(self):
        """
        Load the PPI dataset, which consists of multiple graphs for training, validation, and testing.
        """
        dir_path = os.path.join(self.root, 'PPI')
        print(f"Loaded PPI dataset:")
        self.train_dataset = PPI(root=dir_path, split='train', transform=self.transform)
        self.val_dataset = PPI(root=dir_path, split='val', transform=self.transform)
        self.test_dataset = PPI(root=dir_path, split='test', transform=self.transform)
        self.num_features = self.train_dataset.num_features

        if self.task == 'link-pred':
            train_dataset = []
            val_dataset = []
            test_dataset = []

            for ta, va, te in self.train_dataset:
                train_dataset.append(ta)
                val_dataset.append(va)
                test_dataset.append(te)
            
            for ta, va, te in self.val_dataset:
                train_dataset.append(ta)
                val_dataset.append(va)
                test_dataset.append(te)
            
            for ta, va, te in self.test_dataset:
                train_dataset.append(ta)
                val_dataset.append(va)
                test_dataset.append(te)
            
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
            self.num_classes = 2

        else:
            self.num_classes = self.train_dataset.num_classes
        
        print(f'Number of features: {self.num_features}')
        print(f'Number of classes: {self.num_classes}')

        data = self.train_dataset[0]

        print(f'Number of train graphs: {len(self.train_dataset)}')
        print(f'Number of val graphs: {len(self.val_dataset)}')
        print(f'Number of test graphs: {len(self.test_dataset)}')
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
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
        if self.train_dataset is None:
            datasets['train'] = self.dataset
            datasets['val'] = self.dataset
            datasets['test'] = self.dataset
        else:  # self.dataset_name == 'ppi':
            datasets['train'] = self.train_dataset
            datasets['val'] = self.val_dataset
            datasets['test'] = self.test_dataset
        return datasets