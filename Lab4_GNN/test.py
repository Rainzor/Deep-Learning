import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

import torch_geometric
from torch_geometric.loader import DataLoader

from models import GNNEnocder
from utils.utils import *
from utils.data import GraphDataset

import pytorch_lightning as pl

import torchmetrics


class NodeClassifier(pl.LightningModule):
    def __init__(self, model_config, trainer_config):
        super(NodeClassifier, self).__init__()

        self.save_hyperparameters()

        self.encoder = GNNEnocder(
                            in_channels=model_config.num_features, 
                            hidden_channels=model_config.hidden_channels, 
                            out_channels=model_config.num_classes,
                            gnn_type=model_config.name,
                            num_layers=model_config.num_layers,
                            dropout=model_config.dropout,
                            edge_dropout=model_config.edge_dropout,
                            pairnorm_mode=model_config.pairnorm_mode,
                            self_loop=model_config.self_loop,
                            activation=model_config.activation)
        # self.decoder = nn.Linear(model_config.hidden_channels, model_config.num_classes)
        self.config = trainer_config

        if self.config.dataset in ['cora', 'citeseer']:
            self.criterion = nn.CrossEntropyLoss()
        
            self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=model_config.num_classes)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

            self.test_acc = torchmetrics.Accuracy(task='multilabel', num_labels=model_config.num_classes)

    
    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        # x = self.decoder(F.relu(x))
        return x
    
    def training_step(self, batch, batch_idx):
        pass

    
    def validation_step(self, batch, batch_idx):
        pass

    
    def test_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        out = self(x, edge_index)
        if self.config.dataset in ['cora', 'citeseer']:

            preds = out[batch.test_mask]
            targets = y[batch.test_mask]
        else:
            preds = out
            targets = y

        self.test_acc(preds, targets)
        batch_size = preds.size(0)

        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, batch_size=batch_size)
        
    def on_test_epoch_end(self):
        self.test_acc.reset()
    
    def configure_optimizers(self):

        # params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        params = self.encoder.parameters()
        optimizer = torch.optim.Adam(params, 0.01)

        return optimizer

class LinkPredictor(pl.LightningModule):
    def __init__(self, model_config, trainer_config):
        super(LinkPredictor, self).__init__()

        self.save_hyperparameters()

        self.encoder = GNNEnocder(
                            in_channels=model_config.num_features, 
                            hidden_channels=model_config.hidden_channels, 
                            out_channels=model_config.hidden_channels//2,
                            gnn_type=model_config.name,
                            num_layers=model_config.num_layers,
                            dropout=model_config.dropout,
                            edge_dropout=model_config.edge_dropout,
                            pairnorm_mode=model_config.pairnorm_mode,
                            self_loop=model_config.self_loop,
                            activation=model_config.activation)

        self.config = trainer_config

        self.criterion = nn.BCEWithLogitsLoss()

        self.test_acc = torchmetrics.Accuracy(task='binary')
    
    def decoder(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z, edge_label_index)
        return out
    
    def training_step(self, batch, batch_idx):
        pass

    
    def validation_step(self, batch, batch_idx):
        pass

    
    def test_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index
        edge_label_index = batch.edge_label_index
        logits = self(x, edge_index, edge_label_index)
        preds = torch.sigmoid(logits)

        self.test_acc(preds, batch.edge_label)
        batch_size = preds.size(0)

        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,  batch_size=batch_size, logger=True)
    
    def on_test_epoch_end(self):
        self.test_acc.reset()
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        return optimizer

def parse_args():
    """
    Parse the command-line arguments for the test script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='GNN Node Classification')
    parser.add_argument('--model','-m', type=str, default='models', help='Model directory')
    parser.add_argument('--log','-l', type=str, default='logs', help='Log directory')

    parser.add_argument('--root','-r', type=str, default='data', help='Root directory for the dataset')
    
    return parser.parse_args()


def test():
    args = parse_args()

    task = [
        'node-cls',
        'link-pred',
    ]

    dataset = [
        'cora',
        'citeseer',
        'ppi'
    ]
    
    for t in task:
        for d in dataset:
            pl.seed_everything(42)
            graph_dataset = GraphDataset(d, args.root, t)
            test_dataset = graph_dataset.get_datasets()['test']
            test_loader = DataLoader(test_dataset, batch_size=1)
            checkpoint_path = os.path.join(args.model, t, f"{d}.ckpt")
            if t == 'node-cls':
                model = NodeClassifier.load_from_checkpoint(checkpoint_path)
            else:
                model = LinkPredictor.load_from_checkpoint(checkpoint_path)
            trainer = pl.Trainer(accelerator='gpu')
            trainer.test(model, test_loader)

if __name__ == '__main__':
    test()