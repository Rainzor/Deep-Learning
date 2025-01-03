import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time

import torch_geometric
from torch_geometric.loader import DataLoader

from models import GNNEnocder
from utils.utils import *
from utils.data import GraphDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

import torchmetrics
from transformers import get_scheduler


class NodeClassifier(pl.LightningModule):
    def __init__(self, model_config, trainer_config):
        super(NodeClassifier, self).__init__()

        self.save_hyperparameters()

        self.encoder = GNNEnocder(
                            in_channels=model_config.num_features, 
                            hidden_channels=model_config.hidden_channels, 
                            out_channels=model_config.hidden_channels,
                            gnn_type=model_config.name,
                            num_layers=model_config.num_layers,
                            dropout=model_config.dropout,
                            residual=model_config.residual)
        self.decoder = nn.Linear(model_config.hidden_channels, model_config.num_classes)
        self.config = trainer_config

        if self.config.dataset in ['cora', 'citeseer']:
            self.criterion = nn.CrossEntropyLoss()

            self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=model_config.num_classes)

            self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=model_config.num_classes)
        else:
            self.criterion = nn.BCEWithLogitsLoss()


            self.val_acc = torchmetrics.Accuracy(task='multilabel', num_labels=model_config.num_classes)

            self.test_acc = torchmetrics.Accuracy(task='multilabel', num_labels=model_config.num_classes)

    
    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.decoder(F.relu(x))
        return x
    
    def training_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        out = self(x, edge_index)
        if self.config.dataset in ['cora', 'citeseer']:
            loss = self.criterion(out[batch.train_mask], y[batch.train_mask])
            preds = out[batch.train_mask]
            targets = y[batch.train_mask]
        else:
            loss = self.criterion(out, y)
            preds = out
            targets = y

        batch_size = preds.size(0)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        out = self(x, edge_index)
        if self.config.dataset in ['cora', 'citeseer']:
            preds = out[batch.val_mask]
            targets = y[batch.val_mask]
        else:
            preds = out
            targets = y
        self.val_acc(preds, targets)
        batch_size = preds.size(0)

        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def on_train_start(self):
        self.print("Start training...")
        self.logger.log_hyperparams(self.hparams, { "hp/test_loss": 0, "hp/test_acc": 0})
    
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

        self.log('hp/test_acc', self.test_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
    
    def configure_optimizers(self):

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        optimizer = torch.optim.Adam(params, lr=self.config.lr, weight_decay=self.config.weight_decay)

        scheduler = None
        if self.config.scheduler:
            assert self.config.num_training_steps > 0, "num_training_steps must be specified for LR scheduler"
            scheduler = get_scheduler(
                self.config.scheduler,
                optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=self.config.num_training_steps,
            )

        if scheduler:
            scheduler_config = {"scheduler": scheduler, "interval": "step"}
            return [optimizer], [scheduler_config]
        else:
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
                            residual=model_config.residual)

        self.config = trainer_config

        self.criterion = nn.BCEWithLogitsLoss()

        self.val_acc = torchmetrics.Accuracy(task='binary')

        self.test_acc = torchmetrics.Accuracy(task='binary')
    
    def decoder(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encoder(x, edge_index)
        out = self.decoder(z, edge_label_index)
        return out
    
    def training_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index
        edge_label, edge_label_index = negative_sample(batch)
        logits = self(x, edge_index, edge_label_index)
        loss = self.criterion(logits, edge_label)

        batch_size = logits.size(0)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index
        edge_label_index = batch.edge_label_index
        logits = self(x, edge_index, edge_label_index)
        preds = torch.sigmoid(logits)

        self.val_acc(preds, batch.edge_label)
        batch_size = preds.size(0)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True,prog_bar=True, batch_size=batch_size)


    def on_train_start(self):
        self.print("Start training...")
        self.logger.log_hyperparams(self.hparams, {"hp/test_acc": 0})
    
    def test_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index
        edge_label_index = batch.edge_label_index
        logits = self(x, edge_index, edge_label_index)
        preds = torch.sigmoid(logits)

        self.test_acc(preds, batch.edge_label)
        batch_size = preds.size(0)
        self.log('hp/test_acc', self.test_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        scheduler = None
        if self.config.scheduler:
            assert self.config.num_training_steps > 0, "num_training_steps must be specified for LR scheduler"
            scheduler = get_scheduler(
                self.config.scheduler,
                optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=self.config.num_training_steps,
            )

        if scheduler:
            scheduler_config = {"scheduler": scheduler, "interval": "step"}
            return [optimizer], [scheduler_config]
        else:
            return optimizer            

def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    graph_dataset = GraphDataset(dataset_name=args.dataset, root=args.root, task=args.task)

    datasets = graph_dataset.get_datasets()

    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    train_config = TrainerConfig(
        dataset=args.dataset,
        task=args.task,
        epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        scheduler=args.scheduler,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_loader) * args.epochs,
    )

    model_config = ModelConfig(
        name=args.model.lower(),
        num_features=graph_dataset.num_features,
        num_classes=graph_dataset.num_classes,
        hidden_channels=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        residual=args.residual
    )

    if args.task == 'node-cls':
        model = NodeClassifier(model_config, train_config)
    elif args.task == 'link-pred':
        model = LinkPredictor(model_config, train_config)
    else:
        raise ValueError(f"Task {args.task} not supported.")
        
    print(train_config)
    print(model_config)

    timenow = time.strftime("%Y-%m%d-%H%M")
    if args.tag:
        output_dir = os.path.join(args.output_path, args.task, args.dataset, 
        f"{model_config.name}-{args.tag}")
    else:
        output_dir = os.path.join(args.output_path, args.task, args.dataset, model_config.name)
    
    os.makedirs(output_dir, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(output_dir, 
                            name=timenow,
                            version='')
    checkpoint_dir = os.path.join(output_dir, timenow, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        monitor='val/acc',
        dirpath=checkpoint_dir,
        filename='{epoch:02d}',
        save_top_k=1,
        mode='max',
        save_weights_only=True,
        # verbose=True,
        save_last=True
    )
    patience = args.patience if args.patience>0 else args.epochs
    early_stopping = EarlyStopping(
        monitor='val/acc',
        patience=patience,
        mode='max',
        divergence_threshold=0.05,
        # verbose=True  # Print when early stopping happens
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        max_epochs=train_config.epochs,
        accelerator='gpu',
        log_every_n_steps=5,
    )

    trainer.fit(model, train_loader, val_loader)

    best_checkpoint = checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_checkpoint}")
    if args.task == 'node-cls':
        checkpoint_model = NodeClassifier.load_from_checkpoint(
                                    best_checkpoint,
                                    model_config=model_config,
                                    trainer_config=train_config
                                )
    elif args.task == 'link-pred':
        checkpoint_model = LinkPredictor.load_from_checkpoint(
                                    best_checkpoint,
                                    model_config=model_config,
                                    trainer_config=train_config
                                )
    trainer.test(checkpoint_model, test_loader)              

    if args.visualize:
        data = datasets['test'][0]
        out = checkpoint_model(data.x, data.edge_index)
        visualize(out, data.y)


if __name__ == '__main__':
    main()
