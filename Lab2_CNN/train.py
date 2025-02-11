import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler

import numpy as np
import random
import time
import os
import sys
import argparse
import json

from torch.utils.tensorboard import SummaryWriter

from models.VGG import VGG
from models.ResNet import ResNet
from models.ViT import T2T_ViT
from dataloader.dataset import TinyImageNetDataset, RawData
from config import *
from utils import *

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def DataLoaderSplit(raw_data, batch_size, val_ratio=0.2, force_reload=False,workers=1, half=False):
    """
    Prepare DataLoaders for training, validation, and testing.

    Args:
        raw_data (RawData): Instance of the RawData class, providing data and labels.
        batch_size (int): Batch size for DataLoaders.
        val_ratio (float): Proportion of training data to use for validation.

    Returns:
        train_loader, val_loader, test_loader
    """
    # # Load raw data
    # raw_data = RawData(data_path)
    # print("Raw data loaded, labels: ", len(raw_data.labels_t()))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    print("Loading training data")
    if half:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.ConvertImageDtype(torch.half)
        ])
    else:
        train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,

            ])

    test_transform = transforms.Compose([
                           transforms.ToTensor(),
                            normalize
                       ])

    # Create the test dataset from the validation data in the original dataset
    test_dataset = TinyImageNetDataset(type_='val', raw_data=raw_data, transform=test_transform, force_reload=force_reload)
    print("Validation dataset created, size: ", len(test_dataset))

    # Create the full training dataset from the original training data
    full_train_dataset = TinyImageNetDataset(type_='train', raw_data=raw_data, transform=train_transform, force_reload=force_reload)
    print("Full training dataset created, size: ", len(full_train_dataset))

    # Calculate the sizes of the new training and validation sets
    full_train_size = len(full_train_dataset)
    val_size = int(full_train_size * val_ratio)
    train_size = full_train_size - val_size

    # Split the dataset into new training and validation datasets
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)

    print("DataLoaders created.")

    return train_loader, val_loader, test_loader

def train(model, iterator, optimizer, criterion, device='cpu', scaler=None, writer=None):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    with tqdm(total=len(iterator), desc='Train', leave=False) as t:
        for i, (x,label) in enumerate(iterator):
            x = x.to(device)
            y = label.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                y_pred, h = model(x)
                loss = criterion(y_pred, y)
                acc = calculate_accuracy(y_pred, y)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()


            # y_pred, h = model(x)
            # loss = criterion(y_pred, y)
            # acc = calculate_accuracy(y_pred, y)
            # loss.backward()
            # optimizer.step()


            epoch_loss += loss.item()
            epoch_acc += acc.item()
            t.set_postfix(loss=epoch_loss / (i + 1), acc=epoch_acc / (i + 1))
            if writer is not None:
                writer.add_scalar('Loss/train', loss.item(), i)
                writer.add_scalar('Accuracy/train', acc.item(), i)
            t.update(1)
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device='cpu', writer=None):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(iterator), desc='Eval', leave=False) as t:
            for i, (x, label) in enumerate(iterator):
                x = x.to(device)
                y = label.to(device)
                with autocast('cuda'):
                    y_pred, h = model(x)
                    loss = criterion(y_pred, y)
                    acc = calculate_accuracy(y_pred, y)
                # y_pred, h = model(x)
                # loss = criterion(y_pred, y)
                # acc = calculate_accuracy(y_pred, y)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                t.set_postfix(loss=epoch_loss / (i + 1), acc=epoch_acc / (i + 1))
                if writer is not None:
                    writer.add_scalar('Loss/val', loss.item(), i)
                    writer.add_scalar('Accuracy/val', acc.item(), i)
                t.update(1)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_model(model, num_epochs, train_loader, val_loader, optimizer, criterion, scheduler=None, save_dir=None, device='cpu', writer=None, half=False):
    log_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model = model.to(device)
    best_parms = model.state_dict()
    best_acc  = 0.0
    scaler = GradScaler('cuda') if half else None
    print("Training model on device: ", device)
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device=device, scaler=scaler, writer=writer)
            if scheduler is not None:
                scheduler.step()
            # Validate
            valid_loss, valid_acc = evaluate(model, val_loader, criterion, device=device, writer=writer)

            pbar.set_postfix(train_loss=train_loss, valid_loss=valid_loss)

            # Save the best model
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_parms = model.state_dict()

            # log_history['train_loss'].append(train_loss)
            # log_history['train_acc'].append(train_acc)
            # log_history['val_loss'].append(valid_loss)
            # log_history['val_acc'].append(valid_acc)
            pbar.update(1)
    log_history = {}
    if save_dir is not None:
        timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        save_path = os.path.join(save_dir, f"{timestamp}_model.pth")
        torch.save(best_parms, save_path)
        print('Best model saved as {}'.format(save_path))
        log_history['model'] = save_path

    return log_history

def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Classification Training on Tiny ImageNet", add_help=True)
    parser.add_argument('-d',"--data-path", type=str, default="./data/tiny-imagenet-200", help="Path to the Tiny ImageNet data")
    parser.add_argument('-o',"--save-dir", default="./out", type=str, help="path to save outputs (default: ./out)")
    parser.add_argument("--force-reload", action="store_true", help="Force reload of data")
    parser.add_argument('-m',"--model", type=str, default="resnet18", help="Model to use for training")
    parser.add_argument('-b',"--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument('-n',"--num-epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument('-opt',"--optimizer", default="sgd", type=str, help="optimizer", choices=["sgd", "adam", "adamw"])
    parser.add_argument('-lr',"--learning-rate", default=0.1, type=float, help="initial learning rate")
    parser.add_argument(
        "-wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--lr-scheduler", default="step", type=str, help="the lr scheduler (default: step)", choices=["step", "cosine", "exponential"])
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)", choices=["constant", "linear"])
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")

    parser.add_argument("--smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)")
    
    parser.add_argument("--wo-norm", action="store_false", help="without normalization in the model")
    parser.add_argument("--wo-skip", action="store_false", help="without skip connection in the model")
    parser.add_argument('--writer', action='store_true', help='write the log to tensorboard')
    parser.add_argument('--half', action='store_true', help='use half precision')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to the checkpoint')
    return parser

def main(args):

    # Set up the device
    data_path = args.data_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    save_dir = args.save_dir
    workers = args.workers
    force_reload = args.force_reload

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda available device counts = ", torch.cuda.device_count())
    if not torch.cuda.is_available():
        print("Eror: GPU is not available")
        sys.exit(1)
    # else:
    #     print_gpu_memory()
    # Load raw data
    raw_data = RawData(data_path)
    num_classes = len(raw_data.labels_t())
    print(f"Number of classes: {num_classes}")
    # Create DataLoader objects
    train_loader, val_loader, test_loader = DataLoaderSplit(raw_data, batch_size, val_ratio=0.2, force_reload=force_reload, workers=workers, half=args.half)

    # Create the model
    if args.model == "vgg11":
        model = VGG(vgg11_config, num_classes, use_norm=args.wo_norm)
    elif args.model == "vgg13":
        model = VGG(vgg13_config, num_classes, use_norm=args.wo_norm)
    elif args.model == "vgg16":
        model = VGG(vgg16_config, num_classes, use_norm=args.wo_norm)
    elif args.model == "vgg19":
        model = VGG(vgg19_config, num_classes, use_norm=args.wo_norm)
    elif args.model == "resnet18":
        model = ResNet(resnet18_config, num_classes, use_skip=args.wo_skip)
    elif args.model == "resnet34":
        model = ResNet(resnet34_config, num_classes, use_skip=args.wo_skip)
    elif args.model == "resnet50":
        model = ResNet(resnet50_config, num_classes, use_skip=args.wo_skip)
    elif args.model == "resnet101":
        model = ResNet(resnet101_config, num_classes, use_skip=args.wo_skip)
    elif args.model == "resnext50":
        model = ResNet(resnext50_32x4d_config, num_classes)
    elif args.model == "resnext101":
        model = ResNet(resnext101_32x4d_config, num_classes)
    elif args.model == "t2t_vit_14":
        model = T2T_ViT(t2t_vit_14_config, num_classes)
    elif args.model == "t2t_vit_t_14":
        model = T2T_ViT(t2t_vit_t_14_config, num_classes)
    else:
        raise ValueError(f"Model {args.model} not recognized.")
    
    print(f"Model: {args.model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if 'module.' in next(iter(state_dict)):
            state_dict = {k[7:]: v for k, v in state_dict.items()}  # 去掉module.前缀
        model.load_state_dict(state_dict)
        print(f"Model loaded from {args.checkpoint}")

    
    # Set up the optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not recognized.")
    print("Optimizer: ", optimizer)

    # Set up the learning rate scheduler
    if args.lr_scheduler == "step":
        main_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-args.lr_warmup_epochs, eta_min=args.lr_min)
    elif args.lr_scheduler == 'exponential':
        main_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        print("No scheduler")
        main_lr_scheduler = None
    print("LR Scheduler: ", main_lr_scheduler)

    # Set up the learning rate warmup
    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    
    
    timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    if args.writer:
        writer_log_dir = os.path.join("./logs", f"{args.model}_{timestamp}")
        writer = SummaryWriter(log_dir=writer_log_dir)
    else:
        writer = None

    # Train the model
    save_dir = os.path.join(save_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)
    log_history = train_model(model, num_epochs, train_loader, val_loader, optimizer, criterion, scheduler=lr_scheduler, save_dir=save_dir, device=device, writer=writer, half=args.half)

    # Evaluate the model on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Save the log history
    log_history['writer'] = writer_log_dir
    log_history['test_loss'] = test_loss
    log_history['test_acc'] = test_acc
    log_history['args'] = vars(args)
    # Create TensorBoard writer
    if writer is not None:
        writer.close()

    # 将log_history保存到json文件
    json_log_path = os.path.join(save_dir, f"{timestamp}_log.json")
    with open(json_log_path, 'w') as f:
        json.dump(log_history, f)
    print(f"Log history saved as {json_log_path}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)