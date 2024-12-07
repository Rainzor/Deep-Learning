import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
import time
import os
import sys
import argparse
import json

from models.VGG import *
from models.ResNet import *
from utils import *
from config import *


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def DataLoaderSplit(raw_data, batch_size, val_ratio=0.2, force_reload=False, workers=1, distributed=False, rank=0, world_size=1):
    """
    Prepare DataLoaders for training, validation, and testing.
    If distributed=True, use DistributedSampler for training and validation sets.
    """
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    if rank == 0:
        print("Loading training data")
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

    # test dataset
    test_dataset = TinyImageNetDataset(type_='val', raw_data=raw_data, transform=test_transform, force_reload=force_reload)
    if rank == 0:
        print("Validation dataset created, size: ", len(test_dataset))

    # full training dataset
    full_train_dataset = TinyImageNetDataset(type_='train', raw_data=raw_data, transform=train_transform, force_reload=force_reload)
    if rank == 0:
        print("Full training dataset created, size: ", len(full_train_dataset))

    # split train/val
    full_train_size = len(full_train_dataset)
    val_size = int(full_train_size * val_ratio)
    train_size = full_train_size - val_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 使用DistributedSampler
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), pin_memory=True, num_workers=workers, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers, sampler=test_sampler)

    if rank == 0:
        print("DataLoaders created.")

    return train_loader, val_loader, test_loader


def train(model, iterator, optimizer, criterion, device, rank=0):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    if rank == 0:
        pbar = tqdm(total=len(iterator), desc='Train', leave=False)
    else:
        pbar = None
    for i, (x,label) in enumerate(iterator):
        x = x.to(device)
        y = label.to(device)
        optimizer.zero_grad()
        y_pred, h = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        if rank == 0:
            pbar.set_postfix(loss=epoch_loss / (i + 1), acc=epoch_acc / (i + 1)*100)
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # 在分布式环境下，需要对loss和acc进行reduce求平均
    avg_loss = torch.tensor(epoch_loss / len(iterator), device=device)
    avg_acc = torch.tensor(epoch_acc / len(iterator), device=device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss / dist.get_world_size()
    avg_acc = avg_acc / dist.get_world_size()

    return avg_loss.item(), avg_acc.item()


def evaluate(model, iterator, criterion, device, rank=0):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    if rank == 0:
        pbar = tqdm(total=len(iterator), desc='Eval', leave=False)
    else:
        pbar = None
    with torch.no_grad():
        for i, (x, label) in enumerate(iterator):
            x = x.to(device)
            y = label.to(device)
            y_pred, h = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if rank == 0:
                pbar.set_postfix(loss=epoch_loss / (i + 1), acc=epoch_acc / (i + 1))
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # 分布式求平均
    avg_loss = torch.tensor(epoch_loss / len(iterator), device=device)
    avg_acc = torch.tensor(epoch_acc / len(iterator), device=device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss / dist.get_world_size()
    avg_acc = avg_acc / dist.get_world_size()

    return avg_loss.item(), avg_acc.item()

def train_model(model, num_epochs, train_loader, val_loader, optimizer, criterion, scheduler=None, save_path=None, device='cpu', rank=0):
    log_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    best_val_acc = 0.0
    if rank == 0:
        pbar = tqdm(total=num_epochs)
    else:
        pbar = None

    for epoch in range(num_epochs):
        # 在分布式训练中，每个epoch需要对sampler进行一次set_epoch，以便随机种子同步
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, rank)
        if scheduler is not None:
            scheduler.step()
        valid_loss, valid_acc = evaluate(model, val_loader, criterion, device, rank)

        if rank == 0:
            pbar.set_postfix(train_loss=train_loss, valid_loss=valid_loss)

            # Save the best model
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_parms = {k:v.cpu() for k,v in model.module.state_dict().items()} if isinstance(model, DDP) else model.state_dict()

            log_history['train_loss'].append(train_loss)
            log_history['val_loss'].append(valid_loss)
            log_history['train_acc'].append(train_acc)
            log_history['val_acc'].append(valid_acc)
            log_history['lr'].append(optimizer.param_groups[0]['lr'])
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    if save_path is not None and rank == 0:
        timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        save_path = os.path.join(save_path, f"{timestamp}_ddp_model.pth")
        torch.save(best_parms, save_path)
        print('Best model saved as {}'.format(save_path))
        log_history['model'] = save_path

    return model, log_history

def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Classification Training on Tiny ImageNet", add_help=True)
    parser.add_argument('-i',"--data-path", type=str, default="./data/tiny-imagenet-200", help="Path to the Tiny ImageNet data")
    parser.add_argument("--force-reload", action="store_true", help="Force reload of data")
    parser.add_argument('-m',"--model", type=str, default="resnet18", help="Model to use for training")
    parser.add_argument('-b',"--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument('-n',"--num-epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument('-o',"--save-dir", default="./out", type=str, help="path to save outputs (default: ./out)")

    parser.add_argument('-opt',"--optimizer", default="sgd", type=str, help="optimizer", choices=["sgd", "adam"])
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

    return parser

def main(args):
    # 初始化分布式训练环境
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    data_path = args.data_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    save_dir = args.save_dir
    workers = args.workers
    force_reload = args.force_reload

    # 只在rank=0打印
    if rank == 0:
        print("Running distributed training on {} GPUs.".format(world_size))

    # Load raw data
    raw_data = RawData(data_path)
    num_classes = len(raw_data.labels_t())
    if rank == 0:
        print(f"Number of classes: {num_classes}")
    # Create DataLoader objects
    train_loader, val_loader, test_loader = DataLoaderSplit(raw_data, batch_size, val_ratio=0.2, force_reload=force_reload, workers=workers, distributed=True, rank=rank, world_size=world_size)

    # Create the model
    if args.model == "resnet18":
        model = ResNet(resnet18_config, num_classes)
    elif args.model == "resnet34":
        model = ResNet(resnet34_config, num_classes)
    elif args.model == "resnet50":
        model = ResNet(resnet50_config, num_classes)
    elif args.model == "resnet101":
        model = ResNet(resnet101_config, num_classes)
    elif args.model == "resnet152":
        model = ResNet(resnet152_config, num_classes)
    elif args.model == "vgg11":
        model = VGG11(vgg11_config, num_classes)
    elif args.model == "vgg13":
        model = VGG13(vgg13_config, num_classes)
    elif args.model == "vgg16":
        model = VGG16(vgg16_config, num_classes)
    elif args.model == "vgg19":
        model = VGG19(vgg19_config, num_classes)
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    model.to(device)

    # Set up the optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not recognized.")

    # Set up the learning rate scheduler
    if args.lr_scheduler == "step":
        main_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-args.lr_warmup_epochs, eta_min=args.lr_min)
    elif args.lr_scheduler == 'exponential':
        main_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        main_lr_scheduler = None

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

    criterion = nn.CrossEntropyLoss()

    # 使用DDP包装模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # rank=0的进程才进行输出目录和日志记录
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, args.model)
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Train the model
    model, log_history = train_model(model, num_epochs, train_loader, val_loader, optimizer, criterion, scheduler=lr_scheduler, save_path=save_dir, device=device, rank=rank)

    # Evaluate the model on the test set (在evaluate中会自动all_reduce)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, rank=rank)
    if rank == 0:
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Save the log history
        timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        log_save_path = os.path.join(save_dir, f"{timestamp}_ddp_log.json")
        log_history['test_loss'] = test_loss
        log_history['test_acc'] = test_acc
        log_history['args'] = vars(args)

        with open(log_save_path, 'w') as f:
            json.dump(log_history, f)

        print(f"Log history saved as {log_save_path}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
