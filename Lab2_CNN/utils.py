
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from multiprocessing import Pool, cpu_count
from PIL import Image
import matplotlib.pyplot as plt

def load_image_multiprocess(args):
    """Single image loading function for use with multiprocessing."""
    image_path, label = args
    image = cv2.imread(image_path) # BGR
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return image, label

def apply_transform_multiprocess(args):
    """
    Single image transformation function for use with multiprocessing.
    """
    img, transform = args
    img_pil = Image.fromarray(img)  # 转换为 PIL Image
    return transform(img_pil)


def calculate_accuracy(y_pred: torch.Tensor, y: torch.Tensor):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def print_gpu_memory():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        torch.cuda.set_device(device)
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MiB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # MiB
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # MiB
        free = total - reserved
        gpu_name = torch.cuda.get_device_name(device)
        print(f"GPU {i}: {gpu_name}")
        print(f"  Total Memory: {total:.2f} MiB")
        print(f"  Allocated Memory: {allocated:.2f} MiB")
        print(f"  Reserved Memory : {reserved:.2f} MiB")
        print(f"  Free Memory : {free:.2f} MiB\n")

def plot_results(train_data, val_data, models):
    # 定义一个简单的移动平均函数
    def moving_average(data, window_size=5):
        return data.rolling(window=window_size, min_periods=1).mean()

    # 创建一个图形，包含两个子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # 子图1: 对比 Train 数据
    axes[0].set_title('Comparison of Train Data (Log Scale)', fontsize=16)
    for model in models:
        smoothed_train = moving_average(train_data[model]['Value'], window_size=5)
        axes[0].plot(train_data[model]['Step'], smoothed_train, label=f'{model.split("_")[0]} Train')
    axes[0].set_xlabel('Step', fontsize=14)
    axes[0].set_ylabel('Loss (Log Scale)', fontsize=14)
    axes[0].set_yscale('log')  # 设置Y轴为对数尺度
    axes[0].legend(fontsize=14)
    axes[0].grid(True)

    # 子图2: 对比 Val 数据
    axes[1].set_title('Comparison of Val Data', fontsize=16)
    for model in models:
        smoothed_val = moving_average(val_data[model]['Value'], window_size=5) 
        axes[1].plot(val_data[model]['Step'], smoothed_val, label=f'{model.split("_")[0]} Val')
    axes[1].set_xlabel('Step', fontsize=14)
    axes[1].set_ylabel('Accuracy', fontsize=14)
    axes[1].legend(fontsize=14)
    axes[1].grid(True)

    fig.tight_layout()

    plt.show()
