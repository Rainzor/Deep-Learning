
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from multiprocessing import Pool, cpu_count
from PIL import Image

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
