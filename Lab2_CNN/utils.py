
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

class RawData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.__labels_t_path = os.path.join(self.data_path, "wnids.txt")
        self.__train_data_path = os.path.join(self.data_path, "train/")
        self.__val_data_path = os.path.join(self.data_path, "val/")

        self.__labels_t = None
        self.__image_names = None
        self.__val_labels_t = None
        self.__val_labels = None
        self.__val_names = None

    def train_data_path(self):
        return self.__train_data_path

    def val_data_path(self):
        return self.__val_data_path

    def labels_t(self):
        if self.__labels_t is None:
            labels_t = []
            with open(self.__labels_t_path) as wnid:
                for line in wnid:
                    labels_t.append(line.strip('\n'))
            self.__labels_t = labels_t
        return self.__labels_t

    def image_names(self):
        if self.__image_names is None:
            image_names = []
            labels_t = self.labels_t()
            for label in labels_t:
                txt_path = os.path.join(self.__train_data_path, label, f"{label}_boxes.txt")
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip('\n').split('\t')[0])
                image_names.append(image_name)
            self.__image_names = image_names
        return self.__image_names

    def val_names(self):
        if self.__val_names is None:
            val_names = []
            with open(os.path.join(self.__val_data_path, "val_annotations.txt")) as txt:
                for line in txt:
                    val_names.append(line.strip('\n').split('\t')[0])
            self.__val_names = val_names
        return self.__val_names

    def val_labels_t(self):
        if self.__val_labels_t is None:
            val_labels_t = []
            with open(os.path.join(self.__val_data_path, "val_annotations.txt")) as txt:
                for line in txt:
                    val_labels_t.append(line.strip('\n').split('\t')[1])
            self.__val_labels_t = val_labels_t
        return self.__val_labels_t

    def val_labels(self):
        if self.__val_labels is None:
            val_labels = []
            val_labels_t = self.val_labels_t()
            labels_t = self.labels_t()
            for v in val_labels_t:
                val_labels.append(labels_t.index(v))
            self.__val_labels = np.array(val_labels)
        return self.__val_labels


class TinyImageNetDataset(Dataset):
    def __init__(self, type_, raw_data, transform=None, force_reload=False):
        """
        type_: 'train' or 'val'
        raw_data: RawData instance
        transform: torchvision transforms to apply
        force_reload: If True, ignore cached data and reprocess
        """
        self.type = type_
        self.raw_data = raw_data
        self.force_reload = force_reload

        # Create a directory to save processed data
        self.processed_path = os.path.join(self.raw_data.data_path, "process")
        os.makedirs(self.processed_path, exist_ok=True)

        # Load or preprocess data
        if self.type == "train":
            self.images, self.labels = self._load_or_preprocess_train_data()
        else:
            self.images, self.labels = self._load_or_preprocess_val_data()

        self.transform = transforms.ToTensor() if transform is None else transform


    def _load_or_preprocess_train_data(self):
        train_data_file = os.path.join(self.processed_path, "train_data.npz")
        if os.path.exists(train_data_file) and not self.force_reload:
            print(f"Loading preprocessed training data from {train_data_file}...")
            data = np.load(train_data_file)
            images = data["images"]
            labels = data["labels"]
        else:
            print("Preprocessing training data...")
            images, labels = self._preload_train_data()
            print(f"Saving preprocessed training data to {train_data_file}...")
            np.savez(train_data_file, images=images, labels=labels)
        return images, labels

    def _load_or_preprocess_val_data(self):
        val_data_file = os.path.join(self.processed_path, "val_data.npz")
        if os.path.exists(val_data_file) and not self.force_reload:
            print(f"Loading preprocessed validation data from {val_data_file}...")
            data = np.load(val_data_file)
            images = data["images"]
            labels = data["labels"]
        else:
            print("Preprocessing validation data...")
            images, labels = self._preload_val_data()
            print(f"Saving preprocessed validation data to {val_data_file}...")
            np.savez(val_data_file, images=images, labels=labels)
        return images, labels

    def _preload_train_data(self):
        labels_t = self.raw_data.labels_t()
        image_names = self.raw_data.image_names()
        tasks = []
        for label_idx, image_list in enumerate(image_names):
            label_dir = os.path.join(self.raw_data.train_data_path(), labels_t[label_idx], "images")
            for image_name in image_list:
                image_path = os.path.join(label_dir, image_name)
                tasks.append((image_path, label_idx))

        num_workers = min(cpu_count(), 16)
        print(f"Using {num_workers} workers for multiprocessing (train)...")
        with Pool(processes=num_workers) as pool:
            results = pool.map(load_image_multiprocess, tasks)

        train_images, train_labels = zip(*results)
        train_labels = np.array(train_labels)
        return train_images, train_labels

    def _preload_val_data(self):
        val_names = self.raw_data.val_names()
        val_labels = self.raw_data.val_labels()
        tasks = [(os.path.join(self.raw_data.val_data_path(), "images", image_name), label) 
                 for image_name, label in zip(val_names, val_labels)]

        num_workers = min(cpu_count(), 16)
        print(f"Using {num_workers} workers for multiprocessing (val)...")
        with Pool(processes=num_workers) as pool:
            results = pool.map(load_image_multiprocess, tasks)

        val_images, val_labels = zip(*results)
        val_labels = np.array(val_labels)
        return val_images, val_labels

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.labels)


def calculate_accuracy(y_pred: torch.Tensor, y: torch.Tensor):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
