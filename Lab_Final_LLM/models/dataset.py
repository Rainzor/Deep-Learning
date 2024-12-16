import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import random
import json
from collections import defaultdict
import os
import numpy as np
from typing import Optional, List, Union, Any

class KUAKE_Dataset(Dataset):
    _id2label = ["0", "1", "2"]
    _label2id = {label: i for i, label in enumerate(_id2label)}
    
    def __init__(self, rawdata, tokenizer, max_length=64, type_="train"):
        self.rawdata = rawdata
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.type = type_
        self._data = self._process_data(rawdata)

    @classmethod
    def label2id(cls, label):
        return cls._label2id[label]
    
    @classmethod
    def id2label(cls, idx):
        return cls._id2label[idx]

    def _process_data(self, rawdata):
        data = []
        for item in rawdata:
            keys = item["keys"]
            query = item["query"]
            if self.type == "train" or self.type == "valid":
                input_ids = []
                attention_mask = []
                for key in keys:
                    tokens = self.tokenizer(
                            text = query,
                            text_pair = key,
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt"
                            )
                    input_ids.append(tokens["input_ids"].squeeze(0))
                    attention_mask.append(tokens["attention_mask"].squeeze(0))
                input_ids = torch.stack(input_ids, dim=0)
                attention_mask = torch.stack(attention_mask, dim=0)
                labels = torch.tensor(item["label"], dtype=torch.long)
                keys_num = len(keys)
                data.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    'num': keys_num
                })
            elif self.type == "test":
                tokens = self.tokenizer(
                            text = query,
                            text_pair = keys[0],
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt"
                            )
                input_ids = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]
                data.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    'num': 1,
                    'id': item["id"],
                    'labels': None
                })
            else:
                raise ValueError("type must be 'train', 'valid' or 'test'")
        # if self.type == "train":
        #     data = self.shuffle(data) 
        return data
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
    
    def shuffle(self,dataset):
        dataset = sorted(dataset, key=lambda x: x['num'])
        small_indices = [i for i, item in enumerate(dataset) if item['num'] < 10]
        random.shuffle(small_indices)
        large_indices = list(range(len(small_indices), len(dataset)))
        random.shuffle(large_indices)
        shuffle_indices = small_indices + large_indices
        dataset = [dataset[i] for i in shuffle_indices]
        return dataset

@dataclass
class DataBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    num: torch.Tensor
    batch: torch.Tensor
    id: Optional[Union[List[str], str]] = None

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.labels = self.labels.to(device) if self.labels is not None else None
        self.num = self.num.to(device)
        self.batch = self.batch.to(device)
        return self

def custom_collate_fn(data_batch):
    max_num_keys = max([item["num"] for item in data_batch])
    input_ids = []
    attention_mask = []
    all_labels = torch.tensor([], dtype=torch.long)
    nums = []
    increment = 0
    batch = torch.tensor([], dtype=torch.long)
    ids = []
    for i, sample in enumerate(data_batch):
        input_ids.append(sample["input_ids"])
        attention_mask.append(sample["attention_mask"])
        labels = sample["labels"]                # [num_keys]
        num_keys = sample["num"]
        increment += num_keys
        nums.append(increment)
        if labels is not None:
            all_labels = torch.cat([all_labels, labels])
        batch = torch.cat([batch, torch.ones(num_keys, dtype=torch.long) * i])

        if "id" in sample.keys():
            ids.append(sample["id"])
    
    # 将列表转为tensor
    input_ids = torch.cat(input_ids, dim=0) # [num_keys, max_length]
    attention_mask = torch.cat(attention_mask, dim=0) # [num_keys, max_length]
    all_labels = all_labels if len(all_labels) > 0 else None
    nums = torch.tensor(nums, dtype=torch.long) # [num_keys]
    
    # 返回CustomBatch实例
    return DataBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=all_labels,
        num=nums,
        batch=batch,
        id=ids if len(ids) > 0 else None
    )
