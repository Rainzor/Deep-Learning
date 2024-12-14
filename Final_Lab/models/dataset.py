import torch
from torch.utils.data import Dataset
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
            q_token = self.tokenizer(item["query"], 
                            padding='max_length', # padding to max_length
                            truncation=True, # cut off to max_length if too long
                            max_length=self.max_length, # set max_length
                            return_tensors="pt" # return PyTorch tensor
                            )
            keys = item["keys"]
            if self.type == "train" or self.type == "valid":
                tokens = self.tokenizer(keys,
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt"
                            )
                keys_token = tokens["input_ids"]
                keys_mask = tokens["attention_mask"]
                labels = torch.tensor(item["label"], dtype=torch.long)
                keys_num = keys_token.size(0)
                data.append({
                    "query": (q_token["input_ids"], q_token["attention_mask"]),
                    "keys": (keys_token,keys_mask),
                    "labels": labels,
                    'num': keys_num
                })
            elif self.type == "test":
                tokens = self.tokenizer(keys[0],
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt"
                            )
                keys_token = tokens["input_ids"]
                keys_mask = tokens["attention_mask"]
                data.append({
                    "query": (q_token["input_ids"], q_token["attention_mask"]),
                    "keys": (keys_token,keys_mask),
                    'num': 1,
                    'id': item["id"],
                    'labels': None
                })
            else:
                raise ValueError("type must be 'train', 'valid' or 'test'")
        if self.type == "train":
            data = self.shuffle(data) 
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

def custom_collate_fn(batch):
    # batch 是一个列表，每个元素为 dataset 的 __getitem__ 返回的字典
    # 找出 batch 中最大数量的 keys
    max_num_keys = max([item["num"] for item in batch])
    queries_input_ids = []
    queries_attention_mask = []
    keys_input_ids = []
    keys_attention_mask = []
    all_labels = []
    num_keys_list = []
    batch_mask = []
    nums = 0
    for sample in batch:
        q_ids, q_mask = sample["query"]          # q_ids: [1, seq_len], q_mask: [1, seq_len]
        k_ids, k_mask = sample["keys"]           # k_ids: [num_keys, seq_len], k_mask: [num_keys, seq_len]
        labels = sample["labels"]                # [num_keys]
        num_keys = sample["num"]
        nums += num_keys
        mask = torch.ones(max_num_keys, dtype=torch.bool)
        # # 去掉 query 的额外维度 [1, seq_len] -> [seq_len]
        # q_ids = q_ids.squeeze(0)
        # q_mask = q_mask.squeeze(0)

        # 如果 num_keys 小于 max_num_keys，需要填充
        if num_keys < max_num_keys:
            pad_size = max_num_keys - num_keys
            # 对 k_ids 和 k_mask 在 num_keys 维度上填充 0
            k_ids = torch.cat([k_ids, torch.zeros((pad_size, k_ids.size(1)), dtype=k_ids.dtype)], dim=0)
            k_mask = torch.cat([k_mask, torch.zeros((pad_size, k_mask.size(1)), dtype=k_mask.dtype)], dim=0)
            # 对 labels 也进行填充，假设用 -1e6 填充
            if labels is not None:
                labels = torch.cat([labels, -1e6 * torch.ones(pad_size, dtype=torch.long)], dim=0)
            mask[-pad_size:] = False
        
        queries_input_ids.append(q_ids)
        queries_attention_mask.append(q_mask)
        keys_input_ids.append(k_ids)
        keys_attention_mask.append(k_mask)
        if labels is not None:
            all_labels.append(labels)
        num_keys_list.append(nums)
        batch_mask.append(mask)
    
    # 将列表转为tensor
    queries_input_ids = torch.stack(queries_input_ids, dim=0)       # [batch_size, seq_len]
    queries_attention_mask = torch.stack(queries_attention_mask, dim=0) # [batch_size, seq_len]
    keys_input_ids = torch.stack(keys_input_ids, dim=0)             # [batch_size, max_num_keys, seq_len]
    keys_attention_mask = torch.stack(keys_attention_mask, dim=0)   # [batch_size, max_num_keys, seq_len]
    all_labels = torch.stack(all_labels, dim=0) if len(all_labels) > 0 else None
    num_keys_list = torch.tensor(num_keys_list, dtype=torch.long)   # [batch_size]
    batch_mask = torch.stack(batch_mask, dim=0)                     # [batch_size, max_num_keys]
    return {
        "query": (queries_input_ids, queries_attention_mask),
        "keys": (keys_input_ids, keys_attention_mask),
        "labels": all_labels[batch_mask].to(torch.long) if all_labels is not None else None,
        "num": num_keys_list,
        "mask": batch_mask
    }