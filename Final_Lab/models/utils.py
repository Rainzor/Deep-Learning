import os
import json
import random
import numpy as np
import torch
from collections import defaultdict
from typing import List, Union, Any
from collections.abc import Mapping
from .dataset import KUAKE_Dataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


def load_data(data_dir, task_name):
    dir_path = os.path.join(data_dir, task_name)
    train_path = os.path.join(dir_path, f"{task_name}_train.json")
    dev_path = os.path.join(dir_path, f"{task_name}_dev.json")
    test_path = os.path.join(dir_path, f"{task_name}_test.json")

    def read_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def preprocess_train_dev(samples):
        grouped_data = defaultdict(lambda: [[], []])
        for sample in samples:
            label = sample.get("label", None)
            if label == "NA":
                continue
            try:
                label = int(label)
            except (ValueError, TypeError):
                continue
            query1 = sample["query1"]
            query2 = sample["query2"]
            grouped_data[query1][0].append(query2)
            grouped_data[query1][1].append(label)
        for query1, keys in grouped_data.items():
            find_quary = False
            for quary2 in keys[0]:
                if quary2 == query1:
                    find_quary = True
                    break
            
            if not find_quary:
                keys[0].append(query1)
                keys[1].append(2)

        
        # 转换为列表形式
        processed_samples = [{
            "query": query1,
            "keys": keys[0],
            "label": keys[1]
        } for query1, keys in grouped_data.items()]
        return processed_samples

    def preprocess_test(samples):
        grouped_data = defaultdict(list)
        processed_samples = []
        for sample in samples:
            processed_samples.append({
                "query": sample["query1"],
                "keys": [sample["query2"]],
                "id": sample["id"]
            })
        return processed_samples

    def load_and_preprocess(file_path, is_test=False):
        data = read_file(file_path)
        if is_test:
            return preprocess_test(data)
        else:
            return preprocess_train_dev(data)

    return {
        "train": load_and_preprocess(train_path, is_test=False),
        "valid": load_and_preprocess(dev_path, is_test=False),
        "test": load_and_preprocess(test_path, is_test=True),
    }

def prepare_input(data: Union[torch.Tensor, Any], device: str = 'cuda'):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data

def generate_commit(output_dir, task_name, test_dataset, preds: List[int]):
    pred_test_examples = []
    for idx in range(len(test_dataset)):
        example = test_dataset[idx]
        label  = KUAKE_Dataset.id2label(preds[idx])
        pred_example = {'id': example['id'], 'query1': example['query'], 'query2': example['keys'][0], 'label': label}
        pred_test_examples.append(pred_example)
    
    with open(os.path.join(output_dir, f'{task_name}_test.json'), 'w', encoding='utf-8') as f:
        json.dump(pred_test_examples, f, indent=2, ensure_ascii=False)