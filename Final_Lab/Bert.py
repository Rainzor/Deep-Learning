from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import os
import time 
import json
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Union, Any
from collections.abc import Mapping
from torch.optim import AdamW
from tqdm import tqdm
from dataclasses import dataclass, field

from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_linear_schedule_with_warmup
import argparse

from torch.utils.tensorboard import SummaryWriter

import os
import json
from collections import defaultdict


# 配置参数
MODEL_DIR = "hfl/chinese-bert-wwm-ext"
DATA_DIR = "../data"
OUTPUT_DIR = "./output_data"
TASK_NAME = "KUAKE-QQR"
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 3
LABELS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DataTrainingArguments:

    model_dir: str = field(
        default= MODEL_DIR,
        metadata={'help': 'The pretrained model directory'}
    )
    data_dir: str = field(
        default=DATA_DIR,
        metadata={'help': 'The data directory'}
    )
    max_length: int = field(
        default=MAX_LENGTH,
        metadata={'help': 'Maximum sequence length allowed to input'}
    )

    task_name: str = field(
        default=TASK_NAME,
        metadata={'help': 'The name of the task to train on'}
    )

    labels: int = field(
        default=LABELS,
        metadata={'help': 'The number of labels in the dataset'}
    )

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
        
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

@dataclass
class TrainingArguments:

    output_dir: str = field(
        default='output_data/',
        metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'}
    )
    train_batch_size: int = field(
        default=4,
        metadata={'help': 'batch size for training'}
    )
    eval_batch_size: int = field(
        default=1,
        metadata={'help': 'batch size for evaluation'}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': 'Number of updates steps to accumulate before performing a backward/update pass.'}
    )
    num_train_epochs: int = field(
        default=EPOCHS,
        metadata={"help": "The total number of training epochs"}
    )
    learning_rate: float = field(
        default=3e-5,
        metadata={'help': '"The initial learning rate for AdamW.'}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW"}
    )
    warmup_ratio: float = field(
        default=0.05,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Number of subprocesses to use for data loading (PyTorch only)"}
    )
    
    logging_steps: int = field(
        default=100,
        metadata={'help': 'logging states every X updates steps.'}
    )
    eval_steps: int = field(
        default=50,
        metadata={'help': 'Run an evaluation every X steps.'}
    )
    device: str = field(
        default= "cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": 'The device used for training'}
    )

    def get_warmup_steps(self, num_training_steps):
        return int(num_training_steps * self.warmup_ratio)

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
        
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


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
                "keys": [sample["query2"]]
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


class ContrastiveDataset(Dataset):
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

# 定义自定义模型
# 定义自定义模型
class QKModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(QKModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )


    def forward(self, data):
        """
        Parameters
        ----------
        query_input_ids : Shape: (batch_size, 1, seq_len)
        query_attention_mask : Shape: (batch_size, 1, seq_len)
        key_input_ids : Shape: (batch_size, max_num_keys, seq_len)
        key_attention_mask : Shape: (batch_size, max_num_keys, seq_len)
        nums : Shape: (batch_size,)
        """
        query_input_ids, query_attention_mask = data["query"]
        masks = data["mask"]   
        key_input_ids, key_attention_mask = data["keys"]
        nums = data["num"]

        B, N, L = key_input_ids.size()
        key_input_ids = key_input_ids[masks] # shape: (key_num, seq_len)
        key_attention_mask = key_attention_mask[masks] # shape: (key_num, seq_len)

        # Query Encoder
        query_outputs = self.encoder(input_ids=query_input_ids.view(B,L), attention_mask=query_attention_mask.view(B,L)) # shape: (batch_size, seq_len, hidden_size)
        query_cls = query_outputs.last_hidden_state[:, 0, :]  # CLS token shape: (batch_size, hidden_size)
        query_cls = query_cls.unsqueeze(1).expand(-1, N, -1)  # shape: (batch_size, max_num_keys, hidden_size)
        query_cls = query_cls[masks] # shape: (key_num, hidden_size)

        # Key Encoder
        key_outputs = self.encoder(input_ids=key_input_ids, attention_mask=key_attention_mask) # shape: (key_num, seq_len, hidden_size)
        key_cls = key_outputs.last_hidden_state[:, 0, :]  # CLS token shape: (key_num, hidden_size)
        
        combined = torch.cat([query_cls, key_cls], dim=-1)  # shape: (key_num, hidden_size * 2)

        # 分类
        logits = self.classifier(combined)  # shape: (key_num, num_labels)

        return logits

def create_optimizer_and_scheduler(
    args: TrainingArguments,
    model: nn.Module,
    num_training_steps: int,
):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_training_steps=num_training_steps, 
        num_warmup_steps=args.get_warmup_steps(num_training_steps)
    )

    return optimizer, scheduler



def _prepare_input(data: Union[torch.Tensor, Any], device: str = 'cuda'):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data

def train(model, data, optimizer, scheduler, criterion, device):
    model.train()

    optimizer.zero_grad()
    data = _prepare_input(data, device)
    batch_size = data["query"][0].size(0)
    labals = data["labels"]

    # 前向传播
    logits = model(data)
    nums = 0
    loss = criterion(logits, labals)
    correct = (torch.argmax(logits, dim=1) == labals).sum().item()
    correct /= len(labals)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()
    return loss.item(), correct

# 定义评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    eval_correct = 0
    eval_loss = 0.0
    with torch.no_grad():
        with tqdm(dataloader, desc="Evaluation", leave=False) as pbar:
            for data in dataloader:
                data = _prepare_input(data, device)
                batch_size = data["query"][0].size(0)
                labels = data["labels"]
                # 前向传播
                logits = model(data)
                loss = criterion(logits, labels).item()
                correct = (torch.argmax(logits, dim=1) == labels).sum().item()
                eval_correct += correct/len(labels)
                eval_loss += loss   
                pbar.update(1)
                
    return eval_loss/len(dataloader), eval_correct/len(dataloader)


def train_model(model, train_loader, valid_loader, train_args, tokenizer, writer):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = create_optimizer_and_scheduler(train_args, model, len(train_loader) * train_args.num_train_epochs)
    # 开始训练
    best_val_acc = 0.0
    best_steps = 0
    log_history = {
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": []
        }

    with tqdm(range(train_args.num_train_epochs* len(train_loader)), desc="Epochs") as epochs_pbar:
        global_steps = 0
        for epoch in range(train_args.num_train_epochs):
            epoch_loss= 0
            epoch_correct = 0
            epoch_total = 0
            val_loss = 0
            val_acc = 0
            for batch in train_loader:
                global_steps += 1
                
                train_loss, train_acc = train(model, batch, optimizer, scheduler, criterion, train_args.device)
                epoch_loss += train_loss
                epoch_correct += train_acc
                epoch_total += 1
                
                if (global_steps+1) % train_args.eval_steps == 0:
                    
                    val_loss, val_acc = evaluate(model, valid_loader, criterion, train_args.device)

                    writer.add_scalar("Loss/train", epoch_loss / epoch_total, global_steps)
                    writer.add_scalar("Accuracy/train", epoch_correct / epoch_total, global_steps)
                    writer.add_scalar("Loss/eval", val_loss, global_steps)
                    writer.add_scalar("Accuracy/eval", val_acc, global_steps)


                    # 保存最佳模型
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_steps = epoch
                        os.makedirs(train_args.output_dir, exist_ok=True)
                        save_dir = os.path.join(train_args.output_dir, f"checkpoint-{best_steps}")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

                        tokenizer.save_pretrained(save_directory=save_dir)
                
                epochs_pbar.set_postfix({
                    "train loss": epoch_loss / epoch_total,
                    "train acc": epoch_correct / epoch_total,
                    "eval loss": val_loss,
                    "eval acc": val_acc
                })
                epochs_pbar.update(1)
    return best_val_acc, best_steps


def predict(
    args: TrainingArguments,
    model: nn.Module,
    test_dataloader
):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for item in test_dataloader:
            inputs = _prepare_input(item, device=args.device)
            outputs = model(inputs)

            preds = torch.argmax(outputs.cpu(), dim=-1).numpy()
            preds_list.append(preds)

    print(f'Prediction Finished!')
    preds = np.concatenate(preds_list, axis=0).tolist()

    return preds

def generate_commit(output_dir, task_name, test_dataset, preds: List[int]):

    pred_test_examples = []
    for idx in range(len(test_dataset)):
        example = test_dataset[idx]
        label  = ContrastiveDataset.id2label(preds[idx])
        # pred_example = {'id': example.guid, 'query1': example.text_a, 'query2': example.text_b, 'label': label}
        pred_example = {'id': example['id'], 'query1': example['query'], 'query2': example['keys'][0], 'label': label}
        pred_test_examples.append(pred_example)
    
    with open(os.path.join(output_dir, f'{task_name}_test.json'), 'w', encoding='utf-8') as f:
        json.dump(pred_test_examples, f, indent=2, ensure_ascii=False)

def args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Question-Keyword Matching Training")

    parser.add_argument("--model-dir",'-m', default=MODEL_DIR, type=str, help="The pretrained model directory")
    parser.add_argument("--data-dir",'-d', default=DATA_DIR, type=str, help="The data directory")
    parser.add_argument("--output-dir",'-o', default=OUTPUT_DIR, type=str, help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--epochs",'-n', default=EPOCHS, type=int, help="The total number of training epochs")
    parser.add_argument("--batch-size",'-b', default=BATCH_SIZE, type=int, help="batch size for training")

    parser.add_argument("--learning-rate",'-lr', default=3e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--weight-decay",'-wd', default=0.0, type=float, help="Weight decay for AdamW")
    parser.add_argument("--warmup-ratio",'-wr', default=0.05, type=float, help="Linear warmup over warmup_ratio fraction of total steps.")


    return parser.parse_args()

def main(args):

    set_seed(42)

    data_args = DataTrainingArguments(data_dir=args.data_dir, model_dir=args.model_dir)
    train_args = TrainingArguments()

    model_name = f'bert-{str(int(time.time()))}'
    train_args.output_dir = os.path.join(train_args.output_dir, model_name)

    writer = SummaryWriter(log_dir=train_args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(data_args.model_dir)

    rawdata = load_data(data_args.data_dir, data_args.task_name)

    train_dataset = ContrastiveDataset(rawdata["train"], tokenizer, max_length=data_args.max_length, type_='train')
    valid_dataset = ContrastiveDataset(rawdata["valid"], tokenizer, max_length=data_args.max_length, type_='valid')
    test_dataset = ContrastiveDataset(rawdata["test"], tokenizer, max_length=data_args.max_length, type_='test')

    # 使用自定义的 collate_fn 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=train_args.train_batch_size, shuffle=False, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=train_args.eval_batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=train_args.eval_batch_size, shuffle=False)

    model = QKModel(data_args.model_dir, data_args.labels)
    model.to(train_args.device)

    print("Start training...")
    best_acc, best_steps = train_model(model, train_loader, valid_loader, train_args, tokenizer, writer)

    print(f'Training Finished! Best step - {best_steps} - Best accuracy {best_acc}')
    
    best_state_dict = torch.load(os.path.join(train_args.output_dir, f"checkpoint-{best_steps}", "model.pth"), weights_only=True)
    
    model.load_state_dict(best_state_dict)

    torch.save(model.state_dict(), os.path.join(train_args.output_dir, "model.pth"))

    tokenizer.save_pretrained(save_directory=train_args.output_dir)
    print(f"Final model and tokenizer are saved to {train_args.output_dir}")

    preds = predict(train_args, model, test_loader)
    generate_commit(train_args.output_dir, data_args.task_name, rawdata["test"], preds)
    print(f"Commit file generated to {train_args.output_dir}")

if __name__ == "__main__":
    args = args_parser()
    main(args)




