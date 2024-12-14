from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from torch.utils.data import Dataset
import numpy as np
import os
import random
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from collections import defaultdict
from typing import List, Union, Any
from collections.abc import Mapping

from torch.utils.tensorboard import SummaryWriter
import argparse

from models.utils import (
            DataTrainingArguments, 
            TrainingArguments,
            set_seed,  
            args_parser,
            create_optimizer_and_scheduler)
from models.model import Bert

# 初始化路径和任务参数
MODEL_DIR = "hfl/chinese-bert-wwm-ext"
DATA_DIR = "../data"
OUTPUT_DIR = "./output_data"
TASK_NAME = "KUAKE-QQR"
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 3
LABELS = 3



def load_data(data_dir, task_name, argument=False):
    train_path = os.path.join(data_dir, f"{task_name}_train.json")
    dev_path = os.path.join(data_dir, f"{task_name}_dev.json")
    test_path = os.path.join(data_dir, f"{task_name}_test.json")

    def read_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def preprocess_valid(samples):
        processed_samples = []
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
            processed_samples.append({
                "text_a": query1,
                "text_b": query2,
                "label": label,
            })
        return processed_samples

    def preprocess_train(samples):
        processed_samples = []
        
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
        

        new_grouped_data = defaultdict(lambda: [[], []])
        if argument:
            for query, keys in grouped_data.items():
                key, label = keys
                key2, key1, key0 = [], [], []
                for i in range(len(label)):
                    if label[i] == 2:
                        key2.append(key[i])
                    elif label[i] == 1:
                        key1.append(key[i])
                    else:
                        key0.append(key[i])

                for i in range(len(key2)):

                    if query != key2[i]:
                        for j in range(i+1, len(key2)):
                            new_grouped_data[key2[i]][0].append(key2[j])
                            new_grouped_data[key2[i]][1].append(2)

                            new_grouped_data[key2[j]][0].append(key2[i])
                            new_grouped_data[key2[j]][1].append(2)

                        new_grouped_data[key2[i]][0].append(query)
                        new_grouped_data[key2[i]][1].append(2)

                        for j in range(len(key1)):
                            new_grouped_data[key2[i]][0].append(key1[j])
                            new_grouped_data[key2[i]][1].append(1)

                            # new_grouped_data[key1[j]][0].append(key2[i])
                            # new_grouped_data[key1[j]][1].append(0)
                        
                        for j in range(len(key0)):
                            new_grouped_data[key2[i]][0].append(key0[j])
                            new_grouped_data[key2[i]][1].append(0)

        grouped_data.update(new_grouped_data)
        for query1, keys in grouped_data.items():
            find_quary = False
            for quary2 in keys[0]:
                if quary2 == query1:
                    find_quary = True
                    break
            
            if not find_quary:
                keys[0].append(query1)
                keys[1].append(2)


        for query, keys in grouped_data.items():
            query1 = query
            key = keys[0]
            label = keys[1]
            for query2, label in zip(key, label):
                processed_samples.append({
                    "text_a": query1,
                    "text_b": query2,
                    "label": label,
                })


        return processed_samples

    def preprocess_test(samples):
        return [{
            "id": sample["id"],
            "text_a": sample["query1"],
            "text_b": sample["query2"],
            "label": -1,
        } for sample in samples]

    def load_and_preprocess(file_path, type_="train"):
        data = read_file(file_path)
        if type_ == "train":
            return preprocess_train(data)
        elif type_ == "valid":
            return preprocess_valid(data)
        else:
            return preprocess_test(data)


    return {
        "train": load_and_preprocess(train_path, type_="train"),
        "dev": load_and_preprocess(dev_path, type_="valid"),
        "test": load_and_preprocess(test_path, type_="test")
    }

class KUAKEQQR_Dataset(Dataset):
    _id2label = ["0", "1", "2"]
    _label2id = {label: i for i, label in enumerate(_id2label)}
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.data[idx]["text_a"], self.data[idx]["text_b"], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": torch.tensor(self.data[idx]["label"])
        }
    
    @classmethod
    def id2label(cls, _id):
        return cls._id2label[_id]
    
    @classmethod
    def label2id(cls, label):
        return cls._label2id[label]


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


def train(model, data, optimizer, scheduler, criterion, device):
    model.train()

    optimizer.zero_grad()
    data = prepare_input(data, device)
    
    logits = model(data["input_ids"], data["attention_mask"])

    loss = criterion(logits, data["labels"])
    correct = (logits.argmax(dim=-1) == data["labels"]).sum().item()
    correct = correct / data["labels"].shape[0]

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
                data = prepare_input(data, device)
                logits = model(data["input_ids"], data["attention_mask"])
                labels = data["labels"]
                loss = criterion(logits, labels).item()
                correct = (torch.argmax(logits, dim=1) == labels).sum().item()
                eval_correct += correct/len(labels)
                eval_loss += loss   
                pbar.update(1)
                
    return eval_loss/len(dataloader), eval_correct/len(dataloader)

def predict(
    args: TrainingArguments,
    model: nn.Module,
    test_dataloader
):
    model.eval()
    preds_list = []
    with torch.no_grad():
        with tqdm(test_dataloader, desc="Predicting", leave=False) as pbar:
            for item in test_dataloader:
                inputs = prepare_input(item, device=args.device)
                logits = model(inputs["input_ids"], inputs["attention_mask"])

                preds = torch.argmax(logits.cpu(), dim=-1).numpy()
                preds_list.append(preds)
                pbar.update(1)

    print(f'Prediction Finished!')
    preds = np.concatenate(preds_list, axis=0).tolist()

    return preds

def generate_commit(output_dir, task_name, test_dataset, preds: List[int]):

    pred_test_examples = []
    for idx in range(len(test_dataset)):
        example = test_dataset[idx]
        label  = KUAKEQQR_Dataset.id2label(preds[idx])
        pred_example = {'id': example['id'], 'query1': example['text_a'], 'query2': example['text_b'], 'label': label}
        pred_test_examples.append(pred_example)
    
    with open(os.path.join(output_dir, f'{task_name}_test.json'), 'w', encoding='utf-8') as f:
        json.dump(pred_test_examples, f, indent=2, ensure_ascii=False)

def train_model(model, train_loader, valid_loader, train_args, tokenizer, writer=None):
    # 定义损失函数和优化器
    criterion = model.criterion()
    optimizer, scheduler = create_optimizer_and_scheduler(train_args, model, len(train_loader) * train_args.num_train_epochs)
    # 开始训练
    best_val_acc = 0.0
    val_loss = 0
    val_acc = 0
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

            for batch in train_loader:
                global_steps += 1
                
                train_loss, train_acc = train(model, batch, optimizer, scheduler, criterion, train_args.device)
                epoch_loss += train_loss
                epoch_correct += train_acc
                epoch_total += 1
                
                if (global_steps+1) % train_args.eval_steps == 0:
                    
                    val_loss, val_acc = evaluate(model, valid_loader, criterion, train_args.device)
                    if writer:
                        writer.add_scalar("Loss/train", epoch_loss / epoch_total, global_steps)
                        writer.add_scalar("Accuracy/train", epoch_correct / epoch_total, global_steps)
                        writer.add_scalar("Loss/eval", val_loss, global_steps)
                        writer.add_scalar("Accuracy/eval", val_acc, global_steps)
                        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], global_steps)

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

def main(args):
    set_seed(42)

    data_args = DataTrainingArguments(data_dir=args.data_dir,
                            model_dir=args.model_dir,
                            argument=args.argument)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        tolerance=args.tolerance,
        scheduler=args.scheduler
    )

    data_args.data_dir = os.path.join(data_args.data_dir, data_args.task_name)
    time_str = time.strftime("%Y%m%d-%H%M", time.localtime())
    if args.tag:
        time_str = f"{args.tag}-{time_str}"
    train_args.output_dir = os.path.join(train_args.output_dir, 'baseline', time_str)

    writer = SummaryWriter(log_dir=train_args.output_dir)


    data = load_data(data_args.data_dir, data_args.task_name, data_args.argument)   

    tokenizer = AutoTokenizer.from_pretrained(data_args.model_dir)

    train_dataset = KUAKEQQR_Dataset(data["train"], tokenizer)
    dev_dataset = KUAKEQQR_Dataset(data["dev"], tokenizer)
    test_dataset = KUAKEQQR_Dataset(data["test"], tokenizer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_args.train_batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=train_args.eval_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=train_args.eval_batch_size, shuffle=False)

    model = Bert(data_args.model_dir, data_args.labels).to(train_args.device)

    print("Start training...")
    best_val_acc, best_steps = train_model(model, train_loader, dev_loader, train_args, tokenizer, writer)

    writer.add_scalar("Best Accuracy", best_val_acc, best_steps)
    writer.close()
    print(f"Best Accuracy: {best_val_acc} at epoch {best_steps}")

    best_state_dict = torch.load(os.path.join(train_args.output_dir, f"checkpoint-{best_steps}", "model.pth"), weights_only=True)

    model.load_state_dict(best_state_dict)

    torch.save(model.state_dict(), os.path.join(train_args.output_dir, "model.pth"))

    tokenizer.save_pretrained(save_directory=train_args.output_dir)
    print(f"Final model and tokenizer are saved to {train_args.output_dir}")

    preds = predict(train_args, model, test_loader)
    generate_commit(train_args.output_dir, data_args.task_name, data["test"], preds)
    print(f"Prediction is saved to {train_args.output_dir}")

if __name__ == "__main__":
    args = args_parser()
    main(args)


