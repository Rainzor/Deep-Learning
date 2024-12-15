from transformers import AutoTokenizer
import os
import time 
import json
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from models.utils import (set_seed, 
                        args_parser, 
                        create_optimizer_and_scheduler,
                        DataTrainingArguments,
                        TrainingArguments
                        )
from models.dataset import *
from models.model import ContrastiveModel


def load_data(data_dir, task_name, aug = False):
    dir_path = os.path.join(data_dir, task_name)
    train_path = os.path.join(dir_path, f"{task_name}_train.json")
    dev_path = os.path.join(dir_path, f"{task_name}_dev.json")
    test_path = os.path.join(dir_path, f"{task_name}_test.json")

    def read_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def preprocess_valid(samples):
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
        
        processed_samples = [{
            "query": query1,
            "keys": keys[0],
            "label": keys[1]
        } for query1, keys in grouped_data.items()]
        return processed_samples

    def preprocess_train(samples):
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
        if aug:
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
                        new_grouped_data[key2[i]][0].append(query)
                        new_grouped_data[key2[i]][1].append(2)
                        for j in range(i+1, len(key2)):
                            new_grouped_data[key2[i]][0].append(key2[j])
                            new_grouped_data[key2[i]][1].append(2)

                            new_grouped_data[key2[j]][0].append(key2[i])
                            new_grouped_data[key2[j]][1].append(2)
                        
                        for j in range(len(key1)):
                            new_grouped_data[key2[i]][0].append(key1[j])
                            new_grouped_data[key2[i]][1].append(1)

                            new_grouped_data[key1[j]][0].append(key2[i])
                            new_grouped_data[key1[j]][1].append(0)
                        
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

    def load_and_preprocess(file_path, type_='train'):
        data = read_file(file_path)
        if type_ == 'test':
            return preprocess_test(data)
        elif type_ == 'valid':
            return preprocess_train(data)
        else:
            return preprocess_train(data)


    return {
        "train": load_and_preprocess(train_path, type_='train'),
        "valid": load_and_preprocess(dev_path, type_='valid'),
        "test": load_and_preprocess(test_path, type_='test')
    }

def train(model, data, optimizer, scheduler, device):
    model.train()

    optimizer.zero_grad()
    data = data.to(device)
    labels = data.labels

    # 前向传播
    logits = model(data)
    nums = 0
    loss = model.criterion(data, logits)
    correct = 0
    # correct = (torch.argmax(logits, dim=1) == labels).sum().item()
    # correct /= len(labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()
    return loss.item(), correct

# # 定义评估函数
def evaluate(model, dataloader, device):
    model.eval()
    eval_correct = 0
    eval_loss = 0.0
    with torch.no_grad():
        with tqdm(dataloader, desc="Evaluation", leave=False) as pbar:
            for data in dataloader:
                data = data.to(device)
                labels = data.labels
                # 前向传播
                logits = model(data)
                loss = model.criterion(data, logits).item()
                # correct = (torch.argmax(logits, dim=1) == labels).sum().item()
                # eval_correct += correct/len(labels)
                eval_loss += loss 
                pbar.update(1)
                
    return eval_loss/len(dataloader), eval_correct/len(dataloader)

def train_model(model, train_loader, valid_loader, train_args, tokenizer, writer):
    # 定义损失函数和优化器
    optimizer, scheduler = create_optimizer_and_scheduler(train_args, model, len(train_loader) * train_args.num_train_epochs)
    # 开始训练
    val_acc = 0
    val_loss = 0
    best_loss = 1e8
    best_steps = -1
    log_history = {
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": []
        }
    val_loss, val_acc = evaluate(model, valid_loader, train_args.device)
    # print(f"Initial eval loss: {val_loss})")

    with tqdm(range(train_args.num_train_epochs* len(train_loader)), desc="Epochs") as epochs_pbar:
        global_steps = 0
        for epoch in range(train_args.num_train_epochs):
            epoch_loss= 0
            epoch_correct = 0
            epoch_total = 0

            for batch in train_loader:
                global_steps += 1
                
                train_loss, train_acc = train(model, batch, optimizer, scheduler, train_args.device)
                epoch_loss = epoch_loss + train_loss
                epoch_correct = epoch_correct + train_acc
                epoch_total += 1
                batch_loss = epoch_loss/epoch_total
                if (global_steps+1) % len(train_loader) == 0:
                    
                    val_loss, val_acc = evaluate(model, valid_loader, train_args.device)

                    writer.add_scalar("Loss/train", batch_loss, global_steps)
                    # writer.add_scalar("Accuracy/train", epoch_correct / epoch_total, global_steps)
                    writer.add_scalar("Loss/eval", val_loss, global_steps)
                    # writer.add_scalar("Accuracy/eval", val_acc, global_steps)

                    writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], global_steps)

                    # if best_val_acc- val_acc > train_args.tolerance:
                    #     print(f"Early stop at step {global_steps}")
                    #     return best_val_acc, best_steps

                # 保存最佳模型
                
                if batch_loss < best_loss:
                    best_loss = batch_loss
                    best_steps = epoch
                    tokenizer.save_pretrained(os.path.join(train_args.output_dir, "pretrained"))
                    best_model_state = model.state_dict().copy()

                epochs_pbar.set_postfix({
                    "train loss": batch_loss,
                    # "train acc": epoch_correct/epoch_total,
                    "eval loss": val_loss,
                    # "eval acc": val_acc
                })
                epochs_pbar.update(1)

            if best_steps == epoch:
                os.makedirs(train_args.output_dir, exist_ok=True)
                save_dir = os.path.join(train_args.output_dir, f"checkpoint-{best_steps}")
                os.makedirs(save_dir, exist_ok=True)
                
                torch.save(best_model_state, os.path.join(save_dir, "model.pth"))
                
            print(f"Epoch {global_steps} finished, train loss: {epoch_loss/epoch_total:.4f}, best loss: {best_loss:.4f}")
    return best_loss, best_steps


def main(args):
    print("Model: ", args.model_dir)
    print("Data Directory: ", args.data_dir)
    print("Output Directory: ", args.output_dir)
    print("Task Name: Baseline")
    print("Batch Size: ", args.batch_size)
    print("Learning Rate: ", args.learning_rate)
    if args.checkpoint:
        print("Checkpoint: ", args.checkpoint)
    if args.augment:
        print("Using data augmentation")
    print("Start loading data...")

    data_args = DataTrainingArguments(data_dir=args.data_dir,
                            model_dir=args.model_dir,
                            augment=args.augment,
                            checkpoint=args.checkpoint)
    train_args = TrainingArguments(output_dir=args.output_dir, 
                            num_train_epochs=args.epochs, 
                            train_batch_size=args.batch_size, 
                            learning_rate=args.learning_rate, 
                            weight_decay=args.weight_decay, 
                            warmup_ratio=args.warmup_ratio,
                            tolerance=args.tolerance,
                            scheduler=args.scheduler,
                            logging_steps=args.log_steps,
                            eval_steps=args.log_steps
                            )

    timename = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    model_name = 'SimCSE'
    if args.tag:
        timename = f"{args.tag}-{timename}"
    train_args.output_dir = os.path.join(train_args.output_dir, model_name, timename)

    writer = SummaryWriter(log_dir=train_args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(data_args.model_dir)

    rawdata = load_data(data_args.data_dir, data_args.task_name, data_args.augment)

    train_dataset = KUAKE_Dataset(rawdata["train"], tokenizer, max_length=data_args.max_length, type_='train')
    valid_dataset = KUAKE_Dataset(rawdata["valid"], tokenizer, max_length=data_args.max_length, type_='valid')
    test_dataset = KUAKE_Dataset(rawdata["test"], tokenizer, max_length=data_args.max_length, type_='test')

    # 使用自定义的 collate_fn 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=train_args.train_batch_size, shuffle=True,             
                            collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=train_args.train_batch_size, shuffle=False,          
                             collate_fn=custom_collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
    #                         collate_fn=custom_collate_fn)

    model = ContrastiveModel(data_args.model_dir, data_args.labels)
    model.to(train_args.device)

    if data_args.checkpoint:
        checkpoint_file = os.path.join(data_args.checkpoint, "model.pth")
        model.load_state_dict(torch.load(checkpoint_file, map_location=train_args.device, weights_only=True))
        print(f"Load model from {data_args.checkpoint}")


    print("Start training...")
    best_loss, best_steps = train_model(model, train_loader, valid_loader, train_args, tokenizer, writer)

    writer.add_scalar("Best Loss", best_loss, best_steps)
    writer.close()
    print(f'Training Finished! Best step - {best_steps} - Best Loss - {best_loss}')

    torch.save(model.state_dict(), os.path.join(train_args.output_dir, "model_final.pth"))
    
    best_state_dict = torch.load(os.path.join(train_args.output_dir, f"checkpoint-{best_steps}", "model.pth"), weights_only=True)
    model.load_state_dict(best_state_dict)

    pretrained_dir = os.path.join(train_args.output_dir, "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)
    model.encoder.save_pretrained(save_directory=pretrained_dir)
    torch.save(model.state_dict(), os.path.join(train_args.output_dir, "model.pth"))

    tokenizer.save_pretrained(save_directory=pretrained_dir)
    print(f"Final model and tokenizer are saved to {train_args.output_dir}")

if __name__ == "__main__":
    args = args_parser()
    set_seed(42)
    main(args)