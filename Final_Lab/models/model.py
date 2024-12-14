import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification
from collections import defaultdict
import os
import numpy as np


class Bert(nn.Module):
    def __init__(self, model_dir, num_labels):
        super(Bert, self).__init__()
        self.bert = AutoModel.from_pretrained(model_dir, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)
    
    def criterion(self):
        return nn.CrossEntropyLoss()

class QKModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(QKModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, num_labels)
        )
        # self.classifier = nn.Linear(hidden_size, num_labels)


    def forward(self, data):
        """
        Parameters
        ----------
        
        """
        input_ids = data.input_ids # [num_keys, max_length]
        attention_mask = data.attention_mask # [num_keys, max_length]

        outputs = self.encoder(input_ids, attention_mask) # [num_keys, max_length, hidden_size]
        pooled_output = outputs.pooler_output # [num_keys, hidden_size]
        logits = self.classifier(pooled_output) # [num_keys, num_labels]

        return logits
    
    def criterion(self, inputs, outputs):
        return F.cross_entropy(outputs, inputs.labals)


class ContrastiveModel(nn.Module):
    def __init__(self, model_name, num_labels, temperature=0.05):
        super(ContrastiveModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.score = nn.Linear(hidden_size, 1)
        self.temperature = temperature
    
    def forward(self, data):
        input_ids = data.input_ids
        attention_mask = data.attention_mask

        outputs = self.encoder(input_ids, attention_mask)
        pooled_output = outputs.pooler_output # [num_keys, hidden_size]
        score = self.score(pooled_output).squeeze(-1) # [num_keys]
        return score

    def criterion(self, inputs, outputs):
        
        exp_scores = torch.exp(outputs/self.temperature)
        labels = inputs.labels
        batch = inputs.batch
        # 创建标签掩码
        mask0 = labels == 0
        mask1 = labels == 1
        mask2 = labels == 2

        # 计算批次数量
        num_batches = batch.max().item() + 1

        # 按批次和标签分组求和
        sum0 = torch.bincount(batch, weights=exp_scores * mask0.float(), minlength=num_batches)
        sum1 = torch.bincount(batch, weights=exp_scores * mask1.float(), minlength=num_batches)
        sum2 = torch.bincount(batch, weights=exp_scores * mask2.float(), minlength=num_batches)

        # 计算分子和分母
        numerator = sum2 + sum1
        denominator = sum0 + sum1 + 1e-6 # 防止分母为0

        # 计算比值并取对数
        ratio = numerator / denominator
        result = -torch.log(ratio)

        return result.mean()
