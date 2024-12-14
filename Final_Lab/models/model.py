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
        self.bert = AutoModel.from_pretrained(model_dir)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.pooler_output # [num_keys, hidden_size]
        readout = pooled_output

        logits = self.classifier(readout)
        return logits
    
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
        
        exp_scores = torch.exp(outputs/self.temperature) # [num_keys]
        labels = inputs.labels
        batch = inputs.batch
        # 创建标签掩码
        batch_num = batch.max().item() + 1
        ratio = torch.zeros(batch_num)
        for i in range(batch_num):
            mask = (batch == i)
            label = labels[mask]
            exp_score = exp_scores[mask]
            score2 = exp_score[label==2]
            score1 = exp_score[label==1]
            score0 = exp_score[label==0]
            ratio[i] = (score2.sum() + score1.sum()) / (score1.sum() + score0.sum())
        result = -torch.log(ratio)

        return result.mean()
