import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification
from collections import defaultdict
import os
import numpy as np



class LLM(nn.Module):
    def __init__(self, model_dir, num_labels):
        super(LLM, self).__init__()
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

    def forward(self, data):
        input_ids = data.input_ids # [num_keys, max_length]
        attention_mask = data.attention_mask # [num_keys, max_length]

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask) # [num_keys, max_length, hidden_size]
        pooled_output = outputs.pooler_output # [num_keys, hidden_size]
        logits = self.classifier(pooled_output) # [num_keys, num_labels]

        return pooled_output, logits
    
    def criterion(self, inputs, outputs):
        labels = inputs.labels # [num_keys]
        batch = inputs.batch # [num_keys]
        batch_num = batch.max().item() + 1

        pooled_output, logits = outputs
        contract_loss = 0

        pooled_output = F.normalize(pooled_output, p=2, dim=-1)

        temperature = 0.05
        ratio = torch.zeros(batch_num).to(pooled_output.device)
        for i in range(batch_num):
            mask2 = (batch == i) & (labels == 2)
            mask1 = (batch == i) & (labels == 1)
            mask0 = torch.logical_not(mask2 | mask1)
            if mask2.sum() == 0:
                continue

            value2 = pooled_output[mask2] # [num2, hidden_size]
            value1 = pooled_output[mask1] # [num1, hidden_size]
            value0 = pooled_output[mask0] # [num0, hidden_size]

            sim0 = torch.sum(value1.unsqueeze(0) * value2.unsqueeze(1), dim=-1)/temperature  # [num1, num2]
            sim1 = torch.sum(value0.unsqueeze(0) * value2.unsqueeze(1), dim=-1)/temperature  # [num0, num2]

            score1 = torch.clamp_min(torch.sum(torch.exp(sim0), dim=-1), 1e-9) # [num2]
            score0 = torch.clamp_min(torch.sum(torch.exp(sim1), dim=-1), 1e-9) # [num2]
            probs = score1/(score0) # [num2]
            ratio[i] = torch.clamp(probs.mean(), 1e-9, 1-1e-9)
        
        contract_loss = -torch.log(ratio).mean()

        return F.cross_entropy(logits, inputs.labels), contract_loss