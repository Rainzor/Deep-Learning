import torch
from torch import nn
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
    
    def criterion(self):
        return nn.CrossEntropyLoss()