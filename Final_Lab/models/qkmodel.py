import torch
from torch import nn
from transformers import AutoModel
from collections import defaultdict
import os
import numpy as np


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
        # query_cls = query_outputs.last_hidden_state[:, 0, :]  # CLS token shape: (batch_size, hidden_size)
        query_cls = query_outputs.pooler_output  # shape: (batch_size, hidden_size)
        query_cls = query_cls.unsqueeze(1).expand(-1, N, -1)  # shape: (batch_size, max_num_keys, hidden_size)
        query_cls = query_cls[masks] # shape: (key_num, hidden_size)

        # Key Encoder
        key_outputs = self.encoder(input_ids=key_input_ids, attention_mask=key_attention_mask) # shape: (key_num, seq_len, hidden_size)
        # key_cls = key_outputs.last_hidden_state[:, 0, :]  # CLS token shape: (key_num, hidden_size)
        key_cls = key_outputs.pooler_output  # shape: (key_num, hidden_size)
        combined = torch.cat([query_cls, key_cls], dim=-1)  # shape: (key_num, hidden_size * 2)

        # 分类
        logits = self.classifier(combined)  # shape: (key_num, num_labels)

        return logits
    
    def criterion(self):
        return nn.CrossEntropyLoss()
