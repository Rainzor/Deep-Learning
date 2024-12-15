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
        # self.classifier = nn.Linear(hidden_size, num_labels)


    def forward(self, data):
        """
        Parameters
        ----------
        
        """
        input_ids = data.input_ids # [num_keys, max_length]
        attention_mask = data.attention_mask # [num_keys, max_length]

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask) # [num_keys, max_length, hidden_size]
        pooled_output = outputs.pooler_output # [num_keys, hidden_size]
        logits = self.classifier(pooled_output) # [num_keys, num_labels]

        return logits
    
    def criterion(self, inputs, outputs):
        labels = inputs.labels # [num_keys]
        batch = inputs.batch # [num_keys]
        batch_num = batch.max().item() + 1


        contract_loss = 0
        for i in range(batch_num):
            mask = (batch == i)
            logit = outputs[mask]
            log_prob = F.log_softmax(logit, dim=0)
            label = labels[mask].view(-1,1)
            log_prob = log_prob.gather(1, label).view(-1)
            contract_loss += -log_prob.mean()
        
        contract_loss /= batch_num

        return F.cross_entropy(outputs, inputs.labels), contract_loss


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

    def criterion(self, inputs, outputs, type_="train"):
        outputs = F.tanh(outputs)
        exp_score = torch.exp(outputs/self.temperature) # [num_keys]
        labels = inputs.labels
        batch = inputs.batch
        # 创建标签掩码
        batch_num = batch.max().item() + 1
        ratio = torch.zeros(batch_num)
        for i in range(batch_num):
            mask2 = (batch == i) & (labels == 2)
            mask1 = (batch == i) & (labels == 1)
            mask0 = (batch == i) & (labels == 0)
            score2 = exp_score[mask2].sum()
            score1 = exp_score[mask1].sum()
            score0 = exp_score[mask0].sum()
            ratio[i] = (score2 + 1e-9) / (score1 + score0 + 1e-9) + (score1 + 1e-9) / (score0 + 1e-9)
            # ratio[i] = (score2 + score1+ 1e-9) / (score1 + score0 + 1e-9)
        result = -torch.log(ratio)

        return result.mean()

    # def criterion(self, inputs, outputs):
    #     exp_scores = torch.exp(outputs / self.temperature)  # [num_keys]
    #     labels = inputs.labels
    #     batch = inputs.batch
        
    #     # 获取 batch 的数量
    #     batch_num = batch.max().item() + 1
        
    #     # 创建掩码
    #     mask2 = (labels == 2)  # 类别2的掩码
    #     mask1 = (labels == 1)  # 类别1的掩码
    #     mask0 = (labels == 0)  # 类别0的掩码

    #     # 将掩码与 batch 索引对齐
    #     mask2_batch = mask2.unsqueeze(1) & (batch.unsqueeze(0) == torch.arange(batch_num).to(batch.device).unsqueeze(1))
    #     mask1_batch = mask1.unsqueeze(1) & (batch.unsqueeze(0) == torch.arange(batch_num).to(batch.device).unsqueeze(1))
    #     mask0_batch = mask0.unsqueeze(1) & (batch.unsqueeze(0) == torch.arange(batch_num).to(batch.device).unsqueeze(1))

    #     # 计算每个类别的得分
    #     score2 = exp_scores[mask2_batch].sum(dim=1)
    #     score1 = exp_scores[mask0_batch].sum(dim=1)
    #     score0 = exp_scores[mask1_batch].sum(dim=1)
        
    #     # 计算 ratio
    #     ratio = (score2 + score1) / (score1 + score0)

    #     # 计算结果并返回
    #     result = -torch.log(ratio)
        
    #     return result.mean()
