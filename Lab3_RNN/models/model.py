import torch
import torch.nn as nn
from torch.nn import Parameter
from .RNN import RNN

import torch
import torch.nn as nn
from dataclasses import dataclass
from .utils import ModelConfig

class TextClassifyRNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TextClassifyRNN, self).__init__()  # Corrected class name

        self.bidirectional = config.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.model_type = config.name.lower()

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Encoder: Choose between RNN and LSTM
        if self.model_type == 'rnn':
            self.encoder = nn.RNN(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True
            )
        elif self.model_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Only support 'rnn' and 'lstm', got '{config.name}'")

        self.dropout = nn.Dropout(config.dropout)

        # Classifier: Two linear layers with ReLU activation
        hidden_dim = config.hidden_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.output_dim)
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_ids (Tensor): Input tensor of shape [batch_size, seq_len]
            attention_mask (Tensor, optional): Mask tensor of shape [batch_size, seq_len]
                                               where 1 indicates real tokens and 0 indicates padding.
                                               Defaults to None.

        Returns:
            Tensor: Output tensor of shape [batch_size, output_dim]
        """
        # Embedding
        embedded = self.embedding(input_ids)  # Shape: [batch_size, seq_len, embedding_dim]

        if self.model_type == 'lstm':
            output, (hidden, cell) = self.encoder(embedded)
        else:
            output, hidden = self.encoder(embedded)

        # Process the hidden state for classification
        if self.bidirectional:
            # For bidirectional models, concatenate the last hidden states from both directions
            # hidden shape: (num_layers * 2, batch, hidden_dim)
            pooled_output = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # Shape: [batch_size, hidden_dim * 2]
        else:
            # For unidirectional models, take the last hidden state
            # hidden shape: (num_layers, batch, hidden_dim)
            pooled_output = hidden[-1, :, :]  # Shape: [batch_size, hidden_dim]

        # Dropout
        pooled_output = self.dropout(pooled_output)

        # Classification
        output = self.classifier(pooled_output)  # Shape: [batch_size, output_dim]
        return output