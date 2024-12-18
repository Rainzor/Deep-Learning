import torch
import torch.nn as nn
from torch.nn import Parameter
from .RNN import RNN
from .GRU import GRU
from .LSTM import LSTM
from .Transformer import TransformerEncoder, PositionalEncoding

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .utils import RNNConfig, TransformerConfig

class CustomRNNClassifier(nn.Module):
    def __init__(self, config: RNNConfig):
        super(CustomRNNClassifier, self).__init__()

        self.bidirectional = config.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_dim = (config.hidden_dim // self.num_directions) * self.num_directions

        self.model_type = config.name.lower()
        if self.model_type == 'rnn':
            self.encoder = RNN(
                input_size=config.embedding_dim,
                hidden_size=self.hidden_dim // self.num_directions,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True
            )
        elif self.model_type == 'gru':
            self.encoder = GRU(
                input_size=config.embedding_dim,
                hidden_size=self.hidden_dim // self.num_directions,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True
            )
        elif self.model_type == 'lstm':
            self.encoder = LSTM(
                input_size=config.embedding_dim,
                hidden_size=self.hidden_dim // self.num_directions,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Only support 'rnn', 'gru', and 'lstm', got '{config.name}'")
        
        self.decoder = nn.Linear(self.hidden_dim, config.output_dim)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.size()

        # Embedding
        embedded = self.embedding(input_ids)  # Shape: [batch_size, seq_len, embedding_dim]

        if self.model_type == 'lstm':
            output, (hidden, cell) = self.encoder(embedded)
        else:
            output, hidden = self.encoder(embedded)
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long()  # Shape: [batch_size]
            last_indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
            pooled = output.gather(1, last_indices).squeeze(1)
        else:
            pooled = output[:, -1, :]
        
        logits = self.decoder(pooled)  # Shape: [batch_size, output_dim]

        return logits


class RNNClassifier(nn.Module):
    def __init__(self, config: RNNConfig):
        super(RNNClassifier, self).__init__()

        self.bidirectional = config.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_dim = (config.hidden_dim // self.num_directions) * self.num_directions

        self.model_type = config.name.lower()
        self.pooling = config.pool.lower()
        self.pack = config.pack
        if self.pooling == 'cls':
            self.pooling = 'last'
        assert self.pooling in ['mean', 'max', 'last','attention'], "Pooling must be either 'mean', 'max', 'last' or 'attention'"

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)

        # Encoder: Choose between RNN, GRU, and LSTM
        if self.model_type == 'rnn':
            self.encoder = nn.RNN(
                input_size=config.embedding_dim,
                hidden_size=self.hidden_dim // self.num_directions,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True
            )
        elif self.model_type == 'gru':
            self.encoder = nn.GRU(
                input_size=config.embedding_dim,
                hidden_size=self.hidden_dim // self.num_directions,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True
            )
        elif self.model_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=self.hidden_dim // self.num_directions,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Only support 'rnn', 'gru', and 'lstm', got '{config.name}'")

        self.dropout = nn.Dropout(config.dropout)


        if self.pooling == 'attention':
            self.attention = nn.Linear(self.hidden_dim, 1, bias=False)

        self.decoder = nn.Linear(self.hidden_dim, config.output_dim)

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

        # Sort input by length for pack_padded_sequence
        if attention_mask is not None and self.pack:
            lengths = attention_mask.sum(dim=1).long()  # Shape: [batch_size]
            lengths_sorted, sorted_idx = lengths.sort(0, descending=True)
            embedded_sorted = embedded[sorted_idx]
            attention_mask_sorted = attention_mask[sorted_idx]
            # Pack the sequences
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
        else:
            # If no attention_mask is provided, assume all sequences are of equal length
            attention_mask_sorted = attention_mask
            packed_embedded = embedded

        # Pass through RNN/GRU/LSTM
        if self.model_type == 'lstm':
            packed_output, (hidden, cell) = self.encoder(packed_embedded)
        else:
            packed_output, hidden = self.encoder(packed_embedded)

        # Unpack the output
        if attention_mask is not None and self.pack:
            output_unpacked, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=input_ids.size(1)
            )  # Shape: [batch_size_sorted, seq_len, num_directions * hidden_size]
        else:
            output_unpacked = packed_output  # Shape: [batch_size, seq_len, num_directions * hidden_size]

        pooled = self._pooling(output_unpacked, attention_mask_sorted)  # Shape: [batch_size_sorted, hidden_dim]

        # Apply dropout
        pooled = self.dropout(pooled)  # Shape: [batch_size_sorted, hidden_dim * num_directions]

        # Pass through the decoder
        output = self.decoder(pooled)  # Shape: [batch_size_sorted, output_dim]

        if attention_mask is not None and self.pack:
            # Unsort to the original order
            _, unsort_idx = sorted_idx.sort(0)
            output = output[unsort_idx]

        return output
    
    def _pooling(self, output, mask = None):
        """
        Pooling the output of the RNN/GRU/LSTM.

        Args:
            output (Tensor): Output tensor of shape [batch_size, seq_len, num_directions * hidden_size]
            mask (Tensor, optional): Mask tensor of shape [batch_size, seq_len, 1]
        
        Returns:
            Tensor: Pooled output tensor of shape [batch_size, hidden_dim]
        """
        

        B, L, _ = output.size()
        lengths = mask.sum(dim=1).long()  if mask is not None else output.new_full((B,), L, dtype=torch.long)    # Shape: [batch_size, 1]
        lengths = lengths.view(-1) # Shape: [batch_size]

        if self.pooling == 'last':
            if mask is not None:
                last_indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim) # Shape: [batch_size, 1, hidden_dim]
                pooled = output.gather(1, last_indices).squeeze(1)
            else:
                pooled = output[:, -1, :]
        elif self.pooling == 'mean':
            # Compute the sum over the valid time steps
            if mask is not None:
                sum_output = torch.sum(output * mask, dim=1)  # Shape: [batch_size_sorted, hidden_dim]
                pooled = sum_output / lengths.unsqueeze(1)  # Shape: [batch_size_sorted, hidden_dim]
            else:
                pooled = torch.mean(output, dim=1)
        elif self.pooling == 'max':
            '''
            Recurrent Convolutional Neural Network (RCNN) for text classification.
            https://dl.acm.org/doi/10.5555/2886521.2886636
            '''
            if mask is not None:
                output = output.masked_fill(mask == 0, -1e9)
            pooled, _ = torch.max(output, dim=1)
        elif self.pooling == 'attention':
            """
            Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
            https://ieeexplore.ieee.org/document/9251129/
            """
            M = F.tanh(output)
            scores = self.attention(M)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9).squeeze(-1)
            alpha = F.softmax(scores,dim=1).unsqueeze(2)
            pooled = torch.sum(output*alpha,dim=1)
            pooled = F.tanh(pooled)
        
        return pooled


class RCNNClassifier(nn.Module):

    def __init__(self, config: RNNConfig):
        super(RCNNClassifier, self).__init__()

        self.hidden_dim = (config.hidden_dim//2)*2
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.encoder = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=self.hidden_dim//2,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        self.proj = nn.Linear(config.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(self.hidden_dim, config.output_dim)
    
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        output, _ = self.encoder(embedded)
        x = torch.cat((embedded, output), dim=2) # Shape: [batch_size, seq_len, embedding_dim + hidden_dim]        
        x = self.activation(self.proj(x)).transpose(1, 2) # Shape: [batch_size, hidden_dim, seq_len]

        # Max pooling 
        pooled_output = F.max_pool1d(x, x.size(2)).squeeze(2) # Shape: [batch_size, hidden_dim]

        logits = self.decoder(pooled_output)  # Shape: [batch_size, output_dim]

        return logits

class RNNAttentionClassifier(nn.Module):

    def __init__(self,config:RNNConfig):
        super(RNNAttentionClassifier,self).__init__()

        self.hidden_dim = (config.hidden_dim//2)*2

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.encoder = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=self.hidden_dim//2,
            num_layers=config.n_layers,
            dropout=config.dropout if config.n_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        self.W = nn.Parameter(torch.randn((self.hidden_dim)))
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.decoder = nn.Linear(self.hidden_dim, config.output_dim)
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding layer
        embedded = self.embedding(input_ids)

        # LSTM encoder
        output, _ = self.encoder(embedded)

        # Attention mechanism
        M = self.tanh1(output)
        alpha = F.softmax(torch.matmul(M,self.W),dim=1).unsqueeze(2) # Shape: [batch_size, seq_len, 1]

        r = torch.sum(output*alpha,dim=1) # Shape: [batch_size, hidden_dim]
        r = self.tanh2(r)

        # Classification
        logits = self.decoder(r)  # Shape: [batch_size, output_dim]

        return logits

class TransformerClassifier(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerClassifier, self).__init__()

        self.pooling = config.pool.lower()
        if self.pooling == 'last':
            self.pooling = 'cls'
        if self.pooling not in ['cls', 'mean', 'max']:
            raise ValueError(f"Only support 'cls', 'mean' and 'max', got '{config.pool}'")

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_encoder = PositionalEncoding(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.proj = nn.Linear(config.embedding_dim, config.d_model)

        self.encoder = TransformerEncoder(
            num_layers=config.n_layers,
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )

        self.decoder = nn.Linear(config.d_model, config.output_dim)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)

        x = self.proj(embedded.transpose(0, 1)).transpose(0, 1)

        output = self.encoder(x) # Shape: [batch_size, seq_len, d_model]

        if self.pooling == 'cls':
            cls_output = output[:, 0, :]  # [batch_size, d_model]
            pooled = cls_output
        elif self.pooling == 'mean':
            pooled = output.mean(dim=1)    # [batch_size, d_model]
        elif self.pooling == 'max':
            pooled, _ = output.max(dim=1)  # [batch_size, d_model]

        logits = self.decoder(pooled)  # Shape: [batch_size, output_dim]

        return logits
    