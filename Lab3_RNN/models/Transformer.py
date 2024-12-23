import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Parameters:
            d_model: Embedding dimension
            max_len: Maximum length of the sequence
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Parameters:
            x: Input embeddings, shape [batch_size, seq_len, d_model]
        Returns:
            Embeddings with added positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        Parameters:
            d_model: Embedding dimension
            nhead: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Define linear transformation layers
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Parameters:
            query: Query vectors, shape [batch_size, seq_len, d_model]
            key: Key vectors, shape [batch_size, seq_len, d_model]
            value: Value vectors, shape [batch_size, seq_len, d_model]
            mask: Mask tensor, shape [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        Returns:
            Attention output, shape [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.linear_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # [batch_size, nhead, seq_len, d_k]
        K = self.linear_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)    # [batch_size, nhead, seq_len, d_k]
        V = self.linear_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # [batch_size, nhead, seq_len, d_k]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, nhead, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)  # [batch_size, nhead, seq_len, seq_len]
        attn = self.dropout(attn)
        
        # Weighted sum
        context = torch.matmul(attn, V)  # [batch_size, nhead, seq_len, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len, d_model]
        
        # Final linear transformation
        out = self.linear_out(context)  # [batch_size, seq_len, d_model]
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        """
        Parameters:
            d_model: Embedding dimension
            d_ff: Hidden layer dimension of the feed-forward network
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super(FeedForward, self).__init__()
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'gelu':
            activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        """
        Parameters:
            x: Input, shape [batch_size, seq_len, d_model]
        Returns:
            Output of the feed-forward network, shape [batch_size, seq_len, d_model]
        """
        return self.ff(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation='relu'):
        """
        Parameters:
            d_model: Embedding dimension
            nhead: Number of attention heads
            dim_feedforward: Hidden layer dimension of the feed-forward network
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        """
        Parameters:
            src: Input, shape [batch_size, seq_len, d_model]
            src_mask: Mask tensor, shape [batch_size, seq_len]
        Returns:
            Output of the encoder layer, shape [batch_size, seq_len, d_model]
        """
        # Self-attention sublayer
        B,L,D = src.size()
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            src_mask = src_mask.expand(B, 1, L, L)  # [batch_size, 1, seq_len, seq_len]
        attn_output = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # Feed-forward sublayer
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1, activation='relu'):
        """
        Parameters:
            num_layers: Number of encoder layers
            d_model: Embedding dimension
            nhead: Number of attention heads
            dim_feedforward: Hidden layer dimension of the feed-forward network
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        """
        Parameters:
            src: Input sequence, shape [batch_size, seq_len]
            src_mask: Mask tensor, shape [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        Returns:
            Output of the encoder, shape [batch_size, seq_len, d_model]
        """
 
        for layer in self.layers:
            src = layer(src, src_mask)
        
        return src