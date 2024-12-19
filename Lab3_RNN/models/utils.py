from dataclasses import dataclass,field
import argparse

DATA_DIR = '../data/Yelp'
OUTPUT_DIR = './output'
CHECKPOINT_DIR = None
MODEL = 'rnn'
EPOCHS = 1
LABEL_NUM = 5
LEARNING_RATE = 5e-4
BATCH_SIZE = 256
OPTIMIZER = 'adam'
SCHEDULER = 'cosine'
MAX_LENGTH = 256
DROP_RATE = 0.3
LAYER_NUM = 2
EMBEDDING_DIM = 256
HIDDEN_DIM = 512


@dataclass
class TrainConfig:
    data_path: str = field(default=DATA_DIR)
    model: str = field(default=MODEL)
    output_path: str = field(default=OUTPUT_DIR)
    checkpoint_path: str = field(default= None)
    epochs: int = field(default=EPOCHS)
    total_steps: int = field(default=0)
    learning_rate: float = field(default=LEARNING_RATE)
    batch_size: int = field(default=BATCH_SIZE)
    optimizer: str = field(default=OPTIMIZER)
    scheduler: str = field(default=SCHEDULER)
    num_cycles: int = 0.5
    min_lr: float = 0
    warmup_ratio: float = 0.1
    weight_decay: float = 0

@dataclass
class RNNConfig:
    name: str = 'rnn'
    embedding_dim: int = EMBEDDING_DIM
    hidden_dim: int = HIDDEN_DIM
    output_dim: int = LABEL_NUM
    n_layers: int = LAYER_NUM
    bidirectional: bool = False
    dropout: float = DROP_RATE
    vocab_size: int = 0
    pool: str = 'last'
    pack: bool = False

@dataclass
class TransformerConfig(RNNConfig):
    n_heads: int = 8
    dim_feedforward: int = 256
    pool: str = 'cls'
    

def parse_args():
    parser = argparse.ArgumentParser(description="Text Classification with RNN")

    parser.add_argument('--data_path', '-d', type=str, default=DATA_DIR, help="Directory for data")
    parser.add_argument('--model', '-m', type=str, default=MODEL, help="Model for training")
    parser.add_argument('--output_path', '-o', type=str, default=OUTPUT_DIR, help="Directory for saving outputs")
    parser.add_argument("--checkpoint", "-ckp", type=str, help="Directory for saving checkpoints")
    parser.add_argument('--pretrained', type=str, help="Pretrained model for training")
    
    parser.add_argument('--tag', '-t', type=str, help="Tag for model")
    parser.add_argument('--val_cki', type=float, default=0.5, help="Validation checkpoint interval")
    parser.add_argument('--log_step', type=int, default=20, help="Log interval")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    
    parser.add_argument('--seed',type=int, default=42, help="Seed for reproducibility")

    parser.add_argument('--epochs','-n', type=int, default=10, help="Number of epochs")
    parser.add_argument('--learning_rate','-lr' ,type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--optimizer', '-opt', type=str, default=OPTIMIZER, help="Optimizer for training")
    parser.add_argument('--scheduler', type=str, default=SCHEDULER, help="Scheduler for training")
    parser.add_argument('--num_cycles', type=int, default=1, help="Number of cycles for scheduler")
    parser.add_argument('--min_lr', type=float, default=0, help="Minimum learning rate for scheduler")
    parser.add_argument('--warmup_ratio', '-wr', type=float, default=0.1, help="Warmup ratio for scheduler")
    parser.add_argument('--weight_decay', '-wd', type=float, default=0, help="Weight decay for optimizer")

    parser.add_argument('--max_length', '-len', type=int, default=MAX_LENGTH, help="Maximum length of input sequence")
    parser.add_argument('--bidirectional', '-bi', action='store_true', help="Use bidirectional RNN")
    parser.add_argument('--n_layers', '-nl', type=int, default=LAYER_NUM, help="Number of RNN layers")
    parser.add_argument('--dropout', '-do', type=float, default=DROP_RATE, help="Dropout rate")
    parser.add_argument('--embedding_dim', '-ed', type=int, default=EMBEDDING_DIM, help="Embedding dimension")
    parser.add_argument('--hidden_dim', '-hd', type=int, default=HIDDEN_DIM, help="Hidden dimension")
    parser.add_argument('--n_heads', '-nh', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--pool', type=str, default='last', help="Pooling method for Classification", choices=['last', 'max', 'mean', 'cls', 'attention'])
    parser.add_argument('--pack', '-pk', action='store_true', help="Use pack_padded_sequence")


    # Parse arguments
    args = parser.parse_args()
    return args


rnn_config = RNNConfig(
    name='rnn',
    bidirectional=False 
)

gru_config = RNNConfig(
    name='gru',
    bidirectional=False 
)

# -n 100 -b 512 -lr 1e-3
lstm_config = RNNConfig(
    name='lstm',
    bidirectional=False
)

rcnn_config = RNNConfig(
    name='rcnn',
    bidirectional=True,
    pool='max'
)

rnn_attention_config = RNNConfig(
    name='rnn_attention',
    bidirectional=True,
    pool='attention'
)

transformer_config = TransformerConfig(
    name='transformer',
    embedding_dim=EMBEDDING_DIM,
    output_dim=LABEL_NUM,
    hidden_dim=HIDDEN_DIM,
    n_layers=LAYER_NUM,
    dropout=DROP_RATE,
    n_heads=8,
    dim_feedforward=HIDDEN_DIM,
    pool='cls'
)
