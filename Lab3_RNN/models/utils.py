from dataclasses import dataclass,field
import argparse

DATA_DIR = '../data/Yelp'
OUTPUT_DIR = './output'
CHECKPOINT_DIR = None
MODEL = 'rnn'
EPOCHS = 1
LEARNING_RATE = 1e-5
BATCH_SIZE = 64
OPTIMIZER = 'adam'
SCHEDULER = 'linear'


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
    warmup_ratio: float = 0.1
    weight_decay: float = 0

@dataclass
class RNNConfig:
    name: str = 'rnn'
    embedding_dim: int = 256
    hidden_dim: int = 256
    output_dim: int = 5
    n_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.1
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
    parser.add_argument("--checkpoint", "-c", type=str, help="Directory for saving checkpoints")
    parser.add_argument('--pretrained', type=str, help="Pretrained model for training")


    parser.add_argument('--epochs','-n', type=int, default=10, help="Number of epochs")
    parser.add_argument('--learning_rate','-lr' ,type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=BATCH_SIZE, help="Batch size")

    parser.add_argument('--optimizer', '-opt', type=str, default=OPTIMIZER, help="Optimizer for training")
    parser.add_argument('--scheduler', '-s', type=str, default=SCHEDULER, help="Scheduler for training")
    parser.add_argument('--warmup_ratio', '-wr', type=float, default=0, help="Warmup ratio for scheduler")
    parser.add_argument('--weight_decay', '-wd', type=float, default=0, help="Weight decay for optimizer")

    parser.add_argument('--max_length', '-len', type=int, default=512, help="Maximum length of input sequence")
    parser.add_argument('--bidirectional', '-bi', action='store_true', help="Use bidirectional RNN")
    parser.add_argument('--n_layers', '-nl', type=int, default=2, help="Number of RNN layers")
    parser.add_argument('--dropout', '-do', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--embedding_dim', '-ed', type=int, default=256, help="Embedding dimension")
    parser.add_argument('--hidden_dim', '-hd', type=int, default=256, help="Hidden dimension")
    parser.add_argument('--n_heads', '-nh', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--pool', type=str, default='last', help="Pooling method for Classification", choices=['last', 'max', 'mean', 'cls', 'attention'])
    parser.add_argument('--pack', '-pk', action='store_true', help="Use pack_padded_sequence")

    parser.add_argument('--tag', '-t', type=str, help="Tag for model")

    # Parse arguments
    args = parser.parse_args()
    return args


rnn_config = RNNConfig(
    name='rnn',
    embedding_dim=128,
    output_dim=5,
    hidden_dim=256,
    n_layers=2,
    dropout=0.1,
    bidirectional=False 
)

gru_config = RNNConfig(
    name='gru',
    embedding_dim=128,
    output_dim=5,
    hidden_dim=256,
    n_layers=2,
    dropout=0.1,
    bidirectional=False 
)

lstm_config = RNNConfig(
    name='lstm',
    embedding_dim=128,
    output_dim=5,
    hidden_dim=256,
    n_layers=2,
    dropout=0.1,
    bidirectional=False 
)

rcnn_config = RNNConfig(
    name='rcnn',
    embedding_dim=128,
    output_dim=5,
    hidden_dim=256,
    n_layers=2,
    dropout=0.1,
    bidirectional=True,
    pool='max'
)

rnn_attention_config = RNNConfig(
    name='rnn_attention',
    embedding_dim=128,
    output_dim=5,
    hidden_dim=256,
    n_layers=2,
    dropout=0.1,
    bidirectional=True,
    pool='attention'
)

transformer_config = TransformerConfig(
    name='transformer',
    embedding_dim=128,
    output_dim=5,
    hidden_dim=256,
    n_layers=2,
    dropout=0.1,
    n_heads=8,
    dim_feedforward=256,
    pool='cls'
)
