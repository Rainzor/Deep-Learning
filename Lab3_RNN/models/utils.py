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
class ModelConfig:
    name = 'rnn'
    embedding_dim: int = 100
    hidden_dim: int = 256
    output_dim: int = 5
    n_blocks: int = 1
    n_layers: int = 2
    bidirectional: bool = False
    residual: bool = False
    dropout: float = 0
    vocab_size: int = 0


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

    parser.add_argument('--max_length', '-l', type=int, default=512, help="Maximum length of input sequence")

    parser.add_argument('--tag', '-t', type=str, help="Tag for model")

    # Parse arguments
    args = parser.parse_args()
    return args


rnn_config = ModelConfig(
    embedding_dim=100,
    output_dim=5,
    hidden_dim=256,
    n_layers=2,
    n_blocks=1,
    dropout=0.5,
    bidirectional=False 
)