import torch
from dataclasses import dataclass
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the accuracy of the model.

    Args:
        output (Tensor): Output predictions.
        labels (Tensor): Ground-truth labels.

    Returns:
        float: Accuracy value.
    """
    preds = output.max(1)[1]
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

@dataclass
class TrainerConfig:
    """
    Configuration class for the trainer.
    """
    dataset: str = 'cora'
    epochs: int = 1000
    lr: float = 0.01
    weight_decay: float = 5e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler: str = None
    num_warmup_steps: int = 0
    num_training_steps: int = 0

@dataclass
class ModelConfig:
    """
    Configuration class for the model.

    Attributes:
        hidden_channels (int): Number of hidden units.
        dropout (float): Dropout rate.
    """
    name: str = 'GCN'
    num_features: int = 0
    num_classes: int = 0
    hidden_channels: int = 32
    dropout: float = 0.5
    num_layers: int = 2
    residual: bool = False

def parse_args():
    """
    Parse the command-line arguments for the trainer.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Directory
    parser = argparse.ArgumentParser(description="GNN Node Classification")
    parser.add_argument('--root','-r', type=str, default='data',
                        help='Root directory for the dataset')
    parser.add_argument('--dataset','-d', type=str, default='cora',
                        help='Dataset name (default: cora)')
    parser.add_argument('--model','-m', type=str, default='GCN',
                        help='Model name (default: GCN)')
    parser.add_argument('--output_path','-o', type=str, default='outputs',
                        help='Output directory for the logs and checkpoints (default: outputs)')

    # Model
    parser.add_argument('--hidden-dim','-hd', type=int, default=64,
                        help='Number of hidden units (default: 64)')
    parser.add_argument('--dropout', '-dp', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--num-layers','-nl', type=int, default=2,
                        help='Number of layers (default: 2)')
    parser.add_argument('--residual','-res', action='store_true',
                        help='Use residual connection')

    # Trainer
    parser.add_argument('--epochs','-n', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--batch-size','-b', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Number of epochs to wait before early stopping (default: 20)')
    parser.add_argument('--learning-rate', '-lr',type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='L2 regularization weight (default: 5e-4)')
    parser.add_argument('--scheduler', type=str, default=None,
                        help='Learning rate scheduler (default: None)')     
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Number of warm-up steps for the learning rate scheduler (default: 0)')

    # Misc
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the training on (default: cuda if available)')
    parser.add_argument('--seed', type=int, default=42,)
    parser.add_argument('--tag', '-tag', type=str, default=None,
                        help='Tag for the experiment (default: None)')
    parser.add_argument('--log_steps', type=int, default=10,
                        help='Log training metrics every n steps (default: 10)')
    parser.add_argument('--visualize', '-v', action='store_true',  help='Visualize the node embeddings')

    return parser.parse_args()


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()