"""TrendMaster: Stock Price Prediction using Transformer Deep Learning.

Main package exports for backward compatibility.
"""

# Import from modular structure
from trendmaster.data import DataLoader
from trendmaster.models import TransAm, PositionalEncoding
from trendmaster.training import Trainer, plot_results
from trendmaster.inference import Inferencer, plot_predictions
from trendmaster.utils import logger

# Re-export for backward compatibility
__all__ = [
    'DataLoader',
    'TransAm',
    'PositionalEncoding',
    'Trainer',
    'Inferencer',
    'set_seed',
    'plot_results',
    'plot_predictions',
    'logger',
]

__version__ = '0.3.0'


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
