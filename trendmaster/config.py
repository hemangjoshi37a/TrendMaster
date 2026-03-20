"""Configuration dataclasses for TrendMaster."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the TransAm model."""

    input_size: int = 1
    d_model: int = 30
    num_layers: int = 2
    nhead: int = 5
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    patience: int = 10
    step_size: int = 1
    gamma: float = 0.95


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    input_window: int = 30
    output_window: int = 10
    train_test_split: float = 0.8
    scaler_range: tuple = (-1, 1)
    interval: str = 'day'
    exchange: str = 'NSE'
    instrument_type: str = 'equity'


@dataclass
class AppConfig:
    """Application configuration."""

    data_cache_dir: Optional[str] = None
    model_save_path: str = './models'
    log_level: str = 'INFO'
