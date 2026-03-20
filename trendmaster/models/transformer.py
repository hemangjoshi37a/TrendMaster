"""Transformer-based time series prediction models."""

import math
from typing import Optional

import torch
import torch.nn as nn

from trendmaster.config import ModelConfig
from trendmaster.utils import logger


class PositionalEncoding(nn.Module):
    """Positional encoding module for Transformer.

    Adds positional information to the input embeddings using sinusoidal functions.

    Reference:
        Vaswani et al., "Attention Is All You Need", 2017
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize the PositionalEncoding module.

        Args:
            d_model: The dimension of the model embeddings
            max_len: Maximum sequence length (default: 5000)
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for positional encoding.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    """Transformer-based Time Series Prediction Model.

    This model uses a Transformer encoder architecture for predicting time series data.
    It projects input features to a higher dimension, applies positional encoding,
    processes through multiple transformer layers, and projects back to a single output.

    Architecture:
        Input -> Linear Projection -> Positional Encoding ->
        Transformer Encoder Layers -> Linear Decoder -> Output
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 30,
        num_layers: int = 2,
        nhead: int = 5,
        dropout: float = 0.2,
        config: Optional[ModelConfig] = None,
    ):
        """Initialize the TransAm model.

        Args:
            input_size: Number of input features (default: 1)
            d_model: Dimension of the model embeddings (default: 30)
            num_layers: Number of transformer encoder layers (default: 2)
            nhead: Number of attention heads (default: 5)
            dropout: Dropout rate (default: 0.2)
            config: Optional ModelConfig dataclass (overrides individual params if provided)
        """
        super(TransAm, self).__init__()

        if config is not None:
            input_size = config.input_size
            d_model = config.d_model
            num_layers = config.num_layers
            nhead = config.nhead
            dropout = config.dropout

        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Linear(d_model, 1)

        self.init_weights()
        logger.info(
            f"TransAm initialized: input_size={input_size}, d_model={d_model}, "
            f"num_layers={num_layers}, nhead={nhead}, dropout={dropout}"
        )

    def init_weights(self) -> None:
        """Initialize the weights of the decoder layer."""
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            src: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, 1)
        """
        src = self.input_proj(src)
        src = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = output.transpose(0, 1)  # (batch_size, seq_len, 1)
        return output

    def get_num_params(self, only_trainable: bool = True) -> int:
        """Get the number of parameters in the model.

        Args:
            only_trainable: Count only trainable parameters (default: True)

        Returns:
            Number of parameters
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not only_trainable
        )
