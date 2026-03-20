"""Training module for time series prediction models."""

from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from trendmaster.config import TrainingConfig
from trendmaster.utils import ModelError, logger

# Silence matplotlib output when not in interactive mode
import matplotlib
matplotlib.use('Agg')


class Trainer:
    """Trainer class for training Transformer models.

    Handles:
        - Training loop with progress bar
        - Validation
        - Early stopping
        - Learning rate scheduling
        - Model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        config: TrainingConfig | None = None,
    ):
        """Initialize the Trainer.

        Args:
            model: The model to train
            device: Device to use for training (cpu or cuda)
            learning_rate: Learning rate for optimizer (default: 0.001)
            config: Optional TrainingConfig dataclass (overrides individual params)
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()

        if config is not None:
            learning_rate = config.learning_rate
            self.step_size = config.step_size
            self.gamma = config.gamma
            self.patience = config.patience
        else:
            self.step_size = 1
            self.gamma = 0.95
            self.patience = 10

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma
        )

        logger.info(
            f"Trainer initialized: lr={learning_rate}, device={device}, "
            f"step_size={self.step_size}, gamma={self.gamma}, patience={self.patience}"
        )

    def train(
        self,
        train_data: List[Tuple],
        val_data: List[Tuple],
        epochs: int,
        batch_size: int = 64,
        patience: int = 10,
        verbose: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """Train the model.

        Args:
            train_data: Training data sequences
            val_data: Validation data sequences
            epochs: Number of training epochs
            batch_size: Batch size for training (default: 64)
            patience: Early stopping patience (default: 10)
            verbose: Whether to print progress (default: True)

        Returns:
            Tuple of (train_losses, val_losses) lists
        """
        import numpy as np
        from tqdm import tqdm

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float('inf')
        epochs_no_improve = 0

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for i in tqdm(
                range(0, len(train_data), batch_size),
                desc=f"Epoch {epoch+1}/{epochs}",
                disable=not verbose,
            ):
                batch = train_data[i:i+batch_size]
                input_sequences = np.array([item[0] for item in batch])
                target_sequences = np.array([item[1] for item in batch])

                inputs = torch.FloatTensor(input_sequences).unsqueeze(-1).to(self.device)
                targets = torch.FloatTensor(target_sequences).unsqueeze(-1).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)

            val_loss = self.validate(val_data, batch_size)
            val_losses.append(val_loss)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

            self.scheduler.step()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        logger.info(f"Training completed. Final train loss: {train_losses[-1]:.6f}, "
                   f"Final val loss: {val_losses[-1]:.6f}")
        return train_losses, val_losses

    def validate(
        self,
        val_data: List[Tuple],
        batch_size: int = 64,
    ) -> float:
        """Validate the model on validation data.

        Args:
            val_data: Validation data sequences
            batch_size: Batch size for validation (default: 64)

        Returns:
            Average validation loss
        """
        import numpy as np

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                input_sequences = np.array([item[0] for item in batch])
                target_sequences = np.array([item[1] for item in batch])

                inputs = torch.FloatTensor(input_sequences).unsqueeze(-1).to(self.device)
                targets = torch.FloatTensor(target_sequences).unsqueeze(-1).to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_model(self, path: str) -> None:
        """Save the model state dict to a file.

        Args:
            path: Path to save the model
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load the model state dict from a file.

        Args:
            path: Path to load the model from

        Raises:
            ModelError: If loading fails
        """
        try:
            self.model.load_state_dict(torch.load(path))
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            raise ModelError(f"Failed to load model from {path}: {e}") from e


def plot_results(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot training and validation losses.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot (optional)
        show: Whether to display the plot (default: True)

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
