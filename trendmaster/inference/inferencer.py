"""Inference module for time series prediction models."""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from trendmaster.utils import ModelError, logger

# Silence matplotlib output when not in interactive mode
import matplotlib
matplotlib.use('Agg')


class Inferencer:
    """Inferencer class for making predictions with trained models.

    Handles:
        - Single-step and multi-step predictions
        - Model evaluation
        - Result visualization
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        data_loader,
    ):
        """Initialize the Inferencer.

        Args:
            model: Trained model for inference
            device: Device to use for inference (cpu or cuda)
            data_loader: DataLoader instance for data preprocessing
        """
        self.model = model
        self.device = device
        self.data_loader = data_loader
        logger.info(f"Inferencer initialized with model on {device}")

    def predict(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        input_window: int,
        future_steps: int,
        return_only_predictions: bool = True,
    ) -> pd.DataFrame:
        """Make future stock price predictions.

        Args:
            symbol: Stock symbol
            from_date: Start date for historical data
            to_date: End date for historical data
            input_window: Number of input time steps
            future_steps: Number of future time steps to predict
            return_only_predictions: If True, don't show plot (default: True)

        Returns:
            DataFrame with Date and Predicted_Close columns
        """
        import numpy as np
        import pandas as pd
        import torch

        data = self.data_loader.load_or_download_data(symbol, from_date, to_date)
        processed_data = self.data_loader.preprocess_data(data, fit_scaler=False)

        if len(processed_data) < input_window:
            raise ModelError(
                f"Insufficient data: {len(processed_data)} points, but input_window={input_window}"
            )

        input_seq = processed_data[-input_window:]
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(self.device)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for _ in range(future_steps):
                output = self.model(input_tensor)
                pred = output[:, -1, 0].item()
                predictions.append(pred)
                input_seq = np.append(input_seq[1:], pred)
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(self.device)

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.data_loader.scaler.inverse_transform(predictions)

        last_date = pd.to_datetime(data.index[-1])
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=future_steps
        )
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predictions.flatten(),
        })

        logger.info(f"Generated {future_steps}-step prediction for {symbol}")

        if not return_only_predictions:
            plot_predictions(data['close'], predictions_df, show=False)

        return predictions_df

    def evaluate(
        self,
        test_data: List[Tuple],
        batch_size: int = 32,
    ) -> dict:
        """Evaluate the model on test data.

        Args:
            test_data: Test data sequences
            batch_size: Batch size for evaluation (default: 32)

        Returns:
            Dictionary with MSE, RMSE, and MAE metrics
        """
        import numpy as np
        import torch
        import math

        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0

        criterion = torch.nn.MSELoss()
        mae_criterion = torch.nn.L1Loss()

        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                input_sequences = np.array([item[0] for item in batch])
                target_sequences = np.array([item[1] for item in batch])

                inputs = torch.FloatTensor(input_sequences).unsqueeze(-1).to(self.device)
                targets = torch.FloatTensor(target_sequences).unsqueeze(-1).to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                mae_loss = mae_criterion(outputs, targets)

                batch_size_actual = inputs.size(0)
                total_loss += loss.item() * batch_size_actual
                total_mae += mae_loss.item() * batch_size_actual
                total_samples += batch_size_actual

        mse = total_loss / max(total_samples, 1)
        rmse = math.sqrt(mse)
        mae = total_mae / max(total_samples, 1)

        logger.info(
            f"Test Metrics - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}"
        )

        return {'mse': mse, 'rmse': rmse, 'mae': mae}


def plot_predictions(
    actual: pd.Series,
    predictions: pd.DataFrame,
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot actual vs predicted stock prices.

    Args:
        actual: Actual stock prices as pandas Series
        predictions: Predicted prices DataFrame with 'Date' and 'Predicted_Close' columns
        save_path: Path to save the plot (optional)
        show: Whether to display the plot (default: True)

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual')
    plt.plot(
        predictions['Date'],
        predictions['Predicted_Close'],
        label='Predicted',
        linewidth=2,
    )
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Prediction plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
