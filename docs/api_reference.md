# TrendMaster API Reference

This document provides detailed information about the classes and functions in the TrendMaster package.

## Table of Contents

1. [DataLoader](#dataloader)
2. [TransAm (Model)](#transam-model)
3. [Trainer](#trainer)
4. [Inferencer](#inferencer)
5. [Utility Functions](#utility-functions)

## DataLoader

The `DataLoader` class is responsible for loading and preprocessing stock data.

### Methods

#### `__init__(self)`
Initializes the DataLoader object.

#### `authenticate(self, user_id=None, password=None, twofa=None)`
Authenticates with the Zerodha API.

- Parameters:
  - `user_id` (str, optional): Zerodha user ID
  - `password` (str, optional): Zerodha password
  - `twofa` (str, optional): Zerodha two-factor authentication code
- Returns: Authenticated Zerodha kite instance

#### `get_stock_data(self, symbol, from_date, to_date, interval='minute')`
Fetches stock data for a given symbol.

- Parameters:
  - `symbol` (str): Stock symbol
  - `from_date` (str): Start date in 'YYYY-MM-DD' format
  - `to_date` (str): End date in 'YYYY-MM-DD' format
  - `interval` (str, optional): Data interval (default: 'minute')
- Returns: pandas DataFrame containing stock data

#### `preprocess_data(self, data, column='close')`
Preprocesses the data for model input.

- Parameters:
  - `data` (pandas.DataFrame): Stock data
  - `column` (str, optional): Column to preprocess (default: 'close')
- Returns: Preprocessed numpy array

#### `create_sequences(self, data, input_window, output_window)`
Creates input-output sequences for training.

- Parameters:
  - `data` (numpy.array): Preprocessed data
  - `input_window` (int): Number of input time steps
  - `output_window` (int): Number of output time steps
- Returns: List of (input_sequence, target_sequence) tuples

## TransAm (Model)

The `TransAm` class implements the Transformer model for stock price prediction.

### Methods

#### `__init__(self, feature_size=30, num_layers=2, dropout=0.2)`
Initializes the TransAm model.

- Parameters:
  - `feature_size` (int, optional): Size of input features (default: 30)
  - `num_layers` (int, optional): Number of transformer layers (default: 2)
  - `dropout` (float, optional): Dropout rate (default: 0.2)

#### `forward(self, src)`
Performs a forward pass through the model.

- Parameters:
  - `src` (torch.Tensor): Input tensor
- Returns: Output tensor

## Trainer

The `Trainer` class handles model training and validation.

### Methods

#### `__init__(self, model, device, learning_rate=0.001)`
Initializes the Trainer object.

- Parameters:
  - `model` (TransAm): The model to train
  - `device` (torch.device): Device to use for training
  - `learning_rate` (float, optional): Learning rate (default: 0.001)

#### `train(self, train_data, val_data, epochs, batch_size)`
Trains the model.

- Parameters:
  - `train_data` (list): Training data sequences
  - `val_data` (list): Validation data sequences
  - `epochs` (int): Number of training epochs
  - `batch_size` (int): Batch size for training
- Returns: Lists of training and validation losses

#### `validate(self, val_data, batch_size)`
Validates the model on the provided data.

- Parameters:
  - `val_data` (list): Validation data sequences
  - `batch_size` (int): Batch size for validation
- Returns: Validation loss

## Inferencer

The `Inferencer` class handles model inference and evaluation.

### Methods

#### `__init__(self, model, device, data_loader)`
Initializes the Inferencer object.

- Parameters:
  - `model` (TransAm): Trained model
  - `device` (torch.device): Device to use for inference
  - `data_loader` (DataLoader): DataLoader instance

#### `predict(self, symbol, from_date, to_date, input_window, future_steps)`
Makes predictions for future stock prices.

- Parameters:
  - `symbol` (str): Stock symbol
  - `from_date` (str): Start date for historical data
  - `to_date` (str): End date for historical data
  - `input_window` (int): Number of input time steps
  - `future_steps` (int): Number of future time steps to predict
- Returns: DataFrame with predicted prices

#### `evaluate(self, test_data, batch_size)`
Evaluates the model on test data.

- Parameters:
  - `test_data` (list): Test data sequences
  - `batch_size` (int): Batch size for evaluation
- Returns: Test loss

## Utility Functions

### `set_seed(seed)`
Sets random seed for reproducibility.

- Parameters:
  - `seed` (int): Random seed

### `plot_results(train_losses, val_losses)`
Plots training and validation losses.

- Parameters:
  - `train_losses` (list): Training losses
  - `val_losses` (list): Validation losses

### `plot_predictions(actual, predictions)`
Plots actual vs predicted stock prices.

- Parameters:
  - `actual` (pandas.Series): Actual stock prices
  - `predictions` (pandas.DataFrame): Predicted stock prices