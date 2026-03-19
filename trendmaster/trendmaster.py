import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from jugaad_trader import Zerodha

# ---------------------- Utils ----------------------

def set_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def plot_results(train_losses, val_losses):
    """Plot the training and validation losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_predictions(actual, predictions):
    """Plot the actual vs predicted stock prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual')
    plt.plot(predictions['Date'], predictions['Predicted_Close'], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    plt.show()

# ---------------------- DataLoader ----------------------

class DataLoader:
    """DataLoader class for fetching and preprocessing stock data."""

    def __init__(self):
        self.kite = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scalers = {}  # Per-column scalers for multi-feature mode
        self.instruments_cache = {}

    def authenticate(self, user_id=None, password=None, twofa=None):
        """Authenticate with Zerodha."""
        if not all([user_id, password, twofa]):
            user_id = input("Zerodha User ID: ")
            password = input("Zerodha Password: ")
            twofa = input("Zerodha 2FA: ")
        
        self.kite = Zerodha(user_id=user_id, password=password, twofa=twofa)
        self.kite.login()
        return self.kite

    def get_instrument_token(self, symbol, exchange='NSE', instrument_type='equity'):
        """Fetch the instrument token for a given symbol."""
        cache_key = f"{exchange}_{instrument_type}"
        if cache_key not in self.instruments_cache:
            instruments = self.kite.instruments(exchange=exchange, instrument_type=instrument_type)
            self.instruments_cache[cache_key] = instruments
        else:
            instruments = self.instruments_cache[cache_key]
        
        instrument = next((item for item in instruments if item['tradingsymbol'].upper() == symbol.upper()), None)
        if not instrument:
            raise ValueError(f"Instrument '{symbol}' not found in exchange '{exchange}' and type '{instrument_type}'.")
        
        return instrument['instrument_token']

    def get_stock_data(self, symbol, from_date, to_date, interval='minute'):
        """Fetch stock data for a given symbol."""
        if not self.kite:
            raise ValueError("Please authenticate first using the 'authenticate' method.")
        
        tkn = self.get_instrument_token(symbol)
        data = self.kite.historical_data(tkn, from_date, to_date, interval)
        return pd.DataFrame(data)

    def add_technical_indicators(self, data):
        """Add technical indicators to the DataFrame.
        
        Computes RSI (14), EMA-20, EMA-50, MACD, and Signal line.
        Returns the DataFrame with new columns, NaN rows dropped.
        """
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # RSI (14-period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA-20 and EMA-50
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # MACD and Signal
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        df.dropna(inplace=True) 
        return df

    def preprocess_data(self, data, column='close', columns=None, train=True):
        """Preprocess the data for model input.
        
        Args:
            data: DataFrame with stock data.
            column: Single column name for single-feature mode.
            columns: List of column names for multi-feature mode.
            train: If True, fit the scaler(s) on data. If False, use already-fitted scalers.
        
        Returns:
            Scaled data as numpy array. Shape is (N,) for single-feature, (N, F) for multi-feature.
        """
        if columns and len(columns) > 1:
            # Multi-feature mode
            result = []
            for col in columns:
                values = data[col].to_numpy().reshape(-1, 1)
                if train:
                    self.scalers[col] = MinMaxScaler(feature_range=(-1, 1))
                    scaled = self.scalers[col].fit_transform(values)
                else:
                    if col not in self.scalers:
                        # Fallback for old models or if not fitted
                        self.scalers[col] = MinMaxScaler(feature_range=(-1, 1))
                        scaled = self.scalers[col].fit_transform(values)
                    else:
                        scaled = self.scalers[col].transform(values)
                result.append(scaled.reshape(-1))
            # Also keep the single-column scaler in sync with 'close'
            if 'close' in columns:
                self.scaler = self.scalers['close']
            return np.column_stack(result)
        else:
            # Single-feature mode (backward compatible)
            col = columns[0] if columns else column
            amplitude = data[col].to_numpy().reshape(-1, 1)
            if train:
                amplitude_scaled = self.scaler.fit_transform(amplitude)
            else:
                try:
                    amplitude_scaled = self.scaler.transform(amplitude)
                except:
                    amplitude_scaled = self.scaler.fit_transform(amplitude)
            return amplitude_scaled.reshape(-1)

    def create_sequences(self, data, input_window, output_window):
        """Create input-output sequences for training."""
        sequences = []
        L = len(data)
        for i in range(L - input_window - output_window + 1):
            if data.ndim == 1:
                train_seq = np.append(data[i:i+input_window], output_window * [0])
                train_label = data[i:i+input_window+output_window]
            else:
                padding = np.zeros((output_window, data.shape[1]))
                train_seq = np.vstack([data[i:i+input_window], padding])
                train_label = data[i:i+input_window+output_window, 0] # Assume target is first column
            sequences.append((train_seq, train_label))
        return sequences

    def load_or_download_data(self, symbol, from_date, to_date, force_download=False):
        """Load data from file or download if not available."""
        filename = f"{symbol}_data.joblib"
        if os.path.exists(filename) and not force_download:
            data = joblib.load(filename)
        else:
            data = self.get_stock_data(symbol, from_date, to_date)
            joblib.dump(data, filename)
        return data

    def prepare_data(self, symbol, from_date, to_date, input_window, output_window, train_test_split=0.8):
        """Prepare data for training and testing."""
        data = self.load_or_download_data(symbol, from_date, to_date)
        processed_data = self.preprocess_data(data)
        sequences = self.create_sequences(processed_data, input_window, output_window)
        split_idx = int(len(sequences) * train_test_split)
        train_data = sequences[:split_idx]
        test_data = sequences[split_idx:]
        return train_data, test_data

# ---------------------- Model ----------------------

class PositionalEncoding(nn.Module):
    """Positional encoding module for Transformer."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    """Transformer-based Time Series Prediction Model."""

    def __init__(self, input_size=1, d_model=30, num_layers=2, nhead=5, dropout=0.2):
        super(TransAm, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if hasattr(self, 'input_proj'):
            src = self.input_proj(src)
        # If not, it means the model was trained without input_proj (older version).
        # We assume the input dimensions are already correct for pos_encoder.
        if src.ndim == 3 and src.shape[0] != 1 and src.shape[1] == 1:
            # likely already (S, N, E)
            pass
        else:
            # assume (N, S, E)
            src = src.transpose(0, 1)
        src = self.pos_encoder(src)

        # Handle old models that had src_mask and newer ones that don't
        if hasattr(self, 'transformer_encoder'):
            if hasattr(self, 'src_mask'):
                if self.src_mask is None or self.src_mask.size(0) != len(src):
                    device = src.device
                    mask = self._generate_square_subsequent_mask(len(src)).to(device)
                    self.src_mask = mask
                output = self.transformer_encoder(src, self.src_mask)
            else:
                output = self.transformer_encoder(src)
        
        output = self.decoder(output)
        output = output.transpose(0, 1)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# ---------------------- Trainer ----------------------

class Trainer:
    """Trainer class for training the model."""

    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)

    def train(self, train_data, val_data, epochs, batch_size):
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
                batch = train_data[i:i+batch_size]
                input_sequences = np.array([item[0] for item in batch])
                target_sequences = np.array([item[1] for item in batch])
                
                inputs = torch.FloatTensor(input_sequences).to(self.device)
                if inputs.ndim == 2:
                    inputs = inputs.unsqueeze(-1)
                
                targets = torch.FloatTensor(target_sequences).to(self.device)
                if targets.ndim == 2:
                    targets = targets.unsqueeze(-1)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / (len(train_data) // batch_size + 1)
            train_losses.append(avg_train_loss)
            
            val_loss = self.validate(val_data, batch_size)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            self.scheduler.step()
        
        plot_results(train_losses, val_losses)
        return train_losses, val_losses

    def validate(self, val_data, batch_size):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                input_sequences = np.array([item[0] for item in batch])
                target_sequences = np.array([item[1] for item in batch])
                
                inputs = torch.FloatTensor(input_sequences).to(self.device)
                if inputs.ndim == 2:
                    inputs = inputs.unsqueeze(-1)
                
                targets = torch.FloatTensor(target_sequences).to(self.device)
                if targets.ndim == 2:
                    targets = targets.unsqueeze(-1)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / (len(val_data) // batch_size + 1)
        return avg_loss

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

# ---------------------- Inferencer ----------------------

class Inferencer:
    """Inferencer class for making predictions."""

    def __init__(self, model, device, data_loader):
        self.model = model
        self.device = device
        self.data_loader = data_loader

    def predict(self, symbol, from_date, to_date, input_window, future_steps, columns=None, data=None):
        """Make predictions for a stock.
        
        Args:
            symbol: Stock symbol name.
            from_date: Start date for historical data.
            to_date: End date for historical data.
            input_window: Number of time steps to use as input.
            future_steps: Number of future steps to predict.
            columns: List of feature column names for multi-feature mode.
            data: Pre-fetched DataFrame to use instead of downloading.
        
        Returns:
            DataFrame with 'Date' and 'Predicted_Close' columns.
        """
        if data is not None:
            stock_data = data
        else:
            stock_data = self.data_loader.load_or_download_data(symbol, from_date, to_date)
            # Check if we need to add technical indicators if requested but not present
            if columns and any(col in columns for col in ['rsi', 'ema_20', 'ema_50', 'macd', 'signal']) and not all(col in stock_data.columns for col in columns):
                stock_data = self.data_loader.add_technical_indicators(stock_data)
        
        is_multi = columns is not None and len(columns) > 1
        
        if is_multi:
            processed_data = self.data_loader.preprocess_data(stock_data, columns=columns, train=False)
            real_seq = processed_data[-input_window:]  # shape: (input_window, num_features)
            # Pad with zeros for future steps (vectorized prediction mode)
            zero_pad = np.zeros((future_steps, len(columns)))
            input_seq = np.vstack([real_seq, zero_pad])
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)  # (1, seq_len, F)
        else:
            col = columns[0] if columns else 'close'
            processed_data = self.data_loader.preprocess_data(stock_data, column=col, train=False)
            real_seq = processed_data[-input_window:]
            # Pad with zeros for future steps
            zero_pad = np.zeros(future_steps)
            input_seq = np.concatenate([real_seq, zero_pad])
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            # The model outputs the entire sequence. The future predictions are the last `future_steps` elements.
            predictions = output[0, -future_steps:, 0].cpu().numpy().reshape(-1, 1)
        
        # Use the close scaler for inverse transform
        if is_multi:
            close_scaler = self.data_loader.scalers.get('close')
        else:
            close_scaler = self.data_loader.scaler
            
        if close_scaler is None:
             # Try fallback if not found
             close_scaler = getattr(self.data_loader, 'scaler', None)
             
        if close_scaler is None:
             raise ValueError("Scaler not found. Ensure train=True was called or Data loader initialized properly.")
        
        predictions = close_scaler.inverse_transform(predictions)
        
        last_date = pd.to_datetime(stock_data.index[-1])
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
        predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions.flatten()})
        
        return predictions_df

    def evaluate(self, test_data, batch_size):
        self.model.eval()
        total_loss = 0
        criterion = torch.nn.MSELoss()
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                input_sequences = np.array([item[0] for item in batch])
                target_sequences = np.array([item[1] for item in batch])
                
                inputs = torch.FloatTensor(input_sequences).to(self.device)
                if inputs.ndim == 2:
                    inputs = inputs.unsqueeze(-1)
                
                targets = torch.FloatTensor(target_sequences).to(self.device)
                if targets.ndim == 2:
                    targets = targets.unsqueeze(-1)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / (len(test_data) // batch_size + 1)
        print(f"Test Loss: {avg_loss:.6f}")
        return avg_loss

# ---------------------- Package Metadata ----------------------

__all__ = [
    'DataLoader',
    'PositionalEncoding',
    'TransAm',
    'Trainer',
    'Inferencer',
    'set_seed',
    'plot_results',
    'plot_predictions'
]

__version__ = '0.2.3'
