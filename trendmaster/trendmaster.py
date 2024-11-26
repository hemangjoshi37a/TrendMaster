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

    def preprocess_data(self, data, column='close'):
        """Preprocess the data for model input."""
        amplitude = data[column].to_numpy().reshape(-1, 1)
        amplitude_scaled = self.scaler.fit_transform(amplitude)
        return amplitude_scaled.reshape(-1)

    def create_sequences(self, data, input_window, output_window):
        """Create input-output sequences for training."""
        sequences = []
        L = len(data)
        for i in range(L - input_window - output_window + 1):
            train_seq = np.append(data[i:i+input_window], output_window * [0])
            train_label = data[i:i+input_window+output_window]
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
        src = self.input_proj(src)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = output.transpose(0, 1)
        return output

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
                
                inputs = torch.FloatTensor(input_sequences).unsqueeze(-1).to(self.device)
                targets = torch.FloatTensor(target_sequences).unsqueeze(-1).to(self.device)
                
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
                
                inputs = torch.FloatTensor(input_sequences).unsqueeze(-1).to(self.device)
                targets = torch.FloatTensor(target_sequences).unsqueeze(-1).to(self.device)
                
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

    def predict(self, symbol, from_date, to_date, input_window, future_steps):
        data = self.data_loader.load_or_download_data(symbol, from_date, to_date)
        processed_data = self.data_loader.preprocess_data(data)
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
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
        predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions.flatten()})
        
        plot_predictions(data['close'], predictions_df)
        
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
                
                inputs = torch.FloatTensor(input_sequences).unsqueeze(-1).to(self.device)
                targets = torch.FloatTensor(target_sequences).unsqueeze(-1).to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / (len(test_data) // batch_size + 1)
        print(f"Test Loss: {avg_loss:.6f}")
        return avg_loss

# ---------------------- Package Metadata ----------------------

__all__ = [
    'DataLoader',
    'TransAm',
    'Trainer',
    'Inferencer',
    'set_seed',
    'plot_results',
    'plot_predictions'
]

__version__ = '0.2.3'
