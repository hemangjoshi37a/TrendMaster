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

# jugaad_trader is an optional broker dependency (Zerodha-only).
# It is imported lazily inside authenticate() so the library can be used
# without a Zerodha account (e.g., with yfinance data in the API server).

# ---------------------- Utils ----------------------

def set_seed(seed: int) -> None:
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
        """Authenticate with Zerodha.
        
        Requires the optional `jugaad-trader` package:
            pip install jugaad-trader
        """
        try:
            from jugaad_trader import Zerodha
        except ImportError as exc:
            raise ImportError(
                "jugaad-trader is required for Zerodha authentication. "
                "Install it with: pip install jugaad-trader"
            ) from exc

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
        
        # Standardize column names to lowercase for the library
        column_map = {c: c.lower() for c in df.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume']}
        df = df.rename(columns=column_map)
        
        if 'close' not in df.columns:
            # Fallback if names are still not standard
            if 'Close' in df.columns: df = df.rename(columns={'Close': 'close'})
            else: raise ValueError(f"Missing required 'close' column in data. Found: {df.columns.tolist()}")

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
                        raise ValueError(f"Scaler for column '{col}' not fitted. Call with train=True first.")
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
                amplitude_scaled = self.scaler.transform(amplitude)
            return amplitude_scaled.reshape(-1)

    def save_scaler(self, filename='scaler.joblib'):
        """Save the fitted scaler to a file."""
        joblib.dump(self.scaler, filename)
        print(f"Scaler saved to {filename}")

    def load_scaler(self, filename='scaler.joblib'):
        """Load the scaler from a file."""
        self.scaler = joblib.load(filename)
        print(f"Scaler loaded from {filename}")

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

    def load_or_download_data(self, symbol, from_date, to_date, force_download=False, source='zerodha'):
        """Load data from file or download if not available.
        
        Args:
            symbol: Stock symbol.
            from_date: Start date.
            to_date: End date.
            force_download: If True, skip local cache.
            source: 'zerodha' or 'yfinance'.
        """
        filename = f"{symbol}_data.joblib"
        if os.path.exists(filename) and not force_download:
            data = joblib.load(filename)
        else:
            if source == 'zerodha' and self.kite:
                data = self.get_stock_data(symbol, from_date, to_date)
            else:
                # Fallback to yfinance
                import yfinance as yf
                yf_symbol = symbol if '.' in symbol else f"{symbol}.NS"
                ticker = yf.Ticker(yf_symbol)
                # Note: 'period' mapping might be needed if using fixed dates
                data = ticker.history(start=from_date, end=to_date)
                # Cleanup yfinance data
                data.index.name = 'date'
                data = data.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
            
            joblib.dump(data, filename)
        return data

    def prepare_data(self, symbol, from_date, to_date, input_window, output_window, train_test_split=0.8):
        """Prepare data for training and testing."""
        data = self.load_or_download_data(symbol, from_date, to_date)
        
        # Fit scaler ONLY on the training portion to prevent data leakage
        split_idx = int(len(data) * train_test_split)
        train_data_raw = data.iloc[:split_idx]
        
        train_amplitude = train_data_raw['close'].to_numpy().reshape(-1, 1)
        self.scaler.fit(train_amplitude)
        
        # Transform the whole dataset using the fitted scaler
        processed_data = self.preprocess_data(data, fit_scaler=False)
        sequences = self.create_sequences(processed_data, input_window, output_window)
        
        seq_split_idx = int(len(sequences) * train_test_split)
        train_data = sequences[:seq_split_idx]
        test_data = sequences[seq_split_idx:]
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
        if hasattr(self, 'input_proj'):
            self.input_proj.bias.data.zero_()
            self.input_proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if hasattr(self, 'input_proj'):
            src = self.input_proj(src)
        # If not, it means the model was trained without input_proj (older version).
        # We assume the input dimensions are already correct for pos_encoder.
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

    @staticmethod
    def load_model(path, device=torch.device('cpu')):
        """Robustly load a model from path, handling whole modules and state_dicts.
        
        Sets up __main__ attributes and monkeypatches for older torch/model versions.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Ensure TransAm and PositionalEncoding are in __main__ for torch.load
        import __main__
        __main__.TransAm = TransAm
        __main__.PositionalEncoding = PositionalEncoding

        # Monkeypatch _LinearWithBias for older torch versions
        import torch.nn.modules.linear
        if not hasattr(torch.nn.modules.linear, '_LinearWithBias'):
            torch.nn.modules.linear._LinearWithBias = torch.nn.modules.linear.Linear

        # Load with weights_only=False because whole modules are used in some versions
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            # Fallback for newer torch security restrictions if possible
            checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict):
            # It's a state_dict
            # Inspect keys to guess input_size and d_model
            input_size = 1
            if 'input_proj.weight' in checkpoint:
                d_model = checkpoint['input_proj.weight'].shape[0]
                input_size = checkpoint['input_proj.weight'].shape[1]
            else:
                d_model = 32 # Fallback default
            
            # Note: We prioritize 32 as d_model based on current library training habits
            model = TransAm(input_size=input_size, d_model=d_model).to(device)
            model.load_state_dict(checkpoint, strict=False)
        else:
            # It's a whole module
            model = checkpoint
            # Patch missing attributes for older torch versions if needed
            for module in model.modules():
                if isinstance(module, torch.nn.TransformerEncoderLayer):
                    if not hasattr(module, 'norm_first'):
                        module.norm_first = False
        
        model.eval()
        return model

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

    def train(self, train_data, val_data, epochs, batch_size, patience=10):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
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
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        
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

    def predict(self, symbol, from_date, to_date, input_window, future_steps, columns=None, data=None, num_samples=1, return_confidence=False, return_all_samples=False):
        """
        Perform future price prediction. 
        If num_samples > 1, performs stochastic sampling (Multiverse Mode).
        """
        if data is not None:
            stock_data = data
        else:
            stock_data = self.data_loader.load_or_download_data(symbol, from_date, to_date)
            
        is_multi = columns is not None and len(columns) > 1
        
        if is_multi:
            processed_data = self.data_loader.preprocess_data(stock_data, columns=columns, train=False)
            real_seq = processed_data[-input_window:]
            zero_pad = np.zeros((future_steps, len(columns)))
            input_seq = np.vstack([real_seq, zero_pad])
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
        else:
            processed_data = self.data_loader.preprocess_data(stock_data, train=False)
            real_seq = processed_data[-input_window:]
            zero_pad = np.zeros(future_steps)
            input_seq = np.concatenate([real_seq, zero_pad])
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(self.device)

        # Use the close scaler for inverse transform
        close_scaler = self.data_loader.scalers.get('close', getattr(self.data_loader, 'scaler', None))
        if close_scaler is None:
             raise ValueError("Scaler not found. Ensure train=True was called or Data loader initialized properly.")

        if num_samples > 1:
            # --- Multiverse Mode (Stochastic) ---
            # Set dropout layers to training mode to enable sampling
            self.model.eval() # Base eval
            for m in self.model.modules():
                if isinstance(m, nn.Dropout) or isinstance(m, nn.TransformerEncoderLayer):
                    m.train() # Force dropout active
            
            all_preds = []
            with torch.no_grad():
                for _ in range(num_samples):
                    output = self.model(input_tensor)
                    preds = output[0, -future_steps:, 0].cpu().numpy().reshape(-1, 1)
                    rescaled = close_scaler.inverse_transform(preds).flatten()
                    all_preds.append(rescaled)
            
            all_preds = np.array(all_preds) # Shape: (num_samples, future_steps)
            mean_pred = np.mean(all_preds, axis=0)
            # Calculate 95% Confidence Interval (2 std devs)
            std_pred = np.std(all_preds, axis=0)
            upper_pred = mean_pred + (2 * std_pred)
            lower_pred = mean_pred - (2 * std_pred)
            
            self.model.eval() # Reset to full eval
            
            last_date = pd.to_datetime(stock_data.index[-1])
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
            
            mean_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': mean_pred})
            upper_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': upper_pred})
            lower_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': lower_pred})
            
            if return_all_samples:
                return mean_df, upper_df, lower_df, all_preds
            return mean_df, upper_df, lower_df

        else:
            # --- Standard Deterministic Mode ---
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
                predictions = output[0, -future_steps:, 0].cpu().numpy().reshape(-1, 1)
                
            predictions_rescaled = close_scaler.inverse_transform(predictions)
            last_date = pd.to_datetime(stock_data.index[-1])
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
            predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions_rescaled.flatten()})
            
            if return_confidence:
                pred_flat = predictions_rescaled.flatten()
                if len(pred_flat) > 1:
                    daily_returns = np.diff(pred_flat) / (np.abs(pred_flat[:-1]) + 1e-9)
                    cv = np.std(daily_returns) / (np.mean(np.abs(daily_returns)) + 1e-9)
                    confidence_score = float(np.clip(100 * (1 / (1 + cv)), 0, 100))
                else:
                    confidence_score = 50.0
                return predictions_df, round(confidence_score, 1)

            return predictions_df

    def stochastic_backtest(self, symbol, start_date, end_date, window=30, horizon=10, num_samples=64, columns=None):
        """
        Runs MC Dropout simulations across a historical window to analyze 
        how the model's uncertainty evolved over time.
        """
        stock_data = self.data_loader.load_or_download_data(symbol, start_date, end_date)
        if len(stock_data) < window + horizon:
            return None
            
        results = []
        # Sample at intervals to avoid excessive computation
        step = max(1, len(stock_data) // 20) 
        
        for i in range(window, len(stock_data) - horizon, step):
            current_slice = stock_data.iloc[:i]
            # Use the existing predict logic but on the historical slice
            mean_df, upper_df, lower_df = self.predict(
                symbol, start_date, current_slice.index[-1].strftime('%Y-%m-%d'),
                window, horizon, columns=columns, data=current_slice, num_samples=num_samples
            )
            
            # Record the "Moment of Truth": What actually happened?
            actual_future = stock_data['Close'].iloc[i:i+horizon].values
            
            results.append({
                "date": current_slice.index[-1].strftime('%Y-%m-%d'),
                "expected": mean_df['Predicted_Close'].tolist(),
                "upper": upper_df['Predicted_Close'].tolist(),
                "lower": lower_df['Predicted_Close'].tolist(),
                "actual": actual_future.tolist()
            })
            
        return results

    def calculate_risk_metrics(self, samples, current_price, target_price=None):
        """
        Calculate professional risk metrics from a sample distribution.
        """
        # Final day prices
        final_prices = samples[:, -1]
        returns = (final_prices - current_price) / current_price
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean() if any(returns <= var_95) else var_95
        
        # Kelly Criterion (Simplified: (p*b - q) / b)
        # b = odds (win_avg / loss_avg), p = prob_win
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        
        p = len(wins) / len(returns)
        avg_win = wins.mean() if len(wins) > 0 else 0.01
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.01
        b = avg_win / avg_loss
        
        kelly = (p * b - (1 - p)) / b if b > 0 else 0
        
        # Probability of Profit (above target or current)
        threshold = target_price if target_price else current_price
        pop = (final_prices > threshold).sum() / len(final_prices)
        
        return {
            "var_95": round(float(var_95 * 100), 2),
            "es_95": round(float(es_95 * 100), 2),
            "kelly_fraction": round(float(np.clip(kelly, 0, 1)), 3),
            "pop": round(float(pop * 100), 2)
        }

    def evaluate(self, test_data, batch_size):
        self.model.eval()
        total_mse_loss = 0
        total_mae_loss = 0
        criterion_mse = torch.nn.MSELoss()
        criterion_mae = torch.nn.L1Loss()
        
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
                mse_loss = criterion_mse(outputs, targets)
                mae_loss = criterion_mae(outputs, targets)
                total_mse_loss += mse_loss.item()
                total_mae_loss += mae_loss.item()
        
        avg_mse = total_mse_loss / (len(test_data) // batch_size + 1)
        avg_mae = total_mae_loss / (len(test_data) // batch_size + 1)
        print(f"Test Loss (MSE): {avg_mse:.6f} | Test Error (MAE): {avg_mae:.6f}")
        return avg_mse

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
