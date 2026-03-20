"""Data loader for fetching and preprocessing stock data."""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from jugaad_trader import Zerodha
from sklearn.preprocessing import MinMaxScaler

from trendmaster.config import DataConfig
from trendmaster.utils import (
    InstrumentNotFoundError,
    DataLoadError,
    logger,
    get_env_var,
)


class DataLoader:
    """DataLoader class for fetching and preprocessing stock data.

    This class handles:
        - Zerodha API authentication
        - Stock data fetching and caching
        - Data preprocessing with MinMax scaling
        - Sequence creation for time series training
    """

    def __init__(self, scaler_range: Tuple[float, float] = (-1, 1)):
        """Initialize the DataLoader.

        Args:
            scaler_range: MinMax scaler feature range (default: (-1, 1))
        """
        self.kite: Optional[Zerodha] = None
        self.scaler = MinMaxScaler(feature_range=scaler_range)
        self.instruments_cache: dict = {}
        self._env_loaded = False

    def _ensure_env(self) -> None:
        """Ensure environment variables are loaded."""
        if self._env_loaded:
            return
        try:
            from dotenv import load_dotenv
            load_dotenv()
            self._env_loaded = True
            logger.debug("Environment variables loaded from .env file")
        except ImportError:
            logger.warning("python-dotenv not installed, skipping .env file loading")

    def authenticate(
        self,
        user_id: Optional[str] = None,
        password: Optional[str] = None,
        twofa: Optional[str] = None,
    ) -> Zerodha:
        """Authenticate with Zerodha.

        Args:
            user_id: Zerodha user ID. If None, will try to read from env or prompt.
            password: Zerodha password. If None, will try to read from env or prompt.
            twofa: Zerodha 2FA (TOTP) code. If None, will try to read from env or prompt.

        Returns:
            Authenticated Zerodha kite instance

        Raises:
            ConfigurationError: If credentials not provided and not in environment
            AuthenticationError: If authentication fails
        """
        self._ensure_env()

        if not all([user_id, password, twofa]):
            # Try to get from environment variables
            user_id = user_id or get_env_var('ZERODHA_USER_ID', None)
            password = password or get_env_var('ZERODHA_PASSWORD', None)
            twofa = twofa or get_env_var('ZERODHA_TOTP_KEY', None)

            # If still missing some credentials, prompt user
            if not user_id:
                user_id = input("Zerodha User ID: ")
            if not password:
                password = input("Zerodha Password: ")
            if not twofa:
                twofa = input("Zerodha 2FA: ")

        try:
            self.kite = Zerodha(user_id=user_id, password=password, twofa=twofa)
            self.kite.login()
            logger.info("Successfully authenticated with Zerodha")
            return self.kite
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise AuthenticationError(f"Failed to authenticate with Zerodha: {e}") from e

    def get_instrument_token(
        self,
        symbol: str,
        exchange: str = 'NSE',
        instrument_type: str = 'equity',
    ) -> int:
        """Fetch the instrument token for a given symbol.

        Args:
            symbol: Stock/trading symbol
            exchange: Exchange name (default: 'NSE')
            instrument_type: Instrument type (default: 'equity')

        Returns:
            Instrument token as integer

        Raises:
            DataLoadError: If not authenticated
            InstrumentNotFoundError: If symbol not found
        """
        if not self.kite:
            raise DataLoadError("Not authenticated. Please call authenticate() first.")

        cache_key = f"{exchange}_{instrument_type}"
        if cache_key not in self.instruments_cache:
            try:
                instruments = self.kite.instruments(
                    exchange=exchange, instrument_type=instrument_type
                )
                self.instruments_cache[cache_key] = instruments
                logger.debug(f"Loaded {len(instruments)} instruments for {cache_key}")
            except Exception as e:
                raise DataLoadError(f"Failed to load instruments: {e}") from e
        else:
            instruments = self.instruments_cache[cache_key]

        instrument = next(
            (item for item in instruments if item['tradingsymbol'].upper() == symbol.upper()),
            None,
        )

        if not instrument:
            raise InstrumentNotFoundError(
                f"Instrument '{symbol}' not found in exchange '{exchange}' and type '{instrument_type}'"
            )

        return instrument['instrument_token']

    def get_stock_data(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = 'day',
    ) -> pd.DataFrame:
        """Fetch stock data for a given symbol.

        Args:
            symbol: Stock symbol
            from_date: Start date in 'YYYY-MM-DD' format
            to_date: End date in 'YYYY-MM-DD' format
            interval: Data interval - 'day', 'minute', etc. (default: 'day')

        Returns:
            DataFrame with stock data

        Raises:
            DataLoadError: If data fetch fails
        """
        if not self.kite:
            raise DataLoadError("Not authenticated. Please call authenticate() first.")

        try:
            tkn = self.get_instrument_token(symbol)
            data = self.kite.historical_data(tkn, from_date, to_date, interval)
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            logger.info(f"Fetched {len(df)} data points for {symbol}")
            return df
        except InstrumentNotFoundError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to fetch stock data for {symbol}: {e}") from e

    def preprocess_data(
        self,
        data: pd.DataFrame,
        column: str = 'close',
        fit_scaler: bool = False,
    ) -> np.ndarray:
        """Preprocess the data for model input.

        Args:
            data: DataFrame with stock data
            column: Column to preprocess (default: 'close')
            fit_scaler: Whether to fit the scaler or use existing (default: False)

        Returns:
            Preprocessed numpy array
        """
        if column not in data.columns:
            raise DataLoadError(f"Column '{column}' not found in data. Available columns: {list(data.columns)}")

        amplitude = data[column].to_numpy().reshape(-1, 1)
        if fit_scaler:
            amplitude_scaled = self.scaler.fit_transform(amplitude)
            logger.debug(f"Scaler fitted on {len(amplitude)} data points")
        else:
            amplitude_scaled = self.scaler.transform(amplitude)
        return amplitude_scaled.reshape(-1)

    def save_scaler(self, filename: str = 'scaler.joblib') -> None:
        """Save the fitted scaler to a file.

        Args:
            filename: Path to save the scaler
        """
        import joblib
        joblib.dump(self.scaler, filename)
        logger.info(f"Scaler saved to {filename}")

    def load_scaler(self, filename: str = 'scaler.joblib') -> None:
        """Load the scaler from a file.

        Args:
            filename: Path to load the scaler from
        """
        import joblib
        self.scaler = joblib.load(filename)
        logger.info(f"Scaler loaded from {filename}")

    def create_sequences(
        self,
        data: np.ndarray,
        input_window: int,
        output_window: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create input-output sequences for training.

        Args:
            data: Preprocessed data array
            input_window: Number of input time steps
            output_window: Number of output time steps

        Returns:
            List of (input_sequence, target_sequence) tuples
        """
        if len(data) < input_window + output_window:
            raise DataLoadError(
                f"Data length ({len(data)}) is less than required "
                f"input_window + output_window ({input_window + output_window})"
            )

        sequences = []
        L = len(data)
        for i in range(L - input_window - output_window + 1):
            train_seq = np.append(data[i:i+input_window], output_window * [0])
            train_label = data[i:i+input_window+output_window]
            sequences.append((train_seq, train_label))
        return sequences

    def load_or_download_data(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = 'day',
        force_download: bool = False,
) -> pd.DataFrame:
        """Load data from cache or download if not available.

        Args:
            symbol: Stock symbol
            from_date: Start date in 'YYYY-MM-DD' format
            to_date: End date in 'YYYY-MM-DD' format
            interval: Data interval (default: 'day')
            force_download: Force download even if cached (default: False)

        Returns:
            DataFrame with stock data
        """
        cache_dir = get_env_var('DATA_CACHE_DIR', '')
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            filename = os.path.join(cache_dir, f"{symbol}_{interval}_{from_date}_{to_date}.joblib")
        else:
            filename = f"{symbol}_data.joblib"

        if os.path.exists(filename) and not force_download:
            import joblib
            data = joblib.load(filename)
            logger.info(f"Loaded cached data for {symbol} from {filename}")
        else:
            data = self.get_stock_data(symbol, from_date, to_date, interval)
            import joblib
            joblib.dump(data, filename)
            logger.info(f"Downloaded and cached data for {symbol}")
        return data

    def prepare_data(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        input_window: int,
        output_window: int,
        train_test_split: float = 0.8,
        interval: str = 'day',
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
        """Prepare data for training and testing.

        Args:
            symbol: Stock symbol
            from_date: Start date in 'YYYY-MM-DD' format
            to_date: End date in 'YYYY-MM-DD' format
            input_window: Number of input time steps
            output_window: Number of output time steps
            train_test_split: Training data split ratio (default: 0.8)
            interval: Data interval (default: 'day')

        Returns:
            Tuple of (train_data, test_data) sequences
        """
        data = self.load_or_download_data(symbol, from_date, to_date, interval)

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

        logger.info(
            f"Prepared {len(train_data)} training sequences and {len(test_data)} test sequences"
        )
        return train_data, test_data
