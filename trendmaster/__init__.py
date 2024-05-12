from symbol import sym_name
from .data_loader import DataLoader
from .trainer import Trainer
from .inferencer import Inferencer
import joblib
import torch
from .model_definitions import PositionalEncoding, TransAm
import os

class TrendMaster:
    """
    Class to handle loading data, training, and inferencing for stock price prediction.
    """
    def __init__(self, symbol_name_stk='SBIN'):
        self.data_loader = DataLoader()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Check if the model file exists
        model_path = f'./models/{symbol_name_stk}_model.pkl'
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            # Initialize a new model if the file does not exist
            self.model = TransAm(feature_size=190, num_layers=2, dropout=0.2).to(self.device)
        self.trainer = Trainer(model=self.model, device=self.device)
        self.authenticated_kite = None
        print(model_path)
        self.inferencer = Inferencer
        
    def authenticate(self):
        if self.authenticated_kite is None:
            self.authenticated_kite = DataLoader.authenticate_user()
            joblib.dump(self.authenticated_kite, 'kite_session.pkl')
        else:
            self.authenticated_kite = joblib.load('kite_session.pkl')

    def load_data(self, symbol):
        """
        Load OHLC historical data from a specified file path.
        """
        if self.authenticated_kite is None:
            self.authenticate()
        data = self.data_loader.get_stock_data(self.authenticated_kite, symbol)
        joblib.dump(data, f'{symbol}_data.pkl')
        return data

    def train(self, symbol, transformer_params):
        """
        Start the training process with the given data and transformer parameters.
        """
        data,scaler = self.data_loader.load(symbol=symbol,filepath='.')
        self.trainer.train(data,scaler=scaler, **transformer_params)
        self.trainer.save_model(f'models/{symbol}_model.pth')


    def infer(self,  symbol):
        """
        Perform inference using a trained model for a given stock symbol.

        :param model_path: str, path to the trained model file
        :param symbol: str, stock symbol to infer
        :return: dict, inference results including plots and analytical data
        """
        if self.authenticated_kite is None:
            self.authenticate()
        self.inferencer=Inferencer(f'./models/{symbol}_model.pth')
        results = self.inferencer.infer(symbol)
        return results
