from .trainer import Trainer
from .inferencer import Inferencer
from .data_loader import DataLoader
import torch
import torch.nn as nn
from .model_definitions import  TransAm
import joblib 

class TrendMaster:
    def __init__(self):
        self.data_loader = DataLoader()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransAm(feature_size=190, num_layers=2, dropout=0.2).to(self.device)
        self.trainer = Trainer(model=self.model, device=self.device)
        self.inferencer = Inferencer()
        self.authenticated_kite = None

    def authenticate(self):
        if self.authenticated_kite is None:
            self.authenticated_kite = self.data_loader.authenticate_user()
            joblib.dump(self.authenticated_kite, 'kite_session.pkl')
        else:
            self.authenticated_kite = joblib.load('kite_session.pkl')
            # Additional checks for validity can be added here

    def load_data(self, symbol):
        if self.authenticated_kite is None:
            self.authenticate()
        data = self.data_loader.get_stock_data(self.authenticated_kite, symbol)
        joblib.dump(data, f'{symbol}_data.pkl')
        return data

    def train_model(self, symbol, transformer_params):
        data = joblib.load(f'{symbol}_data.pkl')
        if data is None:
            data = self.load_data(symbol)
        self.trainer.train(data, transformer_params)
        self.trainer.save_model(f'models/{symbol}_model.pth')

    def infer_model(self, symbol):
        if self.authenticated_kite is None:
            self.authenticate()
        model_path = f'models/{symbol}_model.pth'
        results = self.inferencer.infer( symbol,self.authenticated_kite)
        return results