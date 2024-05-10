import joblib
from .data_loader import DataLoader
from .trainer import Trainer
from .inferencer import Inferencer

class TrendMaster:
    def __init__(self):
        self.data_loader = DataLoader()
        self.trainer = Trainer()
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

    def infer_model(self, symbol, from_date, to_date):
        if self.authenticated_kite is None:
            self.authenticate()
        model_path = f'models/{symbol}_model.pth'
        results = self.inferencer.infer(model_path, symbol, from_date, to_date)
        return results

def main():
    tm = TrendMaster()
    symbol = 'SBIN'
    transformer_params = {'num_layers': 3, 'dropout': 0.1}
    tm.train_model(symbol, transformer_params)
    predictions = tm.infer_model(symbol, '2021-01-01', '2021-01-10')
    print(predictions)

if __name__ == "__main__":
    main()