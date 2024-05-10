from .data_loader import DataLoader
from .trainer import Trainer
from .inferencer import Inferencer
import joblib
class TrendMaster:
    """
    Class to handle loading data, training, and inferencing for stock price prediction.
    """
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

    def load_data(self, filepath):
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
        data = joblib.load(f'{symbol}_data.pkl')
        if data is None:
            data = self.load_data(symbol)
        self.trainer.train(data, transformer_params)
        self.trainer.save_model(f'models/{symbol}_model.pth')


    def infer(self, model_path, symbol, from_date, to_date):
        """
        Perform inference using a trained model for a given stock symbol.

        :param model_path: str, path to the trained model file
        :param symbol: str, stock symbol to infer
        :return: dict, inference results including plots and analytical data
        """
        if self.authenticated_kite is None:
            self.authenticate()
        results = self.inferencer.infer(model_path, symbol, from_date, to_date)
        return results

def main():
    # Example usage
    tm = TrendMaster()
    data = tm.load_data('path_to_data.csv')
    tm.train(data, {'num_layers': 3, 'dropout': 0.1})
    predictions = tm.infer('path_to_trained_model.pth')
    print(predictions)
