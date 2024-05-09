from .data_loader import DataLoader
from .trainer import Trainer
from .inferencer import Inferencer

class TrendMaster:
    """
    Class to handle loading data, training, and inferencing for stock price prediction.
    """
    def __init__(self):
        self.data_loader = DataLoader()
        self.trainer = Trainer()
        self.inferencer = Inferencer()

    def load_data(self, filepath):
        """
        Load OHLC historical data from a specified file path.
        """
        return self.data_loader.load(filepath)

    def train(self, data, transformer_params):
        """
        Start the training process with the given data and transformer parameters.
        """
        self.trainer.train(data, transformer_params)

    def infer(self, model_path, symbol):
        """
        Perform inference using a trained model for a given stock symbol.

        :param model_path: str, path to the trained model file
        :param symbol: str, stock symbol to infer
        :return: dict, inference results including plots and analytical data
        """
        self.inferencer.authenticate()
        data = self.inferencer.get_data(symbol)
        results = self.inferencer.run_inference(data)
        self.inferencer.plot_results()
        return results

def main():
    # Example usage
    tm = TrendMaster()
    data = tm.load_data('path_to_data.csv')
    tm.train(data, {'num_layers': 3, 'dropout': 0.1})
    predictions = tm.infer('path_to_trained_model.pth')
    print(predictions)
