import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot
from tqdm import tqdm
from data_loader import *

class Inferencer:
    def __init__(self, model_path, kite=None):
        """
        Initialize the Inferencer with a model path and optionally a Zerodha kite instance for data fetching.

        :param model_path: str, path to the trained model file
        :param kite: Zerodha kite instance for fetching data, defaults to None
        """
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.kite = kite
        self.orig_data = np.array([])
        self.col_list = []

    def authenticate(self):
        """
        Authenticate the user if not already authenticated.
        """
        if not self.kite:
            self.kite = DataLoader.authenticate_user()

    def get_data(self, symbol):
        """
        Fetch data for a given stock symbol using the authenticated kite instance.

        :param symbol: str, the stock symbol to fetch data for
        :return: np.array, the fetched stock data
        """
        return DataLoader.get_stock_data(self.kite, symbol)
    

    def predict_future_open(self, val_data, future_steps, identifier):
        """
        Predict future values using the model for a given number of steps.

        Args:
        model (torch.nn.Module): The loaded PyTorch model for making predictions.
        val_data (np.array): The validation data used for making predictions.
        future_steps (int): The number of future steps to predict.
        identifier (int): A unique identifier for the prediction session.

        Returns:
        np.array: The predicted values.
        """
        # Dummy function to simulate prediction
        pass

    def plot_results(self):
        plot_df = pd.DataFrame(self.col_list)
        trps = plot_df.transpose()
        trps.plot()
        pyplot.savefig(f'./nmnm/test_plot.png')
        pyplot.close()

    def run_inference(self, test_len, identifier):
        for one_part_point in tqdm(range(test_len)):
            val_data = self.get_data(identifier)
            dpp = self.predict_future_open(self.model, val_data, 1000, identifier)
            if self.orig_data.size != 0:
                diff = self.orig_data[-1] - dpp[301].numpy()
                dpp = dpp - diff
            self.col_list.append(np.append(self.orig_data, dpp))
            self.orig_data = np.append(self.orig_data, dpp[:input_window])

# # Example usage
# inferencer = Inferencer('./best_model_multi18.pt')
# inferencer.run_inference(2, 3356417)
# inferencer.plot_results()