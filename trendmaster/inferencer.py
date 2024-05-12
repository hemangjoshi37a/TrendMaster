import symbol
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot
from tqdm import tqdm
from .data_loader import DataLoader
import os 
import joblib

class Inferencer:
    def __init__(self, model_path, kite=None):
        """
        Initialize the Inferencer with a model path and optionally a Zerodha kite instance for data fetching.

        :param model_path: str, path to the trained model file
        :param kite: Zerodha kite instance for fetching data, defaults to None
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file does not exist. Please train and save the model first.")
        self.model = joblib.load(model_path)
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
    

    def predict_future(self, val_data, future_steps, symbol):
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
        # self.model.eval() 
        self.input_window = 200
        self.output_window = 10
        total_loss = 0.
        test_result = torch.Tensor(0)    
        truth = torch.Tensor(0)
        val_data = val_data.values  # Convert pandas DataFrame to numpy array
        _ , data = self.get_batch(val_data, 0,1)
        with torch.no_grad():
            for i in range(0, future_steps,1):
                input = torch.clone(data[-self.input_window:])
                input[-self.output_window:] = 0     
                output = self.model(data[-self.input_window:])                        
                data = torch.cat((data, output[-1:]))
        data = data.cpu().view(-1)
        pyplot.plot(data,color="red")       
        pyplot.plot(data[:self.input_window],color="blue")
        pyplot.grid(True, which='both')
        pyplot.axhline(y=0, color='k')
        pyplot.savefig(f'./transformer-future_{symbol}.png')
        pyplot.close()

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
            self.orig_data = np.append(self.orig_data, dpp[:self.input_window])
            
    def infer(self,  symbol):
        self.authenticate()
        data = DataLoader.get_stock_data(self.kite, symbol)
        # Assuming prediction logic is implemented here
        results = {"data": data, "predictions": None}  # Placeholder for actual prediction results
        return results


    # def get_batch(self, source, i, batch_size):
    #     """
    #     Retrieves a batch of data from the source starting from index i.
        
    #     :param source: The data source.
    #     :param i: The start index of the batch.
    #     :param batch_size: The size of the batch.
    #     :return: A tuple of (input, target) tensors.
    #     """
    #     seq_len = min(batch_size, len(source) - 1 - i)
    #     batch = source[i:i+seq_len]
    #     inputs = torch.FloatTensor([item[0] for item in batch]).to(self.device)
    #     targets = torch.FloatTensor([item[1] for item in batch]).to(self.device)
    #     return inputs, targets
    
    def get_batch(self, source, i, batch_size):
        seq_len = min(batch_size, len(source) - 1 - i)
        batch = source[i:i+seq_len]
        inputs = torch.FloatTensor([item for item in batch]).to(self.device)
        return inputs