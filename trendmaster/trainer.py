import joblib
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
class Trainer:
    def __init__(self, model, device, criterion=None,lr = 0.000001, optimizer=None, scheduler=None, output_window=10, input_window=200,symbol='SBIN'):
        """
        Initializes the Trainer class with a model, device, and optionally a loss criterion, optimizer, and scheduler.
        
        :param model: The model to be trained.
        :param device: The device (e.g., 'cpu' or 'cuda') on which the model will be trained.
        :param criterion: (optional) The loss function used for training. Defaults to MSELoss.
        :param optimizer: (optional) The optimizer used for training. Defaults to Adam optimizer.
        :param scheduler: (optional) The learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
        self.input_window = input_window
        self.output_window= output_window
        self.model = model
        self.device = device
        self.criterion = criterion if criterion else nn.MSELoss()
        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.symbol = 'SBIN'

    def train(self, train_data,scaler=None, batch_size=1, epochs=10, calculate_loss_over_all_values=False, log_interval=100):
        """
        Trains the model for a specified number of epochs on provided training data.
        
        :param train_data: The data on which the model will be trained.
        :param epochs: (optional) The number of epochs to train the model. Defaults to 10.
        :param calculate_loss_over_all_values: (optional) A boolean to decide whether to calculate loss over all values or just the output window. Defaults to False.
        """
        self.scaler = scaler
        self.model.train()
        train_data = self.create_inout_sequences(train_data)
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            total_loss = 0
            for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
                data, targets = self.get_batch(train_data, i, batch_size)
                # print(f"Batch {batch}: data shape {data.shape}, targets shape {targets.shape}")  # Debug statement
                self.optimizer.zero_grad()
                output = self.model(data.to(self.device))
                # print(f"Output shape before windowing: {output.shape}")  # Debug statement
                if calculate_loss_over_all_values:
                    loss = self.criterion(output, targets.to(self.device))
                else:
                    # Ensure output and targets are of the same shape
                    if output.dim() == 1:
                        output = output.unsqueeze(0)
                    if targets.dim() == 1:
                        targets = targets.unsqueeze(0)
                    output = output[:, -self.output_window:]
                    targets = targets[:, -self.output_window:]
                    # print(f"Output shape after windowing: {output.shape}, Targets shape after windowing: {targets.shape}")  # Debug statement
                    loss = self.criterion(output, targets.to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
                if batch % log_interval == 0 and batch > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - epoch_start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // batch_size, self.scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    epoch_start_time = time.time()
            if self.scheduler:
                self.scheduler.step()

    def evaluate(self, data_source, eval_batch_size=1000):
        """
        Evaluates the model on the provided data source.
        
        :param data_source: The data source on which the model will be evaluated.
        :param eval_batch_size: (optional) The batch size used during evaluation. Defaults to 1000.
        :return: The average loss over the evaluated data.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i in range(0, len(data_source) - 1, eval_batch_size):
                data, targets = self.get_batch(data_source, i, eval_batch_size)
                output = self.model(data.to(self.device))
                # Ensure output and targets are of the same shape
                output = output[:, -self.output_window:]
                targets = targets[:, -self.output_window:]
                total_loss += self.criterion(output, targets.to(self.device)).item()
        return total_loss / len(data_source)

    def get_batch(self, source, i, batch_size):
        """
        Retrieves a batch of data from the source starting from index i.
        
        :param source: The data source.
        :param i: The start index of the batch.
        :param batch_size: The size of the batch.
        :return: A tuple of (input, target) tensors.
        """
        seq_len = min(batch_size, len(source) - 1 - i)
        batch = source[i:i+seq_len]
        inputs = torch.FloatTensor([item[0] for item in batch]).to(self.device)
        targets = torch.FloatTensor([item[1] for item in batch]).to(self.device)
        return inputs, targets

    def plot_and_loss(self, data_source, epoch, tknip):
        """
        Evaluates the model on the data source, plots the results, and saves the plot.
        
        :param data_source: The data source for evaluation.
        :param epoch: The current epoch number.
        :param tknip: A token or identifier for the plot file name.
        :return: The average loss over the data source.
        """
        self.model.eval()
        total_loss = 0.
        test_result = torch.Tensor(0)
        truth = torch.Tensor(0)
        with torch.no_grad():
            for i in range(0, len(data_source) - 1):
                data, target = self.get_batch(data_source, i, 1)
                output = self.model(data)
                total_loss += self.criterion(output, target).item()
                test_result = torch.cat((test_result, output[-1].view(-1)), 0)
                truth = torch.cat((truth, target[-1].view(-1)), 0)
        pyplot.plot(test_result.numpy(), color="red")
        pyplot.plot(truth.numpy(), color="blue")
        pyplot.savefig(f'./plots/transformer-epoch_{epoch}_{tknip}.png')
        pyplot.close()
        return total_loss / len(data_source)
    
    def save_model(self, path):
        """
        Save the trained model to a specified path.

        :param path: str, path to save the trained model
        """
        path = f'./models/{self.symbol}_model.pkl'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model,path)
    
    def create_inout_sequences(self, input_data):
        inout_seq = []
        L = len(input_data)
        for i in range(L-self.input_window):
            train_seq = input_data[i:i+self.input_window][:-self.output_window]
            train_label = input_data[i+self.output_window:i+self.input_window+self.output_window]
            inout_seq.append((train_seq, train_label))
        return inout_seq 