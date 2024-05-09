import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot

class Trainer:
    def __init__(self, model, device, criterion=None, optimizer=None, scheduler=None):
        """
        Initializes the Trainer class with a model, device, and optionally a loss criterion, optimizer, and scheduler.
        
        :param model: The model to be trained.
        :param device: The device (e.g., 'cpu' or 'cuda') on which the model will be trained.
        :param criterion: (optional) The loss function used for training. Defaults to MSELoss.
        :param optimizer: (optional) The optimizer used for training. Defaults to Adam optimizer.
        :param scheduler: (optional) The learning rate scheduler.
        """
        self.model = model
        self.device = device
        self.criterion = criterion if criterion else nn.MSELoss()
        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler

    def train(self, train_data, epochs=10, calculate_loss_over_all_values=False):
        """
        Trains the model for a specified number of epochs on provided training data.
        
        :param train_data: The data on which the model will be trained.
        :param epochs: (optional) The number of epochs to train the model. Defaults to 10.
        :param calculate_loss_over_all_values: (optional) A boolean to decide whether to calculate loss over all values or just the output window. Defaults to False.
        """
        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            total_loss = 0
            for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
                data, targets = self.get_batch(train_data, i, batch_size)
                self.optimizer.zero_grad()
                output = self.model(data.to(self.device))
                loss = self.criterion(output, targets.to(self.device)) if calculate_loss_over_all_values else self.criterion(output[-output_window:], targets[-output_window:].to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
                if batch % log_interval == 0 and batch > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // batch_size, self.scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()
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
        data = source[i:i+seq_len]
        input = torch.stack([item[0] for item in data]).to(self.device)
        target = torch.stack([item[1] for item in data]).to(self.device)
        return input, target

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
        torch.save(self.model.state_dict(), path)