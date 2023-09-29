import torch
import os
import csv
from typing import Callable
from torch import nn as nn


class MLP(nn.Module):
    def __init__(self, encoder: nn.Module, n=19, with_softmax=True):
        super(MLP, self).__init__()
        self.with_softmax = with_softmax
        self.batch_size = 1
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.classifier = nn.Sequential(
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, n)
        )
        self.encoder = encoder
        self.encoder.eval()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x
    

def train(model: torch.nn.Module, dataloader, loss_fn: Callable, optimizer, device: str, current_epoch: int, log_path: str):
        size = int(len(dataloader.dataset) // dataloader.batch_size)
        train_loss = [] 
        model.train()
        avg_loss = 0
        loss_counter = 0

        for batch, (X, y) in enumerate(dataloader):
            # send all tensors to device
            X1, X2 = X
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            # get prediction(s)
            pred = model(X1, X2)
            # compute loss
            loss = loss_fn(pred, y)
            avg_loss += loss.item()
            loss_counter += 1
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 10 == 0 and batch > 0:
                loss, current = avg_loss / loss_counter, batch
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                train_loss.append([current + (len(dataloader) * current_epoch), loss])
                avg_loss = 0
                loss_counter = 0
        
        if not os.path.isfile(log_path):
            # Schreibe die Daten in die CSV-Datei
            with open(log_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # Füge den Header hinzu
                writer.writerow(['iteration', 'avg_loss'])
                
                # Füge die Daten hinzu
                for loss in train_loss:
                    writer.writerow(loss)

        else:
            with open(log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Füge die Daten hinzu
                for loss in train_loss:
                    writer.writerow(loss)

def run_test(model: nn.Module, dataloader, loss_fn: Callable, device: str, current_epoch: int, log_path: str):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    # prevent pytorch from calculating the gradients -> better performance
    with torch.no_grad():
        for X, y in dataloader:
            # send all tensors to device
            X1, X2 = X
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            # get prediction(s)
            pred = model(X1, X2)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f} \n")
    test_metrics = [current_epoch, loss, correct]

    if not os.path.isfile(log_path):
        # Schreibe die Daten in die CSV-Datei
        with open(log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Füge den Header hinzu
            writer.writerow(['iteration', 'avg_loss', 'accuracy'])
            writer.writerow(test_metrics)

    else:
        with open(log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(test_metrics)