#!/usr/bin/env python3

import botdata
import torch

from torch import nn
from torch.utils.data import DataLoader

DATA_PATH = 'BotTorch_Data-Training.csv'

DEVICE = "cpu"

#Percentage of samples used for training
TRAINING_SPLIT_PERCENTAGE = 0.7

BATCH_SIZE = 64

#Number of times to do the train / test cycle
EPOCHS = 50

BINARY_THRESHOLD = torch.tensor([0.5])

class BotdataNeuralNetwork(nn.Module):
    def __init__(self, botdata_tensor_size, label_tensor_size):
        super(BotdataNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(botdata_tensor_size, botdata_tensor_size),
            nn.ReLU(),
            nn.Linear(botdata_tensor_size, botdata_tensor_size // 2),
            nn.ReLU(),
            nn.Linear(botdata_tensor_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    trained_counter = 0
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print results after each batch
        trained_counter += len(X)
        print(f"loss: {loss.item():>7f}  [{trained_counter:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)

            #Make predictions binary
            pred = (pred > BINARY_THRESHOLD).float()

            test_loss += loss_fn(pred, y).item()

            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    bot_names, bot_weapons = botdata.get_botdata_features(DATA_PATH)
    botname_lambda, weapon_lambda = botdata.get_botdata_lambdas(bot_names, bot_weapons)
    botdata_transform = botdata.get_tensor_transform(botname_lambda, weapon_lambda)

    botdataset = botdata.BotDataset(DATA_PATH, transform=botdata_transform, target_transform=botdata.get_classification_lambda())

    print(botdataset[0][0].shape)

    print(len(botdataset))

    training_size = int(TRAINING_SPLIT_PERCENTAGE * len(botdataset))
    test_size = len(botdataset) - training_size

    training_data, _, test_data = torch.utils.data.random_split(botdataset, (training_size, 0, test_size))

    print(len(training_data))
    print(len(test_data))

    model = BotdataNeuralNetwork(botdataset[0][0].shape[0], len(bot_names)).to(DEVICE)

    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    print("Done!")

if __name__ == "__main__":
    main()
