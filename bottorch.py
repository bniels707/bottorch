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
EPOCHS = 5

BINARY_THRESHOLD = torch.tensor([0.5])

class BotdataNeuralNetwork(nn.Module):
    def __init__(self, botdata_tensor_size, l1_size=None, l2_size=None):
        super(BotdataNeuralNetwork, self).__init__()

        if l1_size is None:
            l1_size = botdata_tensor_size // 2

        if l2_size is None:
            l2_size = l1_size // 2

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(botdata_tensor_size, l1_size),
            nn.ReLU(),
            nn.Linear(l1_size, l2_size),
            nn.ReLU(),
            nn.Linear(l2_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
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

    return test_loss, correct

def train_iteration(botdataset, training_size, test_size, epochs, l1, l2):
    training_data, _, test_data = torch.utils.data.random_split(botdataset, (training_size, 0, test_size))

    model = BotdataNeuralNetwork(botdataset[0][0].shape[0], l1, l2).to(DEVICE)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_correct = test(test_dataloader, model, loss_fn)
        print(f"Test Error: \n Accuracy: {(100*(test_correct / len(test_dataloader.dataset))):>0.1f}%, Avg loss: {test_loss / len(test_dataloader):>8f} \n")

    return test_loss, test_correct, model

def hyper_tune(botdataset, training_size, test_size, step_size, training_epochs):
    current_tuning_step = 0
    tuning_steps = len(range(botdataset[0][0].shape[0] // 10, botdataset[0][0].shape[0] * 10, step_size)) * len(range(botdataset[0][0].shape[0] // 10, botdataset[0][0].shape[0] * 10, step_size))

    best_correct = 0
    best_l1 = 0
    best_l2 = 0

    #Parameter tuning loop
    for test_l1 in range(botdataset[0][0].shape[0] // 10, botdataset[0][0].shape[0] * 10, step_size):
        for test_l2 in range(botdataset[0][0].shape[0] // 10, botdataset[0][0].shape[0] * 10, step_size):
            print(f"L1: {test_l1}, L2: {test_l2}")

            test_loss, test_correct, _ = train_iteration(botdataset, training_size, test_size, training_epochs, test_l1, test_l2)

            if test_correct > best_correct:
                best_correct = test_correct
                best_l1 = test_l1
                best_l2 = test_l2

            current_tuning_step += 1

            print(f"Tuning step: {current_tuning_step}/{tuning_steps}")

def tune(botdataset, training_size, test_size, l1, l2, training_epochs):
    test_loss, test_correct, model = train_iteration(botdataset, training_size, test_size, training_epochs, l1, l2)

    print(f"Done! Accuracy {(100*(test_correct / test_size)):>0.1f}%")

    return model

def main():
    bot_names, bot_weapons = botdata.get_botdata_features(DATA_PATH)
    botname_lambda, weapon_lambda = botdata.get_botdata_lambdas(bot_names, bot_weapons)
    botdata_transform = botdata.get_tensor_transform(botname_lambda, weapon_lambda)

    botdataset = botdata.BotDataset(DATA_PATH, transform=botdata_transform, target_transform=botdata.get_classification_lambda())

    training_size = int(TRAINING_SPLIT_PERCENTAGE * len(botdataset))
    test_size = len(botdataset) - training_size

    #hyper_tune(botdataset, training_size, test_size, 100, 5) #L1: 32, L2: 1332
    tune(botdataset, training_size, test_size, 32, 1332, 1000)

if __name__ == "__main__":
    main()
