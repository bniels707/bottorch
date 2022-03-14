#!/usr/bin/env python3

import argparse
import torch

from torch import nn
from torch.utils.data import DataLoader
from bottorch.botdata import BotDataset, get_botdata_features, get_botdata_lambdas, get_classification_lambda, get_competitor_list, get_tensor_transform

DATA_PATH = 'BotTorch_Data-Training.csv'
COMPETITORS_PATH = 'BotTorch_Data-S6-Competitors.csv'
BRACKET_PATH = 'BotTorch_Data-S6-Bracket.csv'

DEVICE = "cpu"

#Percentage of samples used for training
TRAINING_SPLIT_PERCENTAGE = 0.7

BATCH_SIZE = 64

BINARY_THRESHOLD = torch.tensor([0.5])

class BotdataNeuralNetwork(nn.Module):
    def __init__(self, botdata_tensor_size, l1_size=None, l2_size=None, state_dict=None):
        super(BotdataNeuralNetwork, self).__init__()

        if state_dict is None:
            if l1_size is None:
                l1_size = botdata_tensor_size // 2

            if l2_size is None:
                l2_size = l1_size // 2
        else:
            botdata_tensor_size = state_dict['linear_relu_stack.0.weight'].size()[1]
            l1_size = state_dict['linear_relu_stack.0.weight'].size()[0]
            l2_size = state_dict['linear_relu_stack.2.weight'].size()[0]

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(botdata_tensor_size, l1_size),
            nn.ReLU(),
            nn.Linear(l1_size, l2_size),
            nn.ReLU(),
            nn.Linear(l2_size, 1),
            nn.Sigmoid()
            )

        if state_dict is not None:
            self.load_state_dict(state_dict)

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

def train_iteration(model, botdataset, training_size, test_size, epochs, l1, l2):
    training_data, test_data = torch.utils.data.random_split(botdataset, (training_size, test_size))

    model.to(DEVICE)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_correct = test(test_dataloader, model, loss_fn)
        print(f"Test Error: \n Accuracy: {(100*(test_correct / len(test_dataloader.dataset))):>0.1f}%, Avg loss: {test_loss / len(test_dataloader):>8f} \n")

    return test_loss, test_correct

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

            model = BotdataNeuralNetwork(botdataset[0][0].shape[0], test_l1, test_l2)

            test_loss, test_correct = train_iteration(model, botdataset, training_size, test_size, training_epochs, test_l1, test_l2)

            if test_correct > best_correct:
                best_correct = test_correct
                best_l1 = test_l1
                best_l2 = test_l2

            current_tuning_step += 1

            print(f"Tuning step: {current_tuning_step}/{tuning_steps}")

    print(f"Done! Best accuracy {(100*(best_correct / test_size)):>0.1f}%, L1: {best_l1}, L2: {best_l2}")

def tune(model, botdataset, training_size, test_size, l1, l2, training_epochs):
    test_loss, test_correct = train_iteration(model, botdataset, training_size, test_size, training_epochs, l1, l2)

    print(f"Done! Accuracy {(100*(test_correct / test_size)):>0.1f}%")

def predict(model, botdata_transform, bot_features, bot_name1, bot_name2):
    #Returns the predicted winner as a name string
    matchup_tensor = botdata_transform([bot_name1, bot_features[bot_name1][0], bot_features[bot_name1][1], bot_name2, bot_features[bot_name2][0], bot_features[bot_name2][1]])

    with torch.no_grad():
        pred = model(matchup_tensor)

    #Make predictions binary
    pred = (pred > BINARY_THRESHOLD).float()

    if pred[0] == 0:
        return bot_name1

    return bot_name2

def predict_rank(model, botdata_transform, bot_features):
    #Prints the list of competitors ordered by number of wins in a round robin
    competitor_list = get_competitor_list(COMPETITORS_PATH)

    win_accumulator = {}

    for competitor in competitor_list:
        win_accumulator[competitor] = 0

    for competitor1 in competitor_list:
        for competitor2 in competitor_list:
            if competitor1 != competitor2:
                winner = predict(model, botdata_transform, bot_features, competitor1, competitor2)

                win_accumulator[winner] += 1

    return win_accumulator

def predict_bracket(model, botdata_transform, bot_features):
    #Prints a list of predicted bracket winners, assuming the competitor list
    #is ordered from highest ranked, to lowest ranked
    bracket_qualifiers = get_competitor_list(BRACKET_PATH)

    round_count = 1
    round_competitors = bracket_qualifiers
    next_round_competitors = []

    while len(round_competitors) >= 2:
        print(f'Round {round_count}')

        for fight_index in range(0, len(round_competitors) // 2):
            competitor1 = round_competitors[fight_index]
            competitor2 = round_competitors[-(fight_index + 1)]

            competitor1_rank = bracket_qualifiers.index(competitor1) + 1
            competitor2_rank = bracket_qualifiers.index(competitor2) + 1

            winner = predict(model, botdata_transform, bot_features, competitor1, competitor2)

            print(f'Fight {fight_index + 1} - #{competitor1_rank} {competitor1} vs #{competitor2_rank} {competitor2} - Winner {winner}')

            next_round_competitors.append(winner)

        round_competitors = next_round_competitors
        next_round_competitors = []
        round_count += 1

def main():
    parser = argparse.ArgumentParser(
        description="Crude fighting robot bracketology using machine learning."
    )

    parser.add_argument(
        "action",
        type=str,
        choices=["hypertune", "tune", "predict", "rank", "bracket"],
        default="predict",
        help="the action to perform, hypertune to print tuned L1, L2 model paramters, tune to iteratively tune the model with the given parameters and save to the specified model, predict to make a prediction using the specified model, rank uses the model to order competitors based on predicted wins in a round robin, bracket takes the competitor list as ranked and predicts a bracket, defaults to predict",
    )

    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default="model.pth",
        help="model to save or load",
    )

    parser.add_argument(
        "competitor1",
        type=str,
        nargs="?",
        help="first competitor for prediction",
    )

    parser.add_argument(
        "competitor2",
        type=str,
        nargs="?",
        help="second competitor for prediction",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="the number of tuning epochs",
    )

    parser.add_argument(
        "--step_size",
        type=int,
        default=100,
        help="the step size to use for hyper tuning",
    )

    parser.add_argument(
        "--l1",
        type=int,
        default=128,
        help="the size of the first neural network layer, required if provided during tune",
    )

    parser.add_argument(
        "--l2",
        type=int,
        default=64,
        help="the size of the second neural network layer, required if provided during tune",
    )

    args = parser.parse_args()

    bot_features = get_botdata_features(DATA_PATH)

    #Convert the features mapping to lists so we can build vectors from them
    bot_names = list(bot_features.keys())

    bot_weapons = []

    for bot_name in bot_features.keys():
        for weapon in bot_features[bot_name]:
            if weapon not in bot_weapons:
                bot_weapons.append(weapon)

    #Build the vector transforms
    botname_lambda, weapon_lambda = get_botdata_lambdas(bot_names, bot_weapons)
    botdata_transform = get_tensor_transform(botname_lambda, weapon_lambda)

    #Make a dataset
    botdataset = BotDataset(DATA_PATH, transform=botdata_transform, target_transform=get_classification_lambda())

    #Calculate the split sizes
    training_size = int(TRAINING_SPLIT_PERCENTAGE * len(botdataset))
    test_size = len(botdataset) - training_size

    if args.action == 'hypertune':
        hyper_tune(botdataset, training_size, test_size, args.step_size, args.epochs) #Best accuracy 62.5%, L1: 2332, L2: 1032
    elif args.action == 'tune':
        model = BotdataNeuralNetwork(botdataset[0][0].shape[0], args.l1, args.l2)

        tune(model, botdataset, training_size, test_size, args.l1, args.l2, args.epochs)

        #Save the model
        torch.save(model.state_dict(), args.model)
    elif args.action == 'predict':
        model = BotdataNeuralNetwork(botdataset[0][0].shape[0], state_dict=torch.load(args.model))

        print(predict(model, botdata_transform, bot_features, args.competitor1, args.competitor2))
    elif args.action == 'rank':
        model = BotdataNeuralNetwork(botdataset[0][0].shape[0], state_dict=torch.load(args.model))

        win_accumulator = predict_rank(model, botdata_transform, bot_features)

        #Print sorted by win
        for idx, competitor in enumerate(sorted(win_accumulator, key=win_accumulator.get, reverse=True)):
            print(idx + 1, ' - ', competitor, win_accumulator[competitor])
    elif args.action == 'bracket':
        model = BotdataNeuralNetwork(botdataset[0][0].shape[0], state_dict=torch.load(args.model))

        predict_bracket(model, botdata_transform, bot_features)
    else:
        raise RuntimeError('Unrecognized action')

if __name__ == "__main__":
    main()
