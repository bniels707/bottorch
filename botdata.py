import torch

import pandas as pd

from functools import partial
from torch.utils.data import Dataset
from torchvision.transforms import Lambda

NAME_1_COL_IDX = 0 #Bot 1 Name
WEAPON_1_1_COL_IDX = 1 #Bot 1, Weapon 1
WEAPON_1_2_COL_IDX = 2 #Bot 1, Weapon 2

NAME_2_COL_IDX = 3 #Bot 2 Name
WEAPON_2_1_COL_IDX = 4 #Bot 2, Weapon 1
WEAPON_2_2_COL_IDX = 5 #Bot 2, Weapon 2

NAME_WINNER_IDX = 6 #Winner bot name

class BotDataset(Dataset):
    def __init__(self, input_file, transform=None, target_transform=None):
        self.input_data = pd.read_csv(input_file, header=None)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        #Shape is rows x cols
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        data_row = self.input_data.iloc[idx]

        data = data_row[0:NAME_WINNER_IDX]
        winner_name = data_row[NAME_WINNER_IDX]

        #Label is 0 for "Red", 1 for "Blue"
        if winner_name == data_row[NAME_1_COL_IDX]:
            label = 0
        else:
            label = 1

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label

def _get_botname_lambda(botnames):
    #Returns a lambda for projecting robot names as a one-hot tensor
    #https://datascience.stackexchange.com/questions/30215/what-is-one-hot-encoding-in-tensorflow
    return Lambda(lambda y: torch.zeros(len(botnames), dtype=torch.float).scatter_(dim=0, index=torch.tensor(botnames.index(y)), value=1))

def _get_weapon_lambda(weapons):
    #Returns a lambda for projecting weapon as a one-hot tensor
    return Lambda(lambda y: torch.zeros(len(weapons), dtype=torch.float).scatter_(dim=0, index=torch.tensor(weapons.index(y)), value=1))

def _tensor_transform(name_lambda, weapon_lambda, data):
    tensor_accumulator = torch.cat((name_lambda(data[NAME_1_COL_IDX]),
                                    weapon_lambda(data[WEAPON_1_1_COL_IDX])), 0)

    tensor_accumulator = torch.cat((tensor_accumulator,
                                    weapon_lambda(data[WEAPON_1_2_COL_IDX])), 0)

    tensor_accumulator = torch.cat((tensor_accumulator,
                                    name_lambda(data[NAME_2_COL_IDX])), 0)

    tensor_accumulator = torch.cat((tensor_accumulator,
                                    weapon_lambda(data[WEAPON_2_1_COL_IDX])), 0)

    tensor_accumulator = torch.cat((tensor_accumulator,
                                    weapon_lambda(data[WEAPON_2_2_COL_IDX])), 0)

    return tensor_accumulator

def _classifier(value):
    tensor = torch.zeros(1, dtype=torch.float)

    if value != 0:
        tensor[0] = 1

    return tensor

def get_classification_lambda():
    #Return single value "binary" one-hot
    return Lambda(lambda y: _classifier(y))

def get_botdata_features(botdata_path):
    #Returns bot names, bot weapons, as lists
    botnames = []
    weapons = []

    input_data = pd.read_csv(botdata_path, header=None)

    for rowidx in range(0, input_data.shape[0]):
        #Add both competitor names to the botnames list
        botnames.append(input_data.iloc[rowidx][NAME_1_COL_IDX])
        botnames.append(input_data.iloc[rowidx][NAME_2_COL_IDX])

        #Bot 1 weapons
        weapons.append(input_data.iloc[rowidx][WEAPON_1_1_COL_IDX])
        weapons.append(input_data.iloc[rowidx][WEAPON_1_2_COL_IDX])

        #Bot 2 weapons
        weapons.append(input_data.iloc[rowidx][WEAPON_2_1_COL_IDX])
        weapons.append(input_data.iloc[rowidx][WEAPON_2_2_COL_IDX])

    #Dedupe the lists
    botnames = list(set(botnames))
    weapons = list(set(weapons))

    return botnames, weapons

def get_botdata_lambdas(botnames, weapons):
    #Returns a function for converting botnames to a one-hot, and weapon to a one-hot
    botname_lambda = _get_botname_lambda(botnames)
    weapon_lambda = _get_weapon_lambda(weapons)

    return botname_lambda, weapon_lambda

def get_tensor_transform(botname_lambda, weapon_lambda):
    #Returns a function converting data row to a 1D tensor
    return partial(_tensor_transform, botname_lambda, weapon_lambda)
