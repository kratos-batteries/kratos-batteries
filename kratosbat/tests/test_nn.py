import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import kratosbat.NN.nn as nn


def test_nn_capacity():
    """Test for nn_capacity"""
    model = nn.nn_capacity('kratosbat/Data/NEWTrainingData_MinMaxScaler.csv',
                            115, 100, 75, 2, 4000)
    assert isinstance(model, torch.nn.modules.container.Sequential),\
        "Output is not the right type"


def test_nn_volume():
    """Test for nn_volume"""
    model = nn.nn_volume('kratosbat/Data/NEWTrainingData_MinMaxScaler.csv',
                            2, 100, 75, 1, 4000)
    assert isinstance(model, torch.nn.modules.container.Sequential),\
        "Output is not the right type"
