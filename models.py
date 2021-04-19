import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class Weather_NN(nn.Module):

    def __init__(self, n_features, input_len):
        super(Weather_NN, self).__init__()
        self.n_features = n_features
        self.input_len = input_len

        self.fc = nn.Sequential(nn.Linear(self.n_features * self.input_len, 128),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.Linear(128, 2))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)
