import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import os

from models import Weather_NN
from dataset import WeatherDataset
from learning import train, test
from utils import train_test_split

device = torch.device("cuda")
n_features = 55
test_split = 0.2
batch_size = 128
num_epochs = 10

model = Weather_NN(n_features, 1).to(device)

dataset = WeatherDataset(interp = False, device = device)

trainloader, testloader = train_test_split(dataset, test_split, batch_size)

print(len(dataset), len(trainloader), len(testloader), batch_size)

model = train(model, trainloader, testloader, device, num_epochs)

torch.save(model.state_dict(),"testmodel.pth")
