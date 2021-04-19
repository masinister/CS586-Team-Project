import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import load_data

class WeatherDataset(Dataset):

    def __init__(self, interp = False, device = torch.device("cpu")):
        self.data, self.labels = load_data(interp, device)

    def __getitem__(self, index):
        # TODO: Return x as a stack of the last N observations
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = WeatherDataset()
    X, y = dataset[0]
    print(X.shape, y)
