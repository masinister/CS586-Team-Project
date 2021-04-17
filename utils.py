import torch
import pandas as pd
import numpy as np

def load_data(interp = False, device = torch.device("cpu")):
    data = pd.read_csv('weatherAUS.csv').replace(('Yes', 'No'), (1, 0))
    if interp:
        data = data.interpolate(method ='pad', limit_direction ='forward')
    data = data.dropna()
    X = featurize(data.values[:,:-1], device = device)
    y = torch.tensor(data.values[:,-1:].astype(np.float), device = device)
    return X, y

def featurize(X_raw, device):
    # TODO: transform X_raw to a tensor of stacked feature vectors
    return X_raw

if __name__ == '__main__':
    X, y = load_data()
    print(X.shape, y.shape)
