import torch
import pandas as pd
import numpy as np
import datetime

def load_data(interp = False, device = torch.device("cpu")):
    data = pd.read_csv('weatherAUS.csv').replace(('Yes', 'No'), (1, 0))
    if interp:
        data = data.interpolate(method ='pad', limit_direction ='forward')
    data = data.dropna()
    X = featurize(data.values[:,:-1], device = device)
    y = torch.tensor(data.values[:,-1:].astype(float), device = device)
    return X, y

def featurize(X_raw, device):
    return torch.tensor(list(map(featurize_one, X_raw)), device = device)

def featurize_one(row):
    return np.concatenate((time_encoding(row[0]),
                           loc_encoding(row[1]),
                           row[2:7],
                           wind_encoding(row[7]),
                           [row[8]],
                           wind_encoding(row[9]),
                           wind_encoding(row[10]),
                           row[11:])).astype(float)

def sin_cos(n):
    theta = 2 * np.pi * n
    return (np.sin(theta), np.cos(theta))

def loc_encoding(loc):
    locs = np.array(['Cobar', 'CoffsHarbour', 'Moree', 'NorfolkIsland', 'Sydney', 'SydneyAirport',
            'WaggaWagga', 'Williamtown', 'Canberra', 'Sale', 'MelbourneAirport',
            'Melbourne', 'Mildura', 'Portland', 'Watsonia', 'Brisbane', 'Cairns',
            'Townsville', 'MountGambier', 'Nuriootpa', 'Woomera', 'PerthAirport', 'Perth',
            'Hobart', 'AliceSprings', 'Darwin'])
    return np.where(locs == loc, 1, 0)

def time_encoding(time):
    d = datetime.datetime.strptime(time, '%Y-%m-%d')
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return list(sum(map(lambda t: sin_cos(t), [(d.month - 1)/ 12, (d.day - 1) / months[d.month - 1], d.weekday() / 7]), ()))

def wind_encoding(dir):
    dirs = ['E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW', 'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE']
    angle = dirs.index(dir) / 16 * np.pi
    return sin_cos(angle)

if __name__ == '__main__':
    X, y = load_data()
    print(X.shape, y.shape)
    print(X[0])
    print(wind_encoding('W'))
