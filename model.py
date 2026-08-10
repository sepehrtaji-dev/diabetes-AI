import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn
import torch.optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes_prediction_dataset_cleaned.csv")
x = df.drop("diabetes", axis=1)
y = df["diabetes"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

    