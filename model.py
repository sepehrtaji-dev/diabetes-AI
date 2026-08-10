import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn
import torch.optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("diabetes_prediction_dataset_cleaned.csv")
    