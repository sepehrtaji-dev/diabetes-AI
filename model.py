import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn
import torch.optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    print(f"Device : {torch.cuda.get_device_name()}")
except Exception:
    print("Using CPU")
df = pd.read_csv("diabetes_prediction_dataset_cleaned.csv")
x = df.drop("diabetes", axis=1)
y = df["diabetes"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
xscaler = StandardScaler()
x_train_scaled = xscaler.fit_transform(x_train)
x_test_scaled = xscaler.transform(x_test)
x_train_tensor = torch.tensor(x_train_scaled, torch.float32, device)
x_test_tensor = torch.tensor(x_test_scaled, torch.float32, device)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), torch.float32, device)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), torch.float32, device)
train_ds = TensorDataset(x_train_tensor, y_train_tensor)
test_ds = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

    