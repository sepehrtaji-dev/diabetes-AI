import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
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

class ClassficaitionModel(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc4(x)
        return x

epochs = 100
train_losses = []
test_losses = []
model = ClassficaitionModel(x_train_tensor.shape[1])
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
cirtersion = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        y_pred = model(batch_x)
        loss = cirtersion(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            loss = cirtersion(y_pred, batch_y)
            epoch_loss += loss
        epoch_loss /= len(test_loader)
        test_losses.append(epoch_loss)