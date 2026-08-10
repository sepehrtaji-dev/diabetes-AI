import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
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
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32, device=device)
train_ds = TensorDataset(x_train_tensor, y_train_tensor)
test_ds = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

class ClassficaitionModel(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
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
        epoch_loss += loss.item()
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
            epoch_loss += loss.item()
        epoch_loss /= len(test_loader)
        test_losses.append(epoch_loss)
    if epoch % 10 == 0:
        print(f"epoch : {epoch}/100")
all_preds = []
all_labels = []
all_probs = []
test_loss = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        outputs = model(X_batch)
        loss = cirtersion(outputs, y_batch.view(-1, 1).float())
        test_loss += loss.item()
        
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy().flatten())

test_loss /= len(test_loader)

print(f'Test Loss: {test_loss:.4f}')
print(f'Accuracy: {accuracy_score(all_labels, all_preds):.4f}')
print(f'Precision: {precision_score(all_labels, all_preds):.4f}')
print(f'Recall: {recall_score(all_labels, all_preds):.4f}')
print(f'F1-Score: {f1_score(all_labels, all_preds):.4f}')
print(f'AUC: {roc_auc_score(all_labels, all_probs):.4f}')
print('\nConfusion Matrix:')
print(confusion_matrix(all_labels, all_preds))

correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
total = len(all_labels)
print(f'\nCorrect: {correct}/{total}')
plt.plot(range(epochs), train_losses, label="train")
plt.plot(range(epochs), test_losses, label="test")
plt.title("test and train loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()