import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


class Net(nn.Module):
    def __init__(self, in_features, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        p_t = torch.exp(-bce_loss)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal.mean()


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

in_features = 20

X, y = make_classification(
    n_samples=19230,
    n_features=in_features,
    n_informative=15,
    weights=[0.91, 0.09],
    flip_y=0.03,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


USE_FOCAL_LOSS = True
EPOCHS = 100
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4

model = Net(in_features=in_features).to(device)

criterion = FocalLoss(alpha=0.25, gamma=2.0) if USE_FOCAL_LOSS else nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4
)


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device).float()
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    return total_loss / n


train_losses, test_losses = [], []
best_test_loss = float("inf")
best_state = None
epochs_no_improve = 0

for epoch in range(EPOCHS):
    train_loss = run_epoch(train_loader, train=True)
    test_loss = run_epoch(test_loader, train=False)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    scheduler.step(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    print(
        f"epoch {epoch+1:3d} | train {train_loss:.4f} | test {test_loss:.4f} "
        f"| lr {optimizer.param_groups[0]['lr']:.2e}"
    )

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1} (best test loss {best_test_loss:.4f})")
        break

model.load_state_dict(best_state)


plt.figure()
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.axvline(len(test_losses) - PATIENCE - 1, color="gray", linestyle="--", alpha=0.5, label="best epoch")
plt.title("test and train loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()


model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device).float()
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(yb.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)


precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)

MIN_RECALL = 0.80
valid = recalls[:-1] >= MIN_RECALL
if valid.any():
    best_idx = np.argmax(f1s[:-1][valid])
    best_idx = np.where(valid)[0][best_idx]
else:
    best_idx = np.argmax(f1s[:-1])

best_threshold = thresholds[best_idx]
print(f"\nBest threshold (max F1, recall >= {MIN_RECALL}): {best_threshold:.4f}")

plt.figure()
plt.plot(recalls, precisions)
plt.scatter(recalls[best_idx], precisions[best_idx], color="red", zorder=5, label="chosen threshold")
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision-recall curve")
plt.legend()
plt.savefig("pr_curve.png", dpi=150, bbox_inches="tight")
plt.show()


final_preds = (all_probs >= best_threshold).astype(int)

test_loss_final = best_test_loss
acc = accuracy_score(all_labels, final_preds)
prec = precision_score(all_labels, final_preds)
rec = recall_score(all_labels, final_preds)
f1 = f1_score(all_labels, final_preds)
auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, final_preds)

print("\n--- Final results (best checkpoint + tuned threshold) ---")
print(f"Test Loss: {test_loss_final:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print("Confusion Matrix:")
print(cm)

torch.save({"model_state": model.state_dict(), "threshold": best_threshold}, "model_best.pt")
print("\nSaved best model + threshold to model_best.pt")