import torch
from torch import nn

torch.manual_seed(42)

n = 2000

# Generate synthetic API request features
payload_kb = torch.rand(n, 1) * 1000

db_queries = torch.randint(
    0, 30, (n, 1)
).float()

X = torch.cat(
    [payload_kb, db_queries],
    dim=1
)

# Synthetic relationship between inputs and latency risk
score = (
    1.5 * (payload_kb[:, 0] / 1000)
    + 2.5 * (db_queries[:, 0] / 30)
    + 1.2 * (payload_kb[:, 0] / 1000)
          * (db_queries[:, 0] / 30)
    - 2.5
    + torch.randn(n) * 0.5
)

y = (score > 0).float().unsqueeze(1)

print("Features:", X.shape)
print("Labels:", y.shape)



indices = torch.randperm(n)

train_idx = indices[:1600]
val_idx = indices[1600:1800]
test_idx = indices[1800:]

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]


mean = X_train.mean(dim=0)

std = X_train.std(dim=0).clamp_min(1e-8)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std




model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

print(model)



criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)






epochs = 100

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    logits = model(X_train)

    loss = criterion(logits, y_train)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 20 == 0:

        model.eval()

        with torch.no_grad():

            val_logits = model(X_val)

            val_loss = criterion(
                val_logits,
                y_val
            )

        print(
            f"Epoch {epoch + 1}, "
            f"Train Loss: {loss.item():.4f}, "
            f"Val Loss: {val_loss.item():.4f}"
        )





model.eval()

with torch.no_grad():

    logits = model(X_test)

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).float()

    accuracy = (
        predictions == y_test
    ).float().mean()

print("Test Accuracy:", accuracy.item())
