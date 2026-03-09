import torch
import torch.nn as nn
import torch.optim as optim

# Dummy data (features and labels)
X = torch.rand(100, 10)  # 100 samples, 10 features each
y = torch.randint(0, 2, (100,))  # Binary labels (0 or 1)

# Simple MLP Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),  # Input layer (10 features) → hidden layer (32 neurons)
            nn.ReLU(),          # Activation function
            nn.Linear(32, 2)    # Output layer (2 classes)
        )

    def forward(self, x):
        return self.fc(x)

# Model, loss, and optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):  # 10 epochs
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')