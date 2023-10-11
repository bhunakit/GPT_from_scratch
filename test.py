import torch
import torch.nn as nn
import torch.optim as optim

# Sample data
x = torch.rand(100, 1)  # 100 data points
y = 2 * x + 1 + 0.1 * torch.randn(100, 1)  # y = 2x + 1 with noise

# Define a linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature and one output

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# Define a loss function (Mean Squared Error) and optimizer (e.g., Stochastic Gradient Descent)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x)
    print(outputs)
    loss = criterion(outputs, y)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get the learned parameters (slope and intercept)
learned_slope, learned_intercept = model.linear.weight.item(), model.linear.bias.item()
print(f'Learned slope: {learned_slope:.4f}, Learned intercept: {learned_intercept:.4f}')

print(model(torch.arange(10, dtype=float).float().reshape(10, 1)))