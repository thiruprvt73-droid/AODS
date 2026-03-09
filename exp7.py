import numpy as np
# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
# Training data (4 samples, 2 features)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
# Target output
y = np.array([[0], [1], [1], [0]])
# Initialize weights randomly
np.random.seed(42)
weights = np.random.rand(2, 1)
bias = np.random.rand(1)

# Learning rate
lr = 0.1

# Training loop
for epoch in range(10000):
    # Forward pass
    linear_output = np.dot(X, weights) + bias
    predictions = sigmoid(linear_output)
    
    # Compute the error
    error = y - predictions
    
    # Backpropagation
    d_pred = error * sigmoid_derivative(predictions)  # Gradient of the loss w.r.t predictions
    weights += np.dot(X.T, d_pred) * lr               # Update weights
    bias += np.sum(d_pred) * lr                       # Update bias

    # Print error every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))  # Mean Squared Error
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Final predictions
print("Final Predictions:")
print(predictions)
