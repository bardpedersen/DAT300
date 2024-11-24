import numpy as np
import matplotlib.pyplot as plt

# Create a simple dataset
np.random.seed(0)
X = np.linspace(0, 10, 100)
true_w, true_b = 2, 1
Y = true_w * X + true_b + np.random.normal(scale=1.0, size=X.shape)  # noisy data

# Define the MSE loss function
def mse_loss(w, b, X, Y):
    Y_pred = w * X + b
    return np.mean((Y - Y_pred) ** 2)

# Compute the gradient of MSE with respect to w and b for a batch
def compute_gradients(w, b, X_batch, Y_batch):
    N = len(X_batch)
    Y_pred = w * X_batch + b
    error = Y_pred - Y_batch
    dw = (2 / N) * np.sum(error * X_batch)
    db = (2 / N) * np.sum(error)
    return dw, db

# Perform stochastic gradient descent and store the path of parameter updates
def stochastic_gradient_descent(X, Y, learning_rate=0.01, iterations=100, batch_size=10):
    w, b = 0.0, 0.0  # Initialize parameters
    history = []  # To store (w, b) at each iteration
    for i in range(iterations):
        # Sample a random mini-batch of data
        indices = np.random.choice(len(X), batch_size, replace=False)
        X_batch, Y_batch = X[indices], Y[indices]
        
        # Compute gradients on the mini-batch
        dw, db = compute_gradients(w, b, X_batch, Y_batch)
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Store current w, b and loss
        history.append((w, b, mse_loss(w, b, X, Y)))  
    return history

# Run stochastic gradient descent and get the path of (w, b) updates
history_sgd = stochastic_gradient_descent(X, Y, learning_rate=0.03, iterations=100, batch_size=50)









# ---------------------------------------------- Visualizing the Loss Landscape ----------------------------------------------
# Create a meshgrid of w and b values to plot the loss landscape
w_vals = np.linspace(-2, 4, 100)
b_vals = np.linspace(-2, 4, 100)
W, B = np.meshgrid(w_vals, b_vals)
Z = np.array([[mse_loss(w, b, X, Y) for w, b in zip(row_w, row_b)] for row_w, row_b in zip(W, B)])

# Extract the path of parameters (w, b) and the corresponding losses
w_path_sgd = [h[0] for h in history_sgd]
b_path_sgd = [h[1] for h in history_sgd]
loss_path_sgd = [h[2] for h in history_sgd]

# Create subplots: one for the loss landscape with SGD path, and one for loss over iterations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot the contour map of the loss landscape on the first subplot
contour = ax1.contour(W, B, Z, levels=50, cmap='coolwarm')
ax1.clabel(contour, inline=True, fontsize=8)

# Plot the SGD path on the loss landscape
ax1.plot(w_path_sgd, b_path_sgd, 'o-', color='green', label='SGD Path', markersize=5)

# Highlight first three and last three steps
#ax1.plot(w_path_sgd[:3], b_path_sgd[:3], 'o-', color='blue', label='First 3 Steps')
#ax1.plot(w_path_sgd[-3:], b_path_sgd[-3:], 'o-', color='red', label='Last 3 Steps')

# Labels and legend for the first plot
ax1.set_title('SGD Path on Loss Landscape')
ax1.set_xlabel('Weight (w)')
ax1.set_ylabel('Bias (b)')
ax1.legend()

# Plot the loss over iterations on the second subplot
ax2.plot(loss_path_sgd, label='SGD Loss', color='green')
ax2.set_title('Loss over Iterations')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('MSE Loss')

# Highlight first three and last three steps in the loss curve
ax2.plot(range(3), loss_path_sgd[:3], 'o-', color='blue', label='First 3 Steps')
ax2.plot(range(len(loss_path_sgd)-3, len(loss_path_sgd)), loss_path_sgd[-3:], 'o-', color='red', label='Last 3 Steps')

# Add legend and show the plots
ax2.legend()
plt.tight_layout()
plt.show()