import numpy as np
import matplotlib.pyplot as plt

# Define the loss function
def loss_function(x, y):
    return 0.5 * x**2 + 10 * y**2

# Compute gradients
def gradients(x, y):
    grad_x = x
    grad_y = 20 * y
    return grad_x, grad_y

# Gradient Descent (without momentum)
def gradient_descent(lr, steps):
    x, y = 8.0, 8.0  # Starting point
    path = [(x, y)]
    for _ in range(steps):
        grad_x, grad_y = gradients(x, y)
        x -= lr * grad_x
        y -= lr * grad_y
        path.append((x, y))
    return path

# Gradient Descent with Momentum
def gradient_descent_momentum(lr, steps, momentum):
    x, y = 8.0, 8.0  # Starting point
    path = [(x, y)]
    v_x, v_y = 0, 0  # Initialize velocities
    for _ in range(steps):
        grad_x, grad_y = gradients(x, y)
        v_x = momentum * v_x + lr * grad_x
        v_y = momentum * v_y + lr * grad_y
        x -= v_x
        y -= v_y
        path.append((x, y))
    return path

# Settings for visualization
learning_rate = 0.1
steps = 14
momentum = 0.9

# Get paths for both methods
gd_path = gradient_descent(learning_rate, steps)
momentum_path = gradient_descent_momentum(learning_rate, steps, momentum)

# Create meshgrid for loss surface
x_vals = np.linspace(-9, 9, 400)
y_vals = np.linspace(-9, 9, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = loss_function(X, Y)

# Plotting
plt.figure(figsize=(10, 5))

# Plot the loss surface
plt.contour(X, Y, Z, levels=50, cmap='coolwarm')

# Plot Gradient Descent Path
gd_path = np.array(gd_path)
plt.plot(gd_path[:, 0], gd_path[:, 1], 'o-', label='Gradient Descent', color='blue')

# Plot Gradient Descent with Momentum Path
momentum_path = np.array(momentum_path)
plt.plot(momentum_path[:, 0], momentum_path[:, 1], 'o-', label='Momentum', color='green')

# Plot labels
plt.title('Comparison of Gradient Descent and Momentum')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()