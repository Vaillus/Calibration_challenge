import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, x0, k):
    """
    Sigmoid function: f(x) = 1 / (1 + exp(-k * (x - x0)))
    
    Args:
        x: input values
        x0: sigmoid center parameter
        k: sigmoid steepness parameter
    
    Returns:
        sigmoid values
    """
    return 1 / (1 + np.exp(-k * (x - x0)))

# Parameters
x0 = 1.245
k = 152

# Create x values from 0 to 1
x = np.linspace(0, 1, 1000)

# Calculate sigmoid values
y = sigmoid(x, x0, k)

# Calculate ymax (value at x = 1)
ymax = sigmoid(1, x0, k)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label=f'Sigmoid (x0={x0}, k={k})')
plt.axhline(y=ymax, color='r', linestyle='--', alpha=0.7, label=f'ymax = {ymax:.6f}')
plt.axvline(x=1, color='g', linestyle='--', alpha=0.7, label='x = 1')

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Sigmoid Function (x0={x0}, k={k})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 1)

# Add annotation for ymax
plt.annotate(f'ymax = {ymax:.6f}', xy=(1, ymax), xytext=(0.8, ymax + 0.1),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

plt.show()

print(f"Sigmoid parameters:")
print(f"x0 = {x0}")
print(f"k = {k}")
print(f"ymax (value at x=1) = {ymax:.6f}") 