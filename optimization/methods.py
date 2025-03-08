import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim

# Set Streamlit App Title
st.title("Understanding Optimizers in Deep Learning")

# Optimizer Selection
optimizer_choice = st.selectbox(
    "Choose an Optimizer to Visualize",
    ["SGD", "Momentum", "RMSprop", "Adam", "Adagrad", "Nadam"]
)

# Optimizer Explanations
optimizer_explanations = {
    "SGD": "Stochastic Gradient Descent (SGD) updates weights using only a single sample (or mini-batch), leading to faster but noisier updates.",
    "Momentum": "Momentum SGD builds upon standard SGD by adding a 'velocity' term that helps smooth out updates and speed up convergence.",
    "RMSprop": "RMSprop adjusts the learning rate dynamically based on the magnitude of recent gradients, preventing large updates and improving stability.",
    "Adam": "Adam combines momentum and RMSprop, adapting learning rates per parameter for efficient training.",
    "Adagrad": "Adagrad scales learning rates based on past gradients, making it good for sparse data but leading to vanishingly small learning rates over time.",
    "Nadam": "Nadam is a combination of Adam and Nesterov Momentum, providing improved convergence speed."
}

# Display Optimizer Explanation
st.markdown(f"### How {optimizer_choice} Works")
st.write(optimizer_explanations[optimizer_choice])

# Parameter sliders
learning_rate = st.slider("Learning Rate:", 0.0001, 0.1, 0.01, 0.0001)
momentum = 0.9
beta_1, beta_2 = 0.9, 0.999

if optimizer_choice == "Momentum":
    momentum = st.slider("Momentum:", 0.0, 1.0, 0.9, 0.01)
elif optimizer_choice in ["Adam", "Nadam"]:
    beta_1 = st.slider("Beta 1:", 0.5, 0.99, 0.9, 0.01)
    beta_2 = st.slider("Beta 2:", 0.5, 0.999, 0.999, 0.001)

# Define a sample quadratic loss function using PyTorch
def loss_function(x, y):
    return x**2 + 3*torch.sin(2*x) + y**2 + 3*torch.sin(2*y)  # Use torch.sin

# Generate loss surface
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Convert X, Y to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Compute the Z values of the loss surface
Z = X_tensor**2 + 3*torch.sin(2*X_tensor) + Y_tensor**2 + 3*torch.sin(2*Y_tensor)

# PyTorch Optimization Simulation
x = torch.tensor([-4.0], requires_grad=True)  # Start at -4
y = torch.tensor([1.0], requires_grad=True)   # Start at 1

optimizer_dict = {
    "SGD": optim.SGD([x, y], lr=learning_rate),
    "Momentum": optim.SGD([x, y], lr=learning_rate, momentum=momentum),
    "RMSprop": optim.RMSprop([x, y], lr=learning_rate),
    "Adam": optim.Adam([x, y], lr=learning_rate, betas=(beta_1, beta_2)),
    "Adagrad": optim.Adagrad([x, y], lr=learning_rate),
    "Nadam": optim.NAdam([x, y], lr=learning_rate, betas=(beta_1, beta_2)),
}

optimizer = optimizer_dict[optimizer_choice]

# Optimization steps
x_history = []
y_history = []
z_history = []
for _ in range(30):
    optimizer.zero_grad()
    loss = loss_function(x, y)  # Combined loss for x and y
    loss.backward()
    optimizer.step()
    x_history.append(x.detach().numpy())  # Detach and convert to numpy
    y_history.append(y.detach().numpy())  # Detach and convert to numpy
    z_history.append(loss.detach().numpy())  # Detach and convert to numpy

# Create 2D plot for the loss function
fig2d, ax2d = plt.subplots(figsize=(8, 6))
loss_vals_2d = loss_function(torch.tensor(x_vals), torch.tensor(np.zeros_like(x_vals))).detach().numpy()
ax2d.plot(x_vals, loss_vals_2d, label="Loss Function (x only)", color="blue")
ax2d.scatter(x_history, z_history, color="red", marker="o", label="Optimizer Path")
ax2d.set_title("2D Loss Function with Optimizer Path")
ax2d.set_xlabel("X values")
ax2d.set_ylabel("Loss")
ax2d.legend()

# Create 3D plot for the loss surface
fig3d = plt.figure(figsize=(10, 7))
ax3d = fig3d.add_subplot(111, projection='3d')

# Plot the 3D loss surface
ax3d.plot_surface(X, Y, Z.detach().numpy(), cmap='viridis', alpha=0.7)

# Plot the optimizer path
ax3d.scatter(x_history, y_history, z_history, color="red", marker="o", label="Optimizer Path")

# Labels and title for the 3D plot
ax3d.set_title(f"3D Optimization Path of {optimizer_choice}")
ax3d.set_xlabel("X values")
ax3d.set_ylabel("Y values")
ax3d.set_zlabel("Loss")
ax3d.legend()

# Display both plots in Streamlit
st.pyplot(fig2d)  # 2D Loss Plot
st.pyplot(fig3d)  # 3D Loss Surface Plot
