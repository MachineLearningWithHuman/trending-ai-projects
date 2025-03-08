import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define loss functions
def mse_loss(y_true, y_pred):
    return (y_pred - y_true) ** 2

def mae_loss(y_true, y_pred):
    return np.abs(y_pred - y_true)

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    return np.where(np.abs(error) <= delta, 0.5 * error ** 2, delta * (np.abs(error) - 0.5 * delta))

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-9
    return -(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# Streamlit UI
st.title("Loss Function Visualizer (2D & 3D)")

loss_type = st.selectbox("Select Loss Function", ["MSE", "MAE", "Huber", "Binary Cross-Entropy"])

y_true = st.slider("Actual Value (y_true)", -2.0, 2.0, 0.0)
y_pred = np.linspace(-2, 2, 100)

equations = {
    "MSE": "MSE = (y_{pred} - y_{true})^2",
    "MAE": "MAE = |y_{pred} - y_{true}|",
    "Huber": "Huber = \begin{cases} 0.5(y_{pred} - y_{true})^2, & |y_{pred} - y_{true}| \leq \delta \\ \delta (|y_{pred} - y_{true}| - 0.5\delta), & \text{otherwise} \end{cases}",
    "Binary Cross-Entropy": "BCE = - [y_{true} \log(y_{pred}) + (1 - y_{true}) \log(1 - y_{pred})]"
}

st.latex(equations[loss_type])

if loss_type == "MSE":
    loss_values = mse_loss(y_true, y_pred)
elif loss_type == "MAE":
    loss_values = mae_loss(y_true, y_pred)
elif loss_type == "Huber":
    delta = st.slider("Delta (Huber Loss)", 0.1, 2.0, 1.0)
    loss_values = huber_loss(y_true, y_pred, delta)
elif loss_type == "Binary Cross-Entropy":
    y_pred = np.linspace(0.01, 0.99, 100)
    loss_values = binary_cross_entropy(y_true, y_pred)

# 2D Plot
fig, ax = plt.subplots()
ax.plot(y_pred, loss_values, label=loss_type, color='b')
ax.set_title(f"{loss_type} Loss (2D)")
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Loss")
ax.grid()
st.pyplot(fig)

# 3D Plot
fig_3d = plt.figure(figsize=(8, 6))
ax_3d = fig_3d.add_subplot(111, projection='3d')
Y_pred, Y_true = np.meshgrid(y_pred, np.linspace(-2, 2, 100))

if loss_type == "MSE":
    Z = mse_loss(Y_true, Y_pred)
elif loss_type == "MAE":
    Z = mae_loss(Y_true, Y_pred)
elif loss_type == "Huber":
    Z = huber_loss(Y_true, Y_pred, delta)
elif loss_type == "Binary Cross-Entropy":
    Y_pred, Y_true = np.meshgrid(np.linspace(0.01, 0.99, 100), [0, 1])
    Z = binary_cross_entropy(Y_true, Y_pred)

ax_3d.plot_surface(Y_pred, Y_true, Z, cmap='viridis')
ax_3d.set_xlabel("Predicted Value")
ax_3d.set_ylabel("Actual Value")
ax_3d.set_zlabel("Loss")
ax_3d.set_title(f"{loss_type} Loss (3D)")
st.pyplot(fig_3d)
