import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define custom loss functions
def log_cosh_loss(y_true, y_pred):
    return np.log(np.cosh(y_pred - y_true))

def hinge_loss(y_true, y_pred):
    return np.maximum(0, 1 - y_true * y_pred)

def kl_divergence(p, q):
    epsilon = 1e-9  # Small value to avoid division by zero
    return np.sum(p * np.log((p + epsilon) / (q + epsilon)))

# Streamlit UI
st.title("Custom Loss Function Visualizer (2D & 3D)")

loss_type = st.selectbox("Select Loss Function", ["Log-Cosh", "Hinge", "KL Divergence"])

y_true = st.slider("Actual Value (y_true)", -2.0, 2.0, 0.0)
y_pred = np.linspace(-2, 2, 100)

# Display equation
equations = {
    "Log-Cosh": r"Log-Cosh = log(cosh(y_{pred} - y_{true}))",
    "Hinge": r"Hinge = max(0, 1 - y_{true} \cdot y_{pred})",
    "KL Divergence": r"KL(p || q) = \sum p \log(\frac{p}{q})"
}

st.latex(equations[loss_type])

if loss_type == "Log-Cosh":
    loss_values = log_cosh_loss(y_true, y_pred)
elif loss_type == "Hinge":
    loss_values = hinge_loss(y_true, y_pred)
elif loss_type == "KL Divergence":
    p = np.linspace(0.01, 0.99, 100)
    q = np.linspace(0.02, 1.0, 100)
    loss_values = [kl_divergence(np.array([p_i]), np.array([q_i])) for p_i, q_i in zip(p, q)]
    y_pred = p  # Adjust x-axis to probability values

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

if loss_type == "Log-Cosh":
    Y_pred, Y_true = np.meshgrid(y_pred, np.linspace(-2, 2, 100))
    Z = log_cosh_loss(Y_true, Y_pred)
elif loss_type == "Hinge":
    Y_pred, Y_true = np.meshgrid(y_pred, np.linspace(-2, 2, 100))
    Z = hinge_loss(Y_true, Y_pred)
elif loss_type == "KL Divergence":
    p = np.linspace(0.01, 0.99, 100)
    q = np.linspace(0.02, 1.0, 100)
    P, Q = np.meshgrid(p, q)
    Z = np.array([[kl_divergence(np.array([p_i]), np.array([q_j])) for p_i, q_j in zip(P.ravel(), Q.ravel())]]).reshape(P.shape)
    Y_pred, Y_true = P, Q  # Use P and Q for the 3D plot axes

ax_3d.plot_surface(Y_pred, Y_true, Z, cmap='viridis')
ax_3d.set_xlabel("Predicted Value")
ax_3d.set_ylabel("Actual Value")
ax_3d.set_zlabel("Loss")
ax_3d.set_title(f"{loss_type} Loss (3D)")
st.pyplot(fig_3d)