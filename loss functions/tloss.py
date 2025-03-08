import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Generate dummy trading data
def generate_dummy_data(n=100):
    np.random.seed(42)
    data = {
        "Open": np.random.uniform(100, 200, n),
        "High": np.random.uniform(150, 250, n),
        "Low": np.random.uniform(90, 190, n),
        "Close": np.random.uniform(100, 200, n),
        "Volume": np.random.randint(1000, 5000, n),
    }
    return pd.DataFrame(data)

df = generate_dummy_data()

# Define loss functions
def mse_loss(y_true, y_pred):
    return (y_pred - y_true) ** 2

def mae_loss(y_true, y_pred):
    return np.abs(y_pred - y_true)

def log_cosh_loss(y_true, y_pred):
    return np.log(np.cosh(y_pred - y_true))

def smooth_huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    return np.where(np.abs(error) <= delta, 0.5 * error ** 2, delta * (np.abs(error) - 0.5 * delta))

# Streamlit UI
st.title("Trading Loss Function Visualizer")

st.dataframe(df.head())

loss_type = st.selectbox("Select Loss Function", ["MSE", "MAE", "Log-Cosh", "Smooth Huber"])

y_true = df["Close"].values
y_pred = df["Open"].values

if loss_type == "MSE":
    loss_values = mse_loss(y_true, y_pred)
elif loss_type == "MAE":
    loss_values = mae_loss(y_true, y_pred)
elif loss_type == "Log-Cosh":
    loss_values = log_cosh_loss(y_true, y_pred)
elif loss_type == "Smooth Huber":
    delta = st.slider("Delta (Smooth Huber Loss)", 0.1, 2.0, 1.0)
    loss_values = smooth_huber_loss(y_true, y_pred, delta)

# 2D Loss Plot
fig, ax = plt.subplots()
ax.plot(loss_values, label=loss_type, color='b')
ax.set_title(f"{loss_type} Loss over Time")
ax.set_xlabel("Time (Trading Days)")
ax.set_ylabel("Loss")
ax.grid()
st.pyplot(fig)

# 3D Loss Plot
fig_3d = plt.figure(figsize=(8, 6))
ax_3d = fig_3d.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(len(y_true)), range(len(y_true)))
Z = np.tile(loss_values, (len(y_true), 1))
ax_3d.plot_surface(X, Y, Z, cmap='viridis')
ax_3d.set_xlabel("Time")
ax_3d.set_ylabel("Sample Index")
ax_3d.set_zlabel("Loss")
ax_3d.set_title(f"{loss_type} Loss (3D View)")
st.pyplot(fig_3d)
