import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Set page layout
st.set_page_config(page_title="Bias-Variance Tradeoff", layout="wide")

# Title
st.title("📊 Bias-Variance Tradeoff Interactive Demo")

# Sidebar for user input
st.sidebar.header("🔧 Model Settings")
degree = st.sidebar.slider("Select Polynomial Degree", min_value=1, max_value=15, value=3, step=1)

# Generate synthetic dataset
np.random.seed(42)
X = np.linspace(-3, 3, 50).reshape(-1, 1)
y_true = np.sin(X).ravel()  # True function
y = y_true + np.random.normal(0, 0.3, size=X.shape[0])  # Noisy observations

# Train polynomial regression model
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)
y_pred = model.predict(X)

# Compute bias and variance
bias = np.mean((y_true - y_pred) ** 2)
variance = np.var(y_pred)
total_error = bias + variance

# Create layout columns
col1, col2 = st.columns(2)

# First plot: Model Fit
with col1:
    st.subheader("📈 Model Fit Visualization")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X, y, label="Noisy Data", color="gray", alpha=0.6)
    ax.plot(X, y_true, label="True Function", color="green", linewidth=2)
    ax.plot(X, y_pred, label=f"Model Fit (Degree {degree})", color="red", linestyle="dashed")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Model Fit Comparison")
    st.pyplot(fig)

# Bias-Variance Decomposition
with col2:
    st.subheader("📊 Bias-Variance Breakdown")
    st.markdown(f"**Bias²**: `{bias:.4f}`")
    st.markdown(f"**Variance**: `{variance:.4f}`")
    st.markdown(f"**Total Error**: `{total_error:.4f}`")
    
    if degree < 5:
        st.success("🔹 Low Complexity: High Bias, Low Variance")
    elif degree > 10:
        st.warning("🔸 High Complexity: Low Bias, High Variance")
    else:
        st.info("⚖️ Balanced Complexity")

# Bias-Variance Tradeoff Curve
st.subheader("📉 Bias-Variance Tradeoff")
complexities = np.arange(1, 16)
bias_vals, var_vals, err_vals = [], [], []

for d in complexities:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)

    bias_vals.append(np.mean((y_true - y_pred) ** 2))
    var_vals.append(np.var(y_pred))
    err_vals.append(bias_vals[-1] + var_vals[-1])

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(complexities, bias_vals, label="Bias²", color="blue", linewidth=2)
ax2.plot(complexities, var_vals, label="Variance", color="red", linewidth=2)
ax2.plot(complexities, err_vals, label="Total Error", color="black", linestyle="dashed")
ax2.axvline(degree, color="gray", linestyle="dotted", label=f"Current Degree: {degree}")
ax2.set_xlabel("Model Complexity (Polynomial Degree)")
ax2.set_ylabel("Error")
ax2.set_title("Bias-Variance Tradeoff Curve")
ax2.legend()

st.pyplot(fig2)

# Explanation
st.subheader("📌 Understanding the Tradeoff")
st.write(
    "The Bias-Variance tradeoff explains how model complexity affects prediction error:\n"
    "- **Low Complexity (Underfitting)** → High Bias, Low Variance.\n"
    "- **High Complexity (Overfitting)** → Low Bias, High Variance.\n"
    "- **Optimal Model** is found where Total Error is minimized."
)
