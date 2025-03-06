import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X = np.linspace(-3, 3, 50).reshape(-1, 1)
y_true = np.sin(X).ravel()
y = y_true + np.random.normal(0, 0.3, size=X.shape[0])

# Streamlit UI
st.title("Bias-Variance Tradeoff Demo")
st.sidebar.header("Model Settings")

degree = st.sidebar.slider("Polynomial Degree", 1, 15, 3)

# Fit polynomial regression model
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)
y_pred = model.predict(X)

# Compute bias, variance, and error
bias = np.mean((y_true - y_pred) ** 2)
variance = np.var(y_pred)
total_error = bias + variance

# Visualization
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X, y, label="Noisy Data", color="gray")
ax.plot(X, y_true, label="True Function", color="green", linewidth=2)
ax.plot(X, y_pred, label=f"Model (Degree {degree})", color="red", linestyle="dashed")
ax.legend()
ax.set_title("Model Fit")

st.pyplot(fig)

# Display Bias-Variance Stats
st.subheader("Bias-Variance Breakdown")
st.write(f"**Bias**: {bias:.4f}")
st.write(f"**Variance**: {variance:.4f}")
st.write(f"**Total Error**: {total_error:.4f}")

# Bias-Variance Tradeoff Plot
complexities = range(1, 16)
bias_vals, var_vals, err_vals = [], [], []

for d in complexities:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)

    bias_vals.append(np.mean((y_true - y_pred) ** 2))
    var_vals.append(np.var(y_pred))
    err_vals.append(bias_vals[-1] + var_vals[-1])

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(complexities, bias_vals, label="Bias^2", color="blue", linewidth=2)
ax2.plot(complexities, var_vals, label="Variance", color="red", linewidth=2)
ax2.plot(complexities, err_vals, label="Total Error", color="black", linestyle="dashed")
ax2.set_xlabel("Model Complexity (Polynomial Degree)")
ax2.set_ylabel("Error")
ax2.set_title("Bias-Variance Tradeoff")
ax2.legend()

st.pyplot(fig2)
