import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit UI
st.set_page_config(page_title="Bias-Variance in Transformers", layout="wide")
st.title("🤖 Bias-Variance Tradeoff in Transformers")

# Sidebar for user input
st.sidebar.header("🔧 Model Settings")
num_heads = st.sidebar.slider("Number of Attention Heads", 1, 8, 2)
num_layers = st.sidebar.slider("Transformer Depth (Layers)", 1, 6, 2)
hidden_dim = st.sidebar.slider("Hidden Layer Size", 32, 256, 64)

# Generate synthetic data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y_true = np.sin(X).ravel()
y_noisy = y_true + np.random.normal(0, 0.3, size=X.shape[0])

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_noisy, dtype=torch.float32).to(device)

# Define a Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.fc(x)
        return x.squeeze()

# Initialize model
model = SimpleTransformer(input_dim=1, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()

# Predictions
model.eval()
y_pred = model(X_tensor).cpu().detach().numpy()

# Compute Bias, Variance, and Error
bias = np.mean((y_true - y_pred) ** 2)
variance = np.var(y_pred)
total_error = bias + variance

# Layout for visualization
col1, col2 = st.columns(2)

# Plot Model Fit
with col1:
    st.subheader("📈 Model Fit Visualization")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X, y_noisy, label="Noisy Data", color="gray", alpha=0.6)
    ax.plot(X, y_true, label="True Function", color="green", linewidth=2)
    ax.plot(X, y_pred, label=f"Transformer Model (Heads: {num_heads}, Layers: {num_layers})", color="red", linestyle="dashed")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Model Fit Comparison")
    st.pyplot(fig)

# Bias-Variance Breakdown
with col2:
    st.subheader("📊 Bias-Variance Breakdown")
    st.markdown(f"**Bias²**: `{bias:.4f}`")
    st.markdown(f"**Variance**: `{variance:.4f}`")
    st.markdown(f"**Total Error**: `{total_error:.4f}`")

    if num_layers < 2:
        st.success("🔹 Low Complexity: High Bias, Low Variance")
    elif num_layers > 4:
        st.warning("🔸 High Complexity: Low Bias, High Variance")
    else:
        st.info("⚖️ Balanced Complexity")

# Bias-Variance Tradeoff Curve
st.subheader("📉 Bias-Variance Tradeoff")
complexities = range(1, 7)
bias_vals, var_vals, err_vals = [], [], []

for layers in complexities:
    temp_model = SimpleTransformer(input_dim=1, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=layers).to(device)
    temp_optimizer = optim.Adam(temp_model.parameters(), lr=0.01)

    # Train temporary model
    for _ in range(100):
        temp_model.train()
        temp_optimizer.zero_grad()
        y_temp_pred = temp_model(X_tensor)
        loss = criterion(y_temp_pred, y_tensor)
        loss.backward()
        temp_optimizer.step()

    temp_model.eval()
    y_temp_pred = temp_model(X_tensor).cpu().detach().numpy()

    bias_vals.append(np.mean((y_true - y_temp_pred) ** 2))
    var_vals.append(np.var(y_temp_pred))
    err_vals.append(bias_vals[-1] + var_vals[-1])

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(complexities, bias_vals, label="Bias²", color="blue", linewidth=2)
ax2.plot(complexities, var_vals, label="Variance", color="red", linewidth=2)
ax2.plot(complexities, err_vals, label="Total Error", color="black", linestyle="dashed")
ax2.axvline(num_layers, color="gray", linestyle="dotted", label=f"Current Layers: {num_layers}")
ax2.set_xlabel("Transformer Layers")
ax2.set_ylabel("Error")
ax2.set_title("Bias-Variance Tradeoff Curve")
ax2.legend()

st.pyplot(fig2)

# Explanation
st.subheader("📌 Understanding the Tradeoff")
st.write(
    "The Bias-Variance tradeoff in Transformers works similarly to traditional ML models:\n"
    "- **Low Complexity (Few Layers)** → High Bias, Low Variance (Underfitting).\n"
    "- **High Complexity (More Layers, More Heads)** → Low Bias, High Variance (Overfitting).\n"
    "- **Optimal Model** balances Bias and Variance to minimize Total Error."
)
