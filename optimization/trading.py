import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import yfinance as yf

# Set Streamlit App Title
st.title("Trading Strategy Optimization with Different Optimizers")

# Select an Optimizer
optimizer_choice = st.selectbox(
    "Choose an Optimizer",
    ["SGD", "Momentum", "RMSprop", "Adam", "Adagrad", "Nadam"]
)

# Optimizer Explanations
optimizer_explanations = {
    "SGD": "SGD (Stochastic Gradient Descent) updates weights using only a single sample (or mini-batch), leading to faster but noisier updates.",
    "Momentum": "Momentum SGD builds upon standard SGD by adding a 'velocity' term that helps smooth out updates and speed up convergence.",
    "RMSprop": "RMSprop adjusts the learning rate dynamically based on the magnitude of recent gradients, preventing large updates and improving stability.",
    "Adam": "Adam combines momentum and RMSprop, adapting learning rates per parameter for efficient training.",
    "Adagrad": "Adagrad scales learning rates based on past gradients, making it good for sparse data but leading to vanishingly small learning rates over time.",
    "Nadam": "Nadam is a combination of Adam and Nesterov Momentum, providing improved convergence speed."
}

# Display Optimizer Explanation
st.markdown(f"### How {optimizer_choice} Works")
st.write(optimizer_explanations[optimizer_choice])

# Hyperparameter Sliders
learning_rate = st.slider("Learning Rate:", 0.0001, 0.1, 0.01, 0.0001)
momentum = 0.9
beta_1, beta_2 = 0.9, 0.999

if optimizer_choice == "Momentum":
    momentum = st.slider("Momentum:", 0.0, 1.0, 0.9, 0.01)
elif optimizer_choice in ["Adam", "Nadam"]:
    beta_1 = st.slider("Beta 1:", 0.5, 0.99, 0.9, 0.01)
    beta_2 = st.slider("Beta 2:", 0.5, 0.999, 0.999, 0.001)

# Load Stock Data
st.subheader("Stock Data for Strategy Optimization")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA)", "AAPL")
data = yf.download(stock_symbol, start="2023-01-01", end="2024-01-01")
# Clean up column names to ensure there are no issues
data["Returns"] = data["Close"].pct_change()
st.line_chart(data["Close"])

# Define a simple trading strategy: moving average crossover
short_window = st.slider("Short Moving Average Window", 5, 50, 10)
long_window = st.slider("Long Moving Average Window", 20, 200, 50)

data["Short_MA"] = data["Close"].rolling(window=short_window).mean()
data["Long_MA"] = data["Close"].rolling(window=long_window).mean()
data["Signal"] = np.where(data["Short_MA"] > data["Long_MA"], 1, -1)
data["Strategy_Returns"] = data["Signal"].shift(1) * data["Returns"]

# Show strategy performance
st.line_chart(data["Returns"].cumsum())
st.line_chart(data["Strategy_Returns"].cumsum())

# Optimize Trading Strategy using the Selected Optimizer
st.subheader("Optimizing the Strategy Parameters")

# Define a loss function (negative cumulative returns)
def trading_loss(params):
    short, long = params
    short, long = int(short), int(long)
    if short >= long:
        return torch.tensor(1e6, requires_grad=True)  # Penalize invalid parameters
    
    data["Short_MA"] = data["Close"].rolling(window=short).mean()
    data["Long_MA"] = data["Close"].rolling(window=long).mean()
    data["Signal"] = np.where(data["Short_MA"] > data["Long_MA"], 1, -1)
    data["Strategy_Returns"] = data["Signal"].shift(1) * data["Returns"]
    
    # Ensure the tensor requires gradients
    return -torch.tensor(data["Strategy_Returns"].sum(), dtype=torch.float32, requires_grad=True)

# Optimization using PyTorch
params = torch.tensor([short_window, long_window], dtype=torch.float32, requires_grad=True)

optimizer_dict = {
    "SGD": optim.SGD([params], lr=learning_rate),
    "Momentum": optim.SGD([params], lr=learning_rate, momentum=momentum),
    "RMSprop": optim.RMSprop([params], lr=learning_rate),
    "Adam": optim.Adam([params], lr=learning_rate, betas=(beta_1, beta_2)),
    "Adagrad": optim.Adagrad([params], lr=learning_rate),
    "Nadam": optim.NAdam([params], lr=learning_rate, betas=(beta_1, beta_2)),
}

optimizer = optimizer_dict[optimizer_choice]

# Run optimization
losses = []
for _ in range(50):
    optimizer.zero_grad()
    loss = trading_loss(params)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plot loss convergence
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(losses, label=f"{optimizer_choice} Optimization Path", marker="o")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss (Negative Returns)")
ax.set_title(f"Optimizer Performance: {optimizer_choice}")
ax.legend()
st.pyplot(fig)

# Display optimized parameters
st.subheader("Optimized Strategy Parameters")
st.write(f"Optimized Short MA Window: {int(params[0].item())}")
st.write(f"Optimized Long MA Window: {int(params[1].item())}")

# Apply optimized strategy
data["Short_MA"] = data["Close"].rolling(window=int(params[0].item())).mean()
data["Long_MA"] = data["Close"].rolling(window=int(params[1].item())).mean()
data["Signal"] = np.where(data["Short_MA"] > data["Long_MA"], 1, -1)
data["Optimized Strategy Returns"] = data["Signal"].shift(1) * data["Returns"]

# Show final strategy performance
st.subheader("Final Optimized Strategy Performance")
# Calculate cumulative returns
cumulative_returns = data[["Optimized Strategy Returns", "Returns"]].cumsum()

# Drop NaN values (optional, if needed)
cumulative_returns = cumulative_returns.dropna()

# Create a Matplotlib plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cumulative_returns.index, cumulative_returns["Optimized Strategy Returns"], label="Optimized Strategy Returns", color="blue")
ax.plot(cumulative_returns.index, cumulative_returns["Returns"], label="Market Returns", color="green")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Returns")
ax.set_title("Cumulative Returns: Optimized Strategy vs Market")
ax.legend()
ax.grid(True)

# Display the plot in Streamlit
st.pyplot(fig)