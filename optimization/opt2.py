import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import LBFGS
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import asyncio

# Ensure an event loop is running
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Load dataset (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Streamlit App
st.title("First-Order vs Second-Order Optimizer Comparison")

# Select optimizer types
optimizer_choice_1st = st.selectbox("Choose First-Order Optimizer", ["SGD", "Adam", "RMSprop"])
optimizer_choice_2nd = st.selectbox("Choose Second-Order Optimizer", ["LBFGS"])

# Learning rate slider
learning_rate = st.slider("Learning Rate:", 0.0001, 0.1, 0.01, 0.0001)

# Additional hyperparameters
momentum = 0.9
beta_1 = 0.9
beta_2 = 0.999

if optimizer_choice_1st == "SGD":
    momentum = st.slider("Momentum (for SGD):", 0.0, 1.0, 0.9, 0.01)
elif optimizer_choice_1st in ["Adam", "RMSprop"]:
    beta_1 = st.slider("Beta 1:", 0.5, 0.99, 0.9, 0.01)
    beta_2 = st.slider("Beta 2:", 0.5, 0.999, 0.999, 0.001)

# Training settings
epochs = st.slider("Number of Epochs:", 1, 10, 3, 1)

# Create models
model_1st = SimpleNN()
model_2nd = SimpleNN()

# Create optimizers
optimizers = {
    "SGD": optim.SGD(model_1st.parameters(), lr=learning_rate, momentum=momentum),
    "Adam": optim.Adam(model_1st.parameters(), lr=learning_rate, betas=(beta_1, beta_2)),
    "RMSprop": optim.RMSprop(model_1st.parameters(), lr=learning_rate),
    "LBFGS": LBFGS(model_2nd.parameters(), lr=learning_rate)
}

optimizer_1st = optimizers[optimizer_choice_1st]
optimizer_2nd = optimizers[optimizer_choice_2nd]

# Loss function
criterion = nn.CrossEntropyLoss()

# Train the models
if st.button("Train Models"):
    loss_history_1st, loss_history_2nd = [], []

    # Train first-order optimizer model
    model_1st.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer_1st.zero_grad()
            outputs = model_1st(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_1st.step()
            total_loss += loss.item()
        loss_history_1st.append(total_loss / len(train_loader))

    # Train second-order optimizer model
    model_2nd.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            def closure():
                optimizer_2nd.zero_grad()
                outputs = model_2nd(images)
                loss = criterion(outputs, labels)
                loss.backward()
                return loss

            optimizer_2nd.step(closure)
            total_loss += closure().item()
        loss_history_2nd.append(total_loss / len(train_loader))

    # Plot the loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history_1st, label=f"{optimizer_choice_1st} (First-Order)", marker="o")
    ax.plot(loss_history_2nd, label=f"{optimizer_choice_2nd} (Second-Order)", marker="s")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Comparison: First-Order vs Second-Order Optimizer")
    ax.legend()
    st.pyplot(fig)