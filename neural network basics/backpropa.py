import streamlit as st
import plotly.graph_objects as go
import numpy as np
#https://github.com/graviraja/MLOps-Basics
# Title of the app
st.title("Backpropagation Algorithm Visualization with Node Visualizations")

# Introduction
st.write("""
This app visualizes the backpropagation algorithm in a simple neural network, including node visualizations and fluctuating gradients.
We will use a single hidden layer neural network to demonstrate the process.
""")

# Define the neural network architecture
st.sidebar.header("Neural Network Parameters")
input_size = st.sidebar.slider("Input Size", 1, 10, 2)
hidden_size = st.sidebar.slider("Hidden Layer Size", 1, 10, 3)
output_size = st.sidebar.slider("Output Size", 1, 10, 1)

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
    return Z1, A1, Z2, A2

# Backpropagation
def backpropagation(X, Y, Z1, A1, Z2, A2, W1, W2, b1, b2):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (1 - np.tanh(Z1) ** 2)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m
    return dW1, db1, dW2, db2

# Generate some sample data
X = np.random.randn(100, input_size)
Y = np.random.randint(2, size=(100, output_size))

# Perform forward and backward pass
Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
dW1, db1, dW2, db2 = backpropagation(X, Y, Z1, A1, Z2, A2, W1, W2, b1, b2)

# Visualize the forward pass
st.header("Forward Propagation")
st.write("""
### Input to Hidden Layer
- **Z1**: Linear transformation of input
- **A1**: Activation (tanh) applied to Z1
""")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=np.arange(len(Z1.flatten())), y=Z1.flatten(), mode='lines+markers', name='Z1'))
fig1.add_trace(go.Scatter(x=np.arange(len(A1.flatten())), y=A1.flatten(), mode='lines+markers', name='A1'))
st.plotly_chart(fig1)

st.write("""
### Hidden Layer to Output
- **Z2**: Linear transformation of hidden layer
- **A2**: Activation (sigmoid) applied to Z2
""")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=np.arange(len(Z2.flatten())), y=Z2.flatten(), mode='lines+markers', name='Z2'))
fig2.add_trace(go.Scatter(x=np.arange(len(A2.flatten())), y=A2.flatten(), mode='lines+markers', name='A2'))
st.plotly_chart(fig2)

# Visualize the backward pass
st.header("Backpropagation")
st.write("""
### Gradients
- **dW1**: Gradient of loss with respect to W1
- **db1**: Gradient of loss with respect to b1
- **dW2**: Gradient of loss with respect to W2
- **db2**: Gradient of loss with respect to b2
""")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=np.arange(len(dW1.flatten())), y=dW1.flatten(), mode='lines+markers', name='dW1'))
fig3.add_trace(go.Scatter(x=np.arange(len(db1.flatten())), y=db1.flatten(), mode='lines+markers', name='db1'))
fig3.add_trace(go.Scatter(x=np.arange(len(dW2.flatten())), y=dW2.flatten(), mode='lines+markers', name='dW2'))
fig3.add_trace(go.Scatter(x=np.arange(len(db2.flatten())), y=db2.flatten(), mode='lines+markers', name='db2'))
st.plotly_chart(fig3)

# Node Visualizations
st.header("Node Visualizations")
st.write("""
### Node Activations and Gradients
- **Z1**: Linear transformation of input
- **A1**: Activation (tanh) applied to Z1
- **Z2**: Linear transformation of hidden layer
- **A2**: Activation (sigmoid) applied to Z2
""")

# Create a figure for node activations
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=np.arange(len(Z1.flatten())), y=Z1.flatten(), mode='lines+markers', name='Z1'))
fig4.add_trace(go.Scatter(x=np.arange(len(A1.flatten())), y=A1.flatten(), mode='lines+markers', name='A1'))
fig4.add_trace(go.Scatter(x=np.arange(len(Z2.flatten())), y=Z2.flatten(), mode='lines+markers', name='Z2'))
fig4.add_trace(go.Scatter(x=np.arange(len(A2.flatten())), y=A2.flatten(), mode='lines+markers', name='A2'))
st.plotly_chart(fig4)

# Fluctuating Gradients
st.header("Fluctuating Gradients")
st.write("""
### Gradient Fluctuations Over Time
- **dW1**: Gradient of loss with respect to W1
- **db1**: Gradient of loss with respect to b1
- **dW2**: Gradient of loss with respect to W2
- **db2**: Gradient of loss with respect to b2
""")

# Simulate gradient fluctuations over time
time_steps = 100
gradient_history = {
    'dW1': np.zeros((time_steps, dW1.size)),
    'db1': np.zeros((time_steps, db1.size)),
    'dW2': np.zeros((time_steps, dW2.size)),
    'db2': np.zeros((time_steps, db2.size))
}

for t in range(time_steps):
    gradient_history['dW1'][t] = dW1.flatten() + np.random.normal(0, 0.1, dW1.size)
    gradient_history['db1'][t] = db1.flatten() + np.random.normal(0, 0.1, db1.size)
    gradient_history['dW2'][t] = dW2.flatten() + np.random.normal(0, 0.1, dW2.size)
    gradient_history['db2'][t] = db2.flatten() + np.random.normal(0, 0.1, db2.size)

# Create a figure for gradient fluctuations
fig5 = go.Figure()
for i in range(dW1.size):
    fig5.add_trace(go.Scatter(x=np.arange(time_steps), y=gradient_history['dW1'][:, i], mode='lines', name=f'dW1_{i}'))
for i in range(db1.size):
    fig5.add_trace(go.Scatter(x=np.arange(time_steps), y=gradient_history['db1'][:, i], mode='lines', name=f'db1_{i}'))
for i in range(dW2.size):
    fig5.add_trace(go.Scatter(x=np.arange(time_steps), y=gradient_history['dW2'][:, i], mode='lines', name=f'dW2_{i}'))
for i in range(db2.size):
    fig5.add_trace(go.Scatter(x=np.arange(time_steps), y=gradient_history['db2'][:, i], mode='lines', name=f'db2_{i}'))
st.plotly_chart(fig5)

# Equations
st.header("Equations")
st.write("""
### Forward Propagation
- **Z1 = X * W1 + b1**
- **A1 = tanh(Z1)**
- **Z2 = A1 * W2 + b2**
- **A2 = sigmoid(Z2)**

### Backpropagation
- **dZ2 = A2 - Y**
- **dW2 = (A1.T * dZ2) / m**
- **db2 = sum(dZ2) / m**
- **dA1 = dZ2 * W2.T**
- **dZ1 = dA1 * (1 - tanh(Z1)^2)**
- **dW1 = (X.T * dZ1) / m**
- **db1 = sum(dZ1) / m**
""")

# Run the app
if __name__ == "__main__":
    st.write("App is running...")