import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to compute a simple loss (quadratic) for demonstration
def loss_function(x):
    return x ** 2 + 10 * np.sin(x)

# Optimizer implementations
def gradient_descent(learning_rate, max_iter=100):
    x = np.random.uniform(-10, 10)
    history = [x]
    for _ in range(max_iter):
        grad = 2 * x + 10 * np.cos(x)  # Gradient of the loss function
        x -= learning_rate * grad
        history.append(x)
    return history

def momentum(learning_rate, beta, max_iter=100):
    x = np.random.uniform(-10, 10)
    velocity = 0
    history = [x]
    for _ in range(max_iter):
        grad = 2 * x + 10 * np.cos(x)
        velocity = beta * velocity + (1 - beta) * grad
        x -= learning_rate * velocity
        history.append(x)
    return history

def rmsprop(learning_rate, beta, epsilon, max_iter=100):
    x = np.random.uniform(-10, 10)
    grad_squared_avg = 0
    history = [x]
    for _ in range(max_iter):
        grad = 2 * x + 10 * np.cos(x)
        grad_squared_avg = beta * grad_squared_avg + (1 - beta) * grad ** 2
        x -= learning_rate / (np.sqrt(grad_squared_avg + epsilon)) * grad
        history.append(x)
    return history

def adam(learning_rate, beta1, beta2, epsilon, max_iter=100):
    x = np.random.uniform(-10, 10)
    m = 0
    v = 0
    t = 0
    history = [x]
    for _ in range(max_iter):
        t += 1
        grad = 2 * x + 10 * np.cos(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(x)
    return history

# Streamlit app interface
st.title("Optimizer Comparison")

st.write("This app compares different optimizers like Gradient Descent, Momentum, RMSProp, and Adam. "
         "Each optimizer works to minimize a simple quadratic loss function.")

# Sidebar for user inputs
st.sidebar.header("Optimizer Hyperparameters")

# Dropdown options for hyperparameters
learning_rate = st.sidebar.selectbox("Learning Rate", [0.001, 0.01, 0.1, 0.5, 1.0])
max_iter = st.sidebar.selectbox("Maximum Iterations", [50, 100, 200, 300])

# Momentum-specific inputs
beta_momentum = st.sidebar.selectbox("Momentum Beta", [0.5, 0.9, 0.95, 0.99])

# RMSProp-specific inputs
beta_rmsprop = st.sidebar.selectbox("RMSProp Beta", [0.5, 0.9, 0.95, 0.99])
epsilon_rmsprop = st.sidebar.selectbox("RMSProp Epsilon", [1e-8, 1e-6, 1e-4, 1e-2])

# Adam-specific inputs
beta1_adam = st.sidebar.selectbox("Adam Beta1", [0.5, 0.9, 0.95, 0.99])
beta2_adam = st.sidebar.selectbox("Adam Beta2", [0.9, 0.99, 0.999])
epsilon_adam = st.sidebar.selectbox("Adam Epsilon", [1e-8, 1e-6, 1e-4])

# Display the values selected
st.sidebar.subheader("Selected Values")
st.sidebar.write(f"Learning Rate: {learning_rate}")
st.sidebar.write(f"Max Iterations: {max_iter}")
st.sidebar.write(f"Momentum Beta: {beta_momentum}")
st.sidebar.write(f"RMSProp Beta: {beta_rmsprop}")
st.sidebar.write(f"RMSProp Epsilon: {epsilon_rmsprop}")
st.sidebar.write(f"Adam Beta1: {beta1_adam}")
st.sidebar.write(f"Adam Beta2: {beta2_adam}")
st.sidebar.write(f"Adam Epsilon: {epsilon_adam}")

# Run optimizations
st.subheader("Optimizer Results")

# Gradient Descent
gd_history = gradient_descent(learning_rate, max_iter)
# Momentum
momentum_history = momentum(learning_rate, beta_momentum, max_iter)
# RMSprop
rmsprop_history = rmsprop(learning_rate, beta_rmsprop, epsilon_rmsprop, max_iter)
# Adam
adam_history = adam(learning_rate, beta1_adam, beta2_adam, epsilon_adam, max_iter)

# Plotting the loss function and optimizer results
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Loss Function Plot
x_vals = np.linspace(-10, 10, 400)
y_vals = loss_function(x_vals)
ax[0].plot(x_vals, y_vals, label="Loss Function", color="black")
ax[0].set_title("Loss Function")
ax[0].set_xlabel("x")
ax[0].set_ylabel("Loss")
ax[0].grid(True)

# Optimizer Trajectories Plot
ax[1].plot(gd_history, label="Gradient Descent", color="blue")
ax[1].plot(momentum_history, label="Momentum", color="green")
ax[1].plot(rmsprop_history, label="RMSprop", color="red")
ax[1].plot(adam_history, label="Adam", color="purple")
ax[1].set_title("Optimizer Trajectories")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Parameter Value (x)")
ax[1].legend()
ax[1].grid(True)

st.pyplot(fig)

# Display the optimizer equations below
st.subheader("Optimizer Equations")

optimizer_equations = """
1. **Gradient Descent**:
   $$ x_{t+1} = x_t - \\alpha \\nabla f(x_t) $$  
   where \\( \\alpha \\) is the learning rate, and \\( \\nabla f(x_t) \\) is the gradient of the loss function.

2. **Momentum**:
   $$ v_{t+1} = \\beta v_t + (1 - \\beta) \\nabla f(x_t) $$  
   $$ x_{t+1} = x_t - \\alpha v_{t+1} $$  
   where \\( \\beta \\) is the momentum coefficient.

3. **RMSProp**:
   $$ v_{t+1} = \\beta v_t + (1 - \\beta) \\nabla f(x_t)^2 $$  
   $$ x_{t+1} = x_t - \\frac{\\alpha}{\\sqrt{v_{t+1} + \\epsilon}} \\nabla f(x_t) $$  
   where \\( \\beta \\) is the decay factor, and \\( \\epsilon \\) is a small constant to avoid division by zero.

4. **Adam**:
   $$ m_{t+1} = \\beta_1 m_t + (1 - \\beta_1) \\nabla f(x_t) $$  
   $$ v_{t+1} = \\beta_2 v_t + (1 - \\beta_2) \\nabla f(x_t)^2 $$  
   $$ m_{t+1}^{hat} = \\frac{m_{t+1}}{1 - \\beta_1^t} $$  
   $$ v_{t+1}^{hat} = \\frac{v_{t+1}}{1 - \\beta_2^t} $$  
   $$ x_{t+1} = x_t - \\frac{\\alpha m_{t+1}^{hat}}{\\sqrt{v_{t+1}^{hat}} + \\epsilon} $$  
   where $$( \\beta_1 \\)$$ and $$( \\beta_2 \\)$$ are decay rates for the first and second moment estimates, and $$( \\epsilon \\)$$ is a small constant.
"""
st.markdown(optimizer_equations)
