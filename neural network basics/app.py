import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sympy import symbols, Eq, latex, exp

def draw_neural_net(ax, layers):
    G = nx.DiGraph()
    pos = {}
    node_count = 0
    labels = {}  # Dictionary to store node labels
    
    for layer_idx, layer_size in enumerate(layers):
        for i in range(layer_size):
            G.add_node(node_count)
            pos[node_count] = (layer_idx, -i)
            # Assign labels like a11, a12, etc.
            if layer_idx == 0:
                labels[node_count] = f"x{i+1}"  # Input layer nodes
            else:
                labels[node_count] = f"a{layer_idx}{i+1}"  # Hidden/output layer nodes
            node_count += 1
    
    node_count = 0
    for layer_idx in range(len(layers) - 1):
        for i in range(layers[layer_idx]):
            for j in range(layers[layer_idx + 1]):
                G.add_edge(node_count + i, node_count + layers[layer_idx] + j)
        node_count += layers[layer_idx]
    
    # Draw the graph with labels
    nx.draw(G, pos, ax=ax, with_labels=True, labels=labels, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')

def forward_propagation(inputs, weights, biases):
    activations = [np.array(inputs)]
    
    for w, b in zip(weights, biases):
        z = np.dot(w, activations[-1]) + b
        a = 1 / (1 + np.exp(-z))  # Sigmoid activation
        activations.append(a)
    
    return activations

st.title("Neural Network Forward Propagation")
st.sidebar.header("Network Parameters")

input_size = st.sidebar.slider("Input Layer Size", 1, 5, 3, key="input_size")
hidden_size = st.sidebar.slider("Hidden Layer Size", 1, 5, 3, key="hidden_size")
output_size = st.sidebar.slider("Output Layer Size", 1, 5, 1, key="output_size")

layers = [input_size, hidden_size, output_size]

st.sidebar.subheader("Input Values")
inputs = [st.sidebar.number_input(f"x{i+1}", value=1.0, key=f"input_{i}") for i in range(input_size)]

st.sidebar.subheader("Weights & Biases")
weights = []
biases = []

for i in range(len(layers) - 1):
    w = np.array([[st.sidebar.number_input(f"W{i+1}_{j+1}_{k+1}", value=0.5, key=f"W_{i+1}_{j+1}_{k+1}") for j in range(layers[i])] for k in range(layers[i + 1])])
    b = np.array([st.sidebar.number_input(f"b{i+1}_{j+1}", value=0.0, key=f"b_{i+1}_{j+1}") for j in range(layers[i + 1])])
    weights.append(w)
    biases.append(b)

activations = forward_propagation(inputs, weights, biases)

fig, ax = plt.subplots(figsize=(5, 5))
draw_neural_net(ax, layers)
st.pyplot(fig)

st.subheader("Forward Propagation Steps")

for i, a in enumerate(activations):
    st.write(f"Layer {i+1} Activations: {a}")

st.subheader("Equations")
symbols_list = [[symbols(f"x{j+1}") for j in range(input_size)]]
for i in range(len(layers) - 1):
    layer_symbols = []
    for j in range(layers[i + 1]):
        # Use raw symbols for calculations
        z = sum(symbols_list[-1][k] * weights[i][j, k] for k in range(layers[i])) + biases[i][j]
        a = 1 / (1 + exp(-z))  # Sigmoid activation
        eq = Eq(symbols(f"a{i+1}_{j+1}"), a)
        layer_symbols.append(symbols(f"a{i+1}_{j+1}"))  # Store the symbol, not the equation
        st.latex(latex(eq))
    symbols_list.append(layer_symbols)