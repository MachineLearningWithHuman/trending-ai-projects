import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Reduce dataset size for faster feedback
x_train, y_train = x_train[:1000], y_train[:1000]  # Using only a subset

# Define a simple neural network
@st.cache(allow_output_mutation=True)
def build_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Streamlit App
st.title("Neural Network Optimizer Tuning App")

# Optimizer Selection
optimizer_choice = st.selectbox("Choose an Optimizer:", 
                                ["SGD", "Momentum", "RMSprop", "Adam", "Adagrad", "Nadam"])

# Learning rate slider
learning_rate = st.slider("Learning Rate:", 0.0001, 0.1, 0.01, 0.0001)

# Additional hyperparameters
momentum = 0.9
beta_1 = 0.9
beta_2 = 0.999

if optimizer_choice == "Momentum":
    momentum = st.slider("Momentum:", 0.0, 1.0, 0.9, 0.01)
elif optimizer_choice in ["Adam", "Nadam"]:
    beta_1 = st.slider("Beta 1:", 0.5, 0.99, 0.9, 0.01)
    beta_2 = st.slider("Beta 2:", 0.5, 0.999, 0.999, 0.001)

# Create optimizer
optimizers = {
    "SGD": keras.optimizers.SGD(learning_rate=learning_rate),
    "Momentum": keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
    "RMSprop": keras.optimizers.RMSprop(learning_rate=learning_rate),
    "Adam": keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2),
    "Adagrad": keras.optimizers.Adagrad(learning_rate=learning_rate),
    "Nadam": keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2),
}

optimizer = optimizers[optimizer_choice]

# Training Configuration
epochs = st.slider("Number of Epochs:", 1, 10, 3, 1)
batch_size = st.slider("Batch Size:", 16, 128, 32, 16)

# Train Model
if st.button("Train Model"):
    with st.spinner('Training the model...'):
        model = build_model()
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # Create a progress bar
        progress_bar = st.progress(0)

        # Train Model
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0,
                           callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: progress_bar.progress((epoch + 1) / epochs))])

        # After training, remove the progress bar
        progress_bar.empty()

    # Plot Loss & Accuracy
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title("Loss Over Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Accuracy plot
    ax[1].plot(history.history['accuracy'], label='Train Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    ax[1].set_title("Accuracy Over Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    st.pyplot(fig)

    # Show final accuracy
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    st.write(f"**Final Training Accuracy:** {final_train_acc:.4f}")
    st.write(f"**Final Validation Accuracy:** {final_val_acc:.4f}")
