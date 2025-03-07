# Bias-Variance Tradeoff Demo

This repository contains a series of interactive demos that illustrate the **Bias-Variance Tradeoff** in machine learning models. The demos are built using **Streamlit** and cover different types of models, including **Polynomial Regression** and **Transformers**. The goal is to help users understand how model complexity affects bias, variance, and overall prediction error.

## Files Overview

### 1. `base.py`
- **Description**: This file demonstrates the bias-variance tradeoff using a **Polynomial Regression** model. It allows users to adjust the polynomial degree and visualize how the model fits the data, along with the corresponding bias, variance, and total error.
- **Key Features**:
  - Interactive slider to adjust the polynomial degree.
  - Visualization of the model fit, true function, and noisy data.
  - Bias, variance, and total error breakdown.
  - Bias-variance tradeoff curve for different polynomial degrees.

### 2. `base1.py`
- **Description**: An enhanced version of `base.py` with a more user-friendly interface and additional explanations. It also includes a **Bias-Variance Tradeoff Curve** and provides feedback on the model's complexity.
- **Key Features**:
  - Improved UI with a wide layout and better visualizations.
  - Feedback on model complexity (low, balanced, high).
  - Detailed explanation of the bias-variance tradeoff.
  - Interactive slider for polynomial degree selection.

### 3. `base3.py`
- **Description**: This file extends the concept of bias-variance tradeoff to **Transformer models**. It allows users to adjust the number of attention heads, transformer layers, and hidden layer size to see how these parameters affect the model's performance.
- **Key Features**:
  - Interactive sliders for adjusting transformer parameters (number of heads, layers, and hidden dimension).
  - Visualization of the transformer model fit compared to the true function and noisy data.
  - Bias, variance, and total error breakdown.
  - Bias-variance tradeoff curve for different transformer layers.
  - Explanation of how bias-variance tradeoff applies to transformers.

## How to Run the Demos

1. **Install Dependencies**:
   Ensure you have the required Python packages installed. You can install them using the following command:
   ```bash
   pip install streamlit numpy matplotlib scikit-learn torch
   ```

2. **Run the Streamlit App**:
   To run any of the demos, navigate to the directory containing the file and use the following command:
   ```bash
   streamlit run base.py
   ```
   Replace `base.py` with `base1.py` or `base3.py` to run the other demos.

3. **Interact with the Demo**:
   - Adjust the sliders in the sidebar to change model parameters.
   - Observe how the model fit, bias, variance, and total error change with different settings.
   - Explore the bias-variance tradeoff curve to understand the relationship between model complexity and error.

## Key Concepts

### Bias-Variance Tradeoff
- **Bias**: Error due to overly simplistic assumptions in the learning algorithm. High bias can cause an algorithm to miss relevant relations between features and target outputs (underfitting).
- **Variance**: Error due to the model's sensitivity to small fluctuations in the training set. High variance can cause overfitting, where the model captures noise instead of the underlying pattern.
- **Total Error**: The sum of bias and variance. The goal is to find a model complexity that minimizes the total error.

### Model Complexity
- **Low Complexity**: Models with low complexity (e.g., low polynomial degree or few transformer layers) tend to have high bias and low variance, leading to underfitting.
- **High Complexity**: Models with high complexity (e.g., high polynomial degree or many transformer layers) tend to have low bias and high variance, leading to overfitting.
- **Optimal Complexity**: The ideal model complexity balances bias and variance to minimize the total error.

## Contributing
Feel free to contribute to this repository by opening issues or submitting pull requests. If you have any suggestions for improving the demos or adding new features, please let us know!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Enjoy exploring the bias-variance tradeoff with these interactive demos! 🚀
