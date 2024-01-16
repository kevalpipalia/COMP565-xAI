import numpy as np


def compute_deepLIFT_contributions(X, w1, w2):
    """
    Compute the DeepLIFT contributions of the input features based on linear and rescale rules.
    """
    # Reference (zero) input
    reference_input = np.zeros_like(X)

    # Forward pass
    hidden_layer = relu(np.dot(X, w1))
    output = np.dot(hidden_layer, w2)

    # Compute difference-from-reference for each neuron in hidden layer
    delta_h = hidden_layer - relu(np.dot(reference_input, w1))

    # Compute difference-from-reference for the output
    delta_y = output - np.dot(relu(np.dot(reference_input, w1)), w2)

    # Backpropagation for linear rule from output to hidden layer
    m_h_y = w2.reshape(-1, 1)

    # Applying rescale rule for backpropagation from hidden layer to input
    m_i_h = delta_h / np.where(delta_h == 0, 1e-7, delta_h)

    # Propagate multipliers
    m_i_y = np.dot(m_i_h, m_h_y)

    # Compute the contributions for each input
    C_i_y = X * m_i_y
    return C_i_y


# Define ReLU function
def relu(x):
    return np.maximum(0, x)


# Test the implementation
N = 100  # Number of samples
num_features = 10  # Number of input features

# Simulate input features
input_features = np.random.binomial(1, 0.5, (N, num_features))

# Set the weights to be the same as the ground truth
w1 = np.ones((10, 2))
w2 = np.ones((2, 1))

contributions = compute_deepLIFT_contributions(input_features, w1, w2)

print("DeepLIFT Contributions:\n", contributions)

import matplotlib.pyplot as plt


def plot_scatterplot(contributions):
    plt.figure(figsize=(10, 6))
    for i in range(contributions.shape[1]):
        plt.scatter(
            contributions[:, i], [i] * contributions.shape[0], label=f"Feature {i+1}"
        )

    plt.axhline(
        y=2.5, color="r", linestyle="--"
    )  # This line denotes the separation between the causal features (first 3) and the rest.
    plt.title("DeepLIFT Feature Importance Scatterplot")
    plt.xlabel("Feature Importance Value")
    plt.ylabel("Feature Index")
    plt.yticks(range(contributions.shape[1]), range(1, contributions.shape[1] + 1))
    plt.legend()
    plt.show()


plot_scatterplot(contributions)


def plot_global_feature_attributions(contributions):
    plt.figure(figsize=(10, 6))
    mean_contributions = np.mean(np.abs(contributions), axis=0)
    plt.barh(range(1, contributions.shape[1] + 1), mean_contributions)
    plt.title("Global Feature Attributions (Mean DeepLIFT Values)")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Absolute DeepLIFT Value")
    plt.xticks(range(1, contributions.shape[1] + 1))
    plt.show()


plot_global_feature_attributions(contributions)
