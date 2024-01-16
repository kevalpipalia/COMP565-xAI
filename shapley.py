# Question 2 : Shapley Values

from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from itertools import chain, combinations
from math import factorial
import shap
import matplotlib.pyplot as plt


rf_regressor = RandomForestRegressor(
    random_state=42
)  # Define the random forest regressor
rf_regressor.fit(X_train, y_train)

# Step 3: Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

# Step 4: Calculate Pearson correlation coefficient
corr_coefficient, _ = pearsonr(y_test, y_pred)

# Print the correlation coefficient
print("Pearson Correlation Coefficient:", corr_coefficient)

# Step 3: Calculate Shapley values for each feature

# Compute Shapley values
# explainer = shap.TreeExplainer(rf_regressor)
# shap_values = explainer.shap_values(X_test)

# # Print Shapley values for the first test instance
# print("Shapley values using the shap library:", shap_values[0])

num_samples = len(X_test)
num_features = X_test.shape[1]
shapley_values = np.zeros(num_features)


def subsets_without_feature(features, d):
    """
    Return all subsets of features that do not contain d.
    """
    # Remove the feature d from the list of features
    features = [f for f in features if f != d]

    # Use chain and combinations to generate all possible subsets of the features
    return list(
        chain(*map(lambda x: combinations(features, x), range(0, len(features) + 1)))
    )


def compute_shapley_value(d, model, X_test):
    F = list(range(X_test.shape[1]))
    phi_d = 0

    for S in subsets_without_feature(F, d)[1:]:
        x_S = X_test.copy()
        x_S_union_d = X_test.copy()

        for j in list(range(10)):
            if j not in list(S):
                np.random.shuffle(x_S[:, j])
            if j not in list(S) and j == d:
                np.random.shuffle(x_S_union_d[:, j])

        f_S = model.predict(x_S)
        f_S_union_d = model.predict(x_S_union_d)

        difference = f_S_union_d - f_S
        comb_term = (
            factorial(len(S)) * factorial(len(F) - len(S) - 1) / factorial(len(F))
        )
        phi_d += comb_term * difference

    return phi_d


shap_values = [
    compute_shapley_value(d, rf_regressor, X_test) for d in range(X_test.shape[1])
]

# print("Shapley values using shap implementation:", shap_values)


# Step 4: Plotting Shapley values
def plot_shapley_scatterplot(shap_values, num_samples, num_features):
    plt.figure(figsize=(10, 6))
    for i in range(num_features):
        plt.scatter(shap_values[i], [i] * num_samples, label=f"Feature {i+1}")

    plt.axhline(
        y=2.5, color="r", linestyle="--"
    )  # This line denotes the separation between the causal features (first 3) and the rest.
    plt.title("Shapley Value Scatterplot")
    plt.xlabel("Shapley Value")
    plt.ylabel("Feature Index")
    plt.yticks(range(num_features), range(1, num_features + 1))
    plt.legend(loc="upper right")
    plt.show()


plot_shapley_scatterplot(shap_values, num_samples, num_features)


def plot_global_shapley_attributions(shap_values, num_features):
    plt.figure(figsize=(10, 6))
    mean_shapley_values = np.mean(np.abs(shap_values), axis=0)
    plt.bar(range(1, num_features + 1), mean_shapley_values)
    plt.title("Global Feature Attributions (Mean Shapley Values)")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Absolute Shapley Value")
    plt.xticks(range(1, num_features + 1))
    plt.show()


plot_global_shapley_attributions(shap_values, num_features)


# Question 2 : Shapley Values

from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from itertools import chain, combinations
from math import factorial
import shap
import matplotlib.pyplot as plt


rf_regressor = RandomForestRegressor(
    random_state=42
)  # Define the random forest regressor
rf_regressor.fit(X_train, y_train)

# Step 3: Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

# Step 4: Calculate Pearson correlation coefficient
corr_coefficient, _ = pearsonr(y_test, y_pred)

# Print the correlation coefficient
print("Pearson Correlation Coefficient:", corr_coefficient)

# Step 3: Calculate Shapley values for each feature

# Compute Shapley values
# explainer = shap.TreeExplainer(rf_regressor)
# shap_values = explainer.shap_values(X_test)

# # Print Shapley values for the first test instance
# print("Shapley values using the shap library:", shap_values[0])

num_samples = len(X_test)
num_features = X_test.shape[1]
shapley_values = np.zeros(num_features)


def subsets_without_feature(features, d):
    """
    Return all subsets of features that do not contain d.
    """
    # Remove the feature d from the list of features
    features = [f for f in features if f != d]

    # Use chain and combinations to generate all possible subsets of the features
    return list(
        chain(*map(lambda x: combinations(features, x), range(0, len(features) + 1)))
    )


def compute_shapley_value(d, model, X_test):
    F = list(range(X_test.shape[1]))
    phi_d = 0

    for S in subsets_without_feature(F, d)[1:]:
        x_S = X_test.copy()
        x_S_union_d = X_test.copy()

        for j in list(range(10)):
            if j not in list(S):
                np.random.shuffle(x_S[:, j])
            if j not in list(S) and j == d:
                np.random.shuffle(x_S_union_d[:, j])

        f_S = model.predict(x_S)
        f_S_union_d = model.predict(x_S_union_d)

        difference = f_S_union_d - f_S
        comb_term = (
            factorial(len(S)) * factorial(len(F) - len(S) - 1) / factorial(len(F))
        )
        phi_d += comb_term * difference

    return phi_d


shap_values = [
    compute_shapley_value(d, rf_regressor, X_test) for d in range(X_test.shape[1])
]

# print("Shapley values using shap implementation:", shap_values)


# Step 4: Plotting Shapley values
def plot_shapley_scatterplot(shap_values, num_samples, num_features):
    plt.figure(figsize=(10, 6))
    for i in range(num_features):
        plt.scatter(shap_values[i], [i] * num_samples, label=f"Feature {i+1}")

    plt.axhline(
        y=2.5, color="r", linestyle="--"
    )  # This line denotes the separation between the causal features (first 3) and the rest.
    plt.title("Shapley Value Scatterplot")
    plt.xlabel("Shapley Value")
    plt.ylabel("Feature Index")
    plt.yticks(range(num_features), range(1, num_features + 1))
    plt.legend(loc="upper right")
    plt.show()


plot_shapley_scatterplot(shap_values, num_samples, num_features)


def plot_global_shapley_attributions(shap_values, num_features):
    plt.figure(figsize=(10, 6))
    mean_shapley_values = np.mean(np.abs(shap_values), axis=0)
    plt.bar(range(1, num_features + 1), mean_shapley_values)
    plt.title("Global Feature Attributions (Mean Shapley Values)")
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Absolute Shapley Value")
    plt.xticks(range(1, num_features + 1))
    plt.show()


plot_global_shapley_attributions(shap_values, num_features)
