"""
This code defines a LinearRegression class with methods to initialize the model, fit it to data,
make predictions, evaluate performance, and visualize results. Each method is documented with its
purpose, parameters, and return values using comments.
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        """
        Initialize the Linear Regression model.

        Parameters:
        - lr (float): Learning rate for gradient descent (default: 0.01).
        - n_iters (int): Number of iterations for gradient descent (default: 1000).
        """
        self.lr = lr  # Learning rate for gradient descent
        self.n_iters = n_iters  # Number of iterations for gradient descent
        self.weights = None  # Initialize weights to None
        self.bias = None  # Initialize bias to None

    def fit(self, X, y):
        """
        Fit the Linear Regression model to the training data.

        Parameters:
        - X (array-like): Input features of shape (n_samples, n_features).
        - y (array-like): Target values of shape (n_samples,).

        Raises:
        - ValueError: If input data is empty.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data is empty")

        n_samples, n_features = X.shape  # Get number of samples and features
        self.weights = np.zeros(n_features)  # Initialize weights to zeros
        self.bias = 0  # Initialize bias to zero

        # Gradient Descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias  # Predicted values
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))  # Gradient of weights
            db = (1/n_samples) * np.sum(y_predicted - y)  # Gradient of bias

            # Update weights and bias using gradient descent
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict target values for input features.

        Parameters:
        - X (array-like): Input features of shape (n_samples, n_features).

        Returns:
        - y_predicted (array-like): Predicted target values of shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias  # Return predicted values

    def evaluate(self, X, y):
        """
        Evaluate the model's performance using Mean Squared Error (MSE).

        Parameters:
        - X (array-like): Input features of shape (n_samples, n_features).
        - y (array-like): True target values of shape (n_samples,).

        Returns:
        - mse (float): Mean Squared Error.
        """
        y_predicted = self.predict(X)  # Predicted target values
        mse = np.mean((y_predicted - y) ** 2)  # Mean Squared Error
        return mse

    def visualize(self, X, y):
        """
        Visualize the actual vs. predicted values.

        Parameters:
        - X (array-like): Input features of shape (n_samples, n_features).
        - y (array-like): True target values of shape (n_samples,).
        """
        y_predicted = self.predict(X)  # Predicted target values
        plt.scatter(y, y_predicted)  # Scatter plot of actual vs. predicted values
        plt.xlabel('Actual Values')  # X-axis label
        plt.ylabel('Predicted Values')  # Y-axis label
        plt.title('Actual vs. Predicted Values')  # Title of the plot
        plt.show()  # Display the plot