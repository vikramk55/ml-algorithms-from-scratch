import unittest
from linear_regression import LinearRegression
import numpy as np

class TestLinearRegression(unittest.TestCase):
    def test_linear_regression_fit(self):
        # Create Linear Regression instance
        model = LinearRegression()
        # Generate some sample data for testing
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 3, 4])
        # Fit the model
        model.fit(X_train, y_train)
        # Assert that weights and bias are not None after fitting
        self.assertIsNotNone(model.weights)
        self.assertIsNotNone(model.bias)

    def test_linear_regression_predict(self):
        # Create Linear Regression instance
        model = LinearRegression()
        # Generate some sample data for testing
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 3, 4])
        # Fit the model
        model.fit(X_train, y_train)
        # Predict on new data
        X_test = np.array([[4], [5]])
        predictions = model.predict(X_test)
        # Assert that predictions are of the correct shape
        self.assertEqual(predictions.shape, (2,))

if __name__ == '__main__':
    unittest.main()