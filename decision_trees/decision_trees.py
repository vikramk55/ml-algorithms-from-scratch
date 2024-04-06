'''
In summary, this code defines a DecisionTree class with methods for fitting a decision tree model (fit),
growing the tree recursively (_grow_tree), finding the best split for a node (_find_best_split),
and calculating Gini impurity (_gini_impurity). The class allows for customization of the maximum depth
of the tree and utilizes NumPy for array operations.
'''

import numpy as np  # Importing NumPy for array operations

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth  # Maximum depth of the decision tree

    def fit(self, X, y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data is empty")  # Check if input data is empty

        self.n_classes = len(np.unique(y))  # Number of unique classes in target variable y
        self.n_features = X.shape[1]  # Number of features in input data X
        self.tree = self._grow_tree(X, y)  # Grow the decision tree

    def _grow_tree(self, X, y, depth=0):
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(n_samples_per_class)  # Predicted class based on majority vote
        node = {'predicted_class': predicted_class}  # Node represents a split in the decision tree

        if depth < self.max_depth:  # Check if maximum depth has not been reached
            feature_idx, threshold = self._find_best_split(X, y)  # Find the best feature and threshold to split on
            if feature_idx is not None:  # If a split is found
                indices_left = X[:, feature_idx] < threshold  # Indices for left child nodes
                X_left, y_left = X[indices_left], y[indices_left]  # Data for left child nodes
                X_right, y_right = X[~indices_left], y[~indices_left]  # Data for right child nodes
                node['feature_idx'] = feature_idx  # Index of feature used for splitting
                node['threshold'] = threshold  # Threshold value for splitting
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)  # Recursive call to grow left subtree
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)  # Recursive call to grow right subtree

        return node  # Return the decision tree node

    def _find_best_split(self, X, y):
        best_gini = 1  # Initialize best Gini impurity to maximum
        best_feature_idx, best_threshold = None, None  # Initialize best feature index and threshold

        for feature_idx in range(self.n_features):  # Loop over each feature
            thresholds = np.unique(X[:, feature_idx])  # Unique values of the feature
            for threshold in thresholds:  # Loop over each unique value of the feature
                indices_left = X[:, feature_idx] < threshold  # Indices for left child nodes
                y_left = y[indices_left]  # Target values for left child nodes
                y_right = y[~indices_left]  # Target values for right child nodes
                # Calculate Gini impurity for the split
                gini = (len(y_left) * self._gini_impurity(y_left) +
                        len(y_right) * self._gini_impurity(y_right)) / len(y)
                if gini < best_gini:  # Update best split if Gini impurity is lower
                    best_gini = gini
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold  # Return best feature index and threshold

    def _gini_impurity(self, y):
        p = np.bincount(y) / len(y)  # Calculate class probabilities
        return 1 - np.sum(p ** 2)  # Calculate Gini impurity for the class distribution