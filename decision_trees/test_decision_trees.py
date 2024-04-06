import unittest
from decision_trees import DecisionTree
import numpy as np

class TestDecisionTrees(unittest.TestCase):
    def test_decision_tree(self):
        # Create DecisionTree instance
        decision_tree = DecisionTree(max_depth=3)
        # Generate some sample data for testing
        X = np.array([[2.771244718,1.784783929],
                      [1.728571309,1.169761413],
                      [3.678319846,2.81281357],
                      [3.961043357,2.61995032],
                      [2.999208922,2.209014212],
                      [7.497545867,3.162953546],
                      [9.00220326,3.339047188],
                      [7.444542326,0.476683375],
                      [10.12493903,3.234550982],
                      [6.642287351,3.319983761]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # Fit the model
        decision_tree.fit(X, y)
        # Assert that tree is built
        self.assertIsNotNone(decision_tree.root)

if __name__ == '__main__':
    unittest.main()
