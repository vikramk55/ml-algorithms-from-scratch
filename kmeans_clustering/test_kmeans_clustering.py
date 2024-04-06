import unittest
from kmeans_clustering import KMeans
import numpy as np

class TestKMeans(unittest.TestCase):
    def test_kmeans_clustering(self):
        # Create KMeans instance
        kmeans = KMeans(k=2)
        # Generate some sample data for testing
        X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
        # Fit the model
        kmeans.fit(X)
        # Assert that clusters are assigned
        self.assertIsNotNone(kmeans.clusters)

if __name__ == '__main__':
    unittest.main()
