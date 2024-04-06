"""
This code defines a KMeans class that implements the K-means clustering algorithm. 
The class has methods for initialization, fitting the model to data, and visualizing the clusters. 
Detailed comments are provided to explain each part of the code, including parameters, attributes, 
and the functionality of each method.
"""

import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=2, max_iters=100):
        """
        Constructor method for KMeans class.

        Parameters:
        - k: int, default=2
            Number of clusters to form.
        - max_iters: int, default=100
            Maximum number of iterations for the algorithm.

        Attributes:
        - k: int
            Number of clusters.
        - max_iters: int
            Maximum number of iterations.
        - centroids: ndarray
            Array containing the centroids of the clusters.
        - clusters: list
            List containing indices of data points assigned to each cluster.
        """
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        """
        Fit method to perform K-means clustering on the input data.

        Parameters:
        - X: ndarray of shape (n_samples, n_features)
            Input data.

        Raises:
        - ValueError: If input data is empty.
        """
        if len(X) == 0:
            raise ValueError("Input data is empty")

        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from the input data
        random_sample_idx = np.random.choice(n_samples, self.k, replace=False)
        centroids = X[random_sample_idx]

        # Perform K-means iterations
        for _ in range(self.max_iters):
            # Initialize clusters list to store indices of data points for each cluster
            clusters = [[] for _ in range(self.k)]

            # Assign each data point to the nearest centroid
            for sample_idx, sample in enumerate(X):
                closest_centroid_idx = np.argmin(
                    np.linalg.norm(sample - centroids, axis=1))
                clusters[closest_centroid_idx].append(sample_idx)

            # Update centroids based on the mean of data points in each cluster
            prev_centroids = centroids.copy()
            for cluster_idx, cluster in enumerate(clusters):
                centroids[cluster_idx] = np.mean(X[cluster], axis=0)

            # Check for convergence
            if np.all(prev_centroids == centroids):
                break

        # Store centroids and clusters as attributes
        self.centroids = centroids
        self.clusters = clusters

    def visualize(self, X):
        """
        Visualization method to plot the clusters and centroids.

        Parameters:
        - X: ndarray of shape (n_samples, n_features)
            Input data.
        """
        # Create a new figure
        plt.figure(figsize=(8, 6))
        
        # Plot centroids and data points for each cluster
        for i in range(len(self.centroids)):
            plt.scatter(self.centroids[i][0], self.centroids[i][1], marker='o', color='black', s=200)
            plt.scatter(X[self.clusters[i]][:, 0], X[self.clusters[i]][:, 1], label=f'Cluster {i+1}')
        
        # Add labels and title
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-means Clustering')
        plt.legend()
        
        # Show plot
        plt.show()