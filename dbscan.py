import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate a sample dataset with 3 clusters and some noise
X, _ = make_blobs(n_samples=500, cluster_std=0.5, random_state=0)

# Initialize DBSCAN with chosen parameters
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='CMRmap', s=30)
plt.title("DBSCAN Clustering on make_blobs Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
