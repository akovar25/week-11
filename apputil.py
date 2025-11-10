import numpy as np
from sklearn.cluster import KMeans
from time import time
import seaborn as sns
import pandas as pd

def kmeans(X, k):
    """
    Perform k-means clustering on a numerical NumPy array X.

    Raises:
        ValueError: If X is not a 2D NumPy array.

    Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features)
        k (int): Number of clusters

    Returns:
        tuple: (centroids, labels)
            - centroids: np.ndarray of shape (k, n_features)
            - labels: np.ndarray of shape (n_samples,)
    """
    model = KMeans(n_clusters=k, n_init='auto', random_state=42)
    model.fit(X)
    return model.cluster_centers_, model.labels_


# Load and cache the numeric portion of the diamonds dataset
diamonds = sns.load_dataset('diamonds')
numeric_diamonds = diamonds.select_dtypes(include='number')

def kmeans_diamonds(n, k):
    """
    Run k-means clustering on the first n rows of the numeric diamonds dataset.

    Raises:
        ValueError: If n or k is invalid

    Parameters:
        n (int): Number of rows to use
        k (int): Number of clusters

    Returns:
        tuple: (centroids, labels)
    """
    X = numeric_diamonds.iloc[:n].to_numpy()
    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    """
    Time the kmeans_diamonds function over n_iter runs.

    Raises:
        ValueError: If n_iter is not a positive integer
    
    Parameters:
        n (int): Number of rows
        k (int): Number of clusters
        n_iter (int): Number of iterations

    Returns:
        float: Average runtime in seconds
    """
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        times.append(time() - start)
    return sum(times) / n_iter