import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
    X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(X_train[:, 0], X_train[:, 1],
            c='b', label="training set", s=60)
    axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^',
            c='r', label="test set", s=60)
    axes[0].legend(loc="upper left")
    axes[0].set_title("original data")

    # Scaling part
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
            c='b', label="training set", s=60)
    axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^',
            c='r', label="test set", s=60)
    axes[1].set_title("scaled data")

    plt.show()
