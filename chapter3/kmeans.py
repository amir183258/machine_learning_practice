import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import mglearn
from sklearn.datasets import fetch_lfw_people
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


if __name__ == "__main__":
    X, y = make_blobs(random_state=170, n_samples=600)
    rng = np.random.RandomState(74)

    transformation = rng.normal(size=(2, 2))
    X = np.dot(X, transformation)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    y_pred = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm3)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='^', c=['b', 'r', 'g'], s=60, linewidth=2)
    plt.show()

