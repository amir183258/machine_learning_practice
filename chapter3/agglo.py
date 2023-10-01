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
from sklearn.cluster import AgglomerativeClustering

if __name__ == "__main__":
    X, y = make_blobs(random_state=1)

    agg = AgglomerativeClustering(n_clusters=3)
    assignment = agg.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=assignment, cmap=mglearn.cm3, s=60)
    plt.show()
